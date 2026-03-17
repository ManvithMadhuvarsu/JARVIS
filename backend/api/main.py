"""
JARVIS FastAPI Backend
"""
import asyncio
import logging
import os
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from config import settings
from agents.jarvis_agent import chat, stream_chat, jarvis_agent
from memory.vector_store import vector_store
from ingestion.engine import (
    ingest_file, ingest_directory, ingest_all_chat_exports
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="JARVIS API",
    description="Personal RAG AI Assistant",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Models ───────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"
    stream: bool = True

class IngestRequest(BaseModel):
    directory: Optional[str] = None
    chatgpt_export: Optional[str] = None
    cursor_dir: Optional[str] = None
    claude_export: Optional[str] = None
    gemini_dir: Optional[str] = None

class SearchRequest(BaseModel):
    query: str
    collections: Optional[list[str]] = None
    source_filter: Optional[str] = None
    top_k: int = 8


# ─── Endpoints ────────────────────────────────────────────────────

@app.get("/health")
async def health():
    stats = vector_store.get_stats()
    total = sum(s["count"] for s in stats.values())
    return {
        "status": "online",
        "total_documents": total,
        "collections": stats,
        "llm": settings.llm_model,
    }


@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    """Non-streaming chat."""
    try:
        response = await chat(request.message, request.session_id)
        return {"response": response, "session_id": request.session_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/stream")
async def chat_stream_endpoint(request: ChatRequest):
    """Server-Sent Events streaming chat."""
    async def event_stream():
        try:
            async for chunk in stream_chat(request.message, request.session_id):
                yield f"data: {chunk}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: [ERROR] {str(e)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """WebSocket for real-time bidirectional chat."""
    await websocket.accept()
    logger.info(f"WebSocket connected: {session_id}")

    try:
        while True:
            data = await websocket.receive_json()
            message = data.get("message", "")

            if not message:
                continue

            # Stream response back chunk by chunk
            async for chunk in stream_chat(message, session_id):
                await websocket.send_json({"type": "chunk", "content": chunk})

            await websocket.send_json({"type": "done"})

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {session_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.send_json({"type": "error", "content": str(e)})


@app.post("/search")
async def search_endpoint(request: SearchRequest):
    """Direct vector search without LLM."""
    filters = {}
    if request.source_filter:
        filters["source_type"] = request.source_filter

    docs = vector_store.search(
        query=request.query,
        collections=request.collections,
        top_k=request.top_k,
        filters=filters if filters else None
    )

    return {
        "query": request.query,
        "results": [
            {
                "content": doc.page_content,
                "source": doc.metadata.get("source", ""),
                "source_type": doc.metadata.get("source_type", ""),
                "platform": doc.metadata.get("platform", ""),
                "date": doc.metadata.get("date", ""),
                "score": doc.metadata.get("_score", 0),
                "collection": doc.metadata.get("_collection", ""),
            }
            for doc in docs
        ]
    }


@app.post("/ingest")
async def ingest_endpoint(request: IngestRequest, background_tasks: BackgroundTasks):
    """Trigger ingestion of files or chat exports."""

    async def run_ingestion():
        # Chat exports
        exports = {}
        if request.chatgpt_export:
            exports["chatgpt"] = request.chatgpt_export
        if request.cursor_dir:
            exports["cursor"] = request.cursor_dir
        if request.claude_export:
            exports["claude"] = request.claude_export
        if request.gemini_dir:
            exports["gemini"] = request.gemini_dir

        if exports:
            stats = await ingest_all_chat_exports(exports)
            logger.info(f"Chat ingestion: {stats}")

        # File directory
        if request.directory:
            stats = await ingest_directory(request.directory)
            logger.info(f"File ingestion: {stats}")

    background_tasks.add_task(run_ingestion)
    return {"status": "ingestion_started", "message": "Ingestion running in background"}


@app.get("/ingest/auto")
async def auto_ingest(background_tasks: BackgroundTasks):
    """Run auto-detection and ingest from common locations."""

    async def smart_ingest():
        home = Path(settings.user_home)
        stats = {}

        # Common Cursor locations
        cursor_paths = [
            home / ".cursor" / "logs",
            home / "Library" / "Application Support" / "Cursor" / "logs",
            home / ".config" / "Cursor" / "logs",
        ]
        for cp in cursor_paths:
            if cp.exists():
                from ingestion.engine import parse_cursor_export
                docs = parse_cursor_export(str(cp))
                count = await vector_store.add_documents("chat_history", docs)
                stats["cursor"] = count
                break

        # Ingest home directory files (code, docs)
        for folder in ["Documents", "Projects", "Code", "repos", "work", "Desktop"]:
            target = home / folder
            if target.exists():
                result = await ingest_directory(str(target))
                stats[folder] = result

        logger.info(f"Auto-ingest complete: {stats}")

    background_tasks.add_task(smart_ingest)
    return {"status": "auto_ingestion_started"}


@app.get("/memory/stats")
async def memory_stats():
    """Get knowledge base statistics."""
    return vector_store.get_stats()


@app.get("/memory/history/{session_id}")
async def get_session_history(session_id: str):
    """Get conversation history for a session."""
    try:
        from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
        config = {"configurable": {"thread_id": session_id}}
        state = jarvis_agent.get_state(config)
        messages = state.values.get("messages", [])
        
        history = []
        for m in messages[-20:]:
            if isinstance(m, HumanMessage):
                role = "user"
            elif isinstance(m, AIMessage):
                role = "assistant"
            elif isinstance(m, SystemMessage):
                role = "system"
            else:
                role = "unknown"
            
            history.append({
                "role": role,
                "content": m.content if hasattr(m, "content") else str(m),
                "type": type(m).__name__
            })
            
        return {
            "session_id": session_id,
            "message_count": len(messages),
            "messages": history
        }
    except Exception as e:
        return {"session_id": session_id, "messages": [], "error": str(e)}


@app.get("/files/browse")
async def browse_files(path: str = "~", pattern: str = "*"):
    """Browse filesystem."""
    try:
        target = Path(path).expanduser()
        if not target.exists():
            raise HTTPException(status_code=404, detail=f"Path not found: {path}")

        items = []
        for item in sorted(target.iterdir())[:200]:
            items.append({
                "name": item.name,
                "path": str(item),
                "is_dir": item.is_dir(),
                "size": item.stat().st_size if item.is_file() else 0,
                "modified": item.stat().st_mtime,
            })

        return {"path": str(target), "items": items}
    except PermissionError:
        raise HTTPException(status_code=403, detail="Permission denied")
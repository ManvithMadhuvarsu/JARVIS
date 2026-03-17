"""
JARVIS Agent — LangGraph-powered personal AI assistant.

Capabilities:
  - RAG across all your ingested data
  - File system access (read, list)
  - Terminal command execution (sandboxed)
  - Web search
  - Persistent memory
  - Streaming responses
"""
import asyncio
import subprocess
import os
import logging
from pathlib import Path
from datetime import datetime
from typing import TypedDict, Annotated
import operator

from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from config import settings
from memory.vector_store import vector_store

logger = logging.getLogger(__name__)


# ─── Agent State ─────────────────────────────────────────────────

class JarvisState(TypedDict):
    messages: Annotated[list, operator.add]
    retrieved_context: str
    tool_results: list


# ─── Tools ───────────────────────────────────────────────────────

@tool
def search_my_knowledge(query: str, source_filter: str = "") -> str:
    """
    Search across ALL personal data — chat histories, code, documents.
    Use this for anything about past work, previous conversations, or stored files.

    Args:
        query: What to search for
        source_filter: Optional: 'chatgpt', 'cursor', 'claude', 'gemini', 'local_file'
    """
    filters = {}
    if source_filter:
        filters["source_type"] = source_filter

    docs = vector_store.search(
        query=query,
        top_k=8,
        filters=filters if filters else None
    )

    if not docs:
        return "No relevant information found in the personal knowledge base."

    results = []
    for i, doc in enumerate(docs, 1):
        meta = doc.metadata
        source = meta.get("source_type", "unknown")
        platform = meta.get("platform", meta.get("project", ""))
        date = meta.get("date", "")[:10] if meta.get("date") else ""

        header = f"[{i}] Source: {source}"
        if platform:
            header += f" | {platform}"
        if date:
            header += f" | {date}"

        results.append(f"{header}\n{doc.page_content[:600]}")

    return "\n\n---\n\n".join(results)


@tool
def read_file(file_path: str) -> str:
    """
    Read the contents of any file on the system.

    Args:
        file_path: Absolute or ~ path to the file
    """
    try:
        path = Path(file_path).expanduser().resolve()
        if not path.exists():
            return f"File not found: {file_path}"
        if not path.is_file():
            return f"Path is a directory, not a file: {file_path}"
        if path.stat().st_size > 100 * 1024:
            return f"File too large to display ({path.stat().st_size:,} bytes). Use search_my_knowledge instead."
        return path.read_text(errors="replace")
    except PermissionError:
        return f"Permission denied: {file_path}"
    except Exception as e:
        return f"Error reading file: {e}"


@tool
def list_directory(dir_path: str = "~", pattern: str = "*") -> str:
    """
    List files in a directory, optionally filtered by pattern.

    Args:
        dir_path: Directory path (default: home directory)
        pattern: Glob pattern e.g. '*.py', '*.md'
    """
    try:
        path = Path(dir_path).expanduser().resolve()
        if not path.exists():
            return f"Directory not found: {dir_path}"
        if not path.is_dir():
            return f"Path is a file, not a directory: {dir_path}"

        items = sorted(path.glob(pattern))[:100]
        if not items:
            return f"No files matching '{pattern}' in {dir_path}"

        lines = []
        for item in items:
            try:
                size = f"{item.stat().st_size:,}B" if item.is_file() else "DIR"
                lines.append(f"{'📁' if item.is_dir() else '📄'} {item.name} ({size})")
            except PermissionError:
                lines.append(f"🔒 {item.name} (no permission)")

        return f"Contents of {path} ({len(items)} items):\n" + "\n".join(lines)
    except PermissionError:
        return f"Permission denied: {dir_path}"
    except Exception as e:
        return f"Error listing directory: {e}"


@tool
def run_terminal_command(command: str) -> str:
    """
    Execute a terminal command and return output.
    Good for: git status, ls, ps, system info, running scripts.
    Only safe read-oriented commands — never destructive.

    Args:
        command: Shell command to run
    """
    if not settings.enable_system_exec:
        return "System execution is disabled. Set ENABLE_SYSTEM_EXEC=true in .env to enable."

    BLOCKED = [
        "rm -rf", "sudo rm", "mkfs", "dd if=",
        ":(){ :|:& };:", "curl | bash", "wget | sh",
        "> /dev/", "chmod 777", "chown root"
    ]
    if any(b in command for b in BLOCKED):
        return f"Blocked: potentially destructive command detected — '{command}'"

    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=30,
            env={**os.environ, "TERM": "dumb"},
            cwd=str(Path(settings.user_home).expanduser())
        )
        output = (result.stdout or "") + (result.stderr or "")
        return output[:3000] if output.strip() else f"Command completed (exit code {result.returncode})"
    except subprocess.TimeoutExpired:
        return "Command timed out after 30 seconds."
    except Exception as e:
        return f"Error running command: {e}"


@tool
def search_web(query: str) -> str:
    """
    Search the web for current information not in the knowledge base.

    Args:
        query: Search query string
    """
    if not settings.enable_web_search:
        return "Web search is disabled. Set ENABLE_WEB_SEARCH=true in .env to enable."
    # FIX #19: DuckDuckGoSearchRun frequently raises RatelimitException.
    # Added 3-attempt retry with exponential backoff.
    import time
    last_err = ""
    for attempt in range(3):
        try:
            result = DuckDuckGoSearchRun().run(query)
            if result:
                return result
        except Exception as e:
            last_err = str(e)
            if attempt < 2:
                time.sleep(2 ** attempt)
    return (
        f"Web search unavailable after 3 attempts (likely DuckDuckGo rate limit). "
        f"Try using search_my_knowledge instead. Error: {last_err}"
    )


@tool
def remember_this(content: str, tags: str = "") -> str:
    """
    Explicitly save something important to long-term memory.
    Use when user says 'remember this', 'save this', or shares key info.

    Args:
        content: The content to remember
        tags: Comma-separated tags for easier retrieval later
    """
    # FIX: was using deprecated asyncio.get_event_loop().run_until_complete()
    # which raises RuntimeError when called inside a running event loop.
    # Correct approach: run sync via the qdrant client directly.
    from langchain_core.documents import Document

    doc = Document(
        page_content=content,
        metadata={
            "source": "explicit_memory",
            "source_type": "user_note",
            "tags": tags,
            "date": datetime.utcnow().isoformat(),
            "project": "jarvis_memory",
        }
    )

    # FIX #3: QdrantVectorStore only has aadd_documents() (async).
    # store.add_documents() doesn't exist — raises AttributeError every call.
    # This tool runs inside a running async event loop (LangGraph agent node),
    # so asyncio.run() would also raise RuntimeError("event loop already running").
    # Solution: use the raw synchronous qdrant-client upsert directly.
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from langchain_ollama import OllamaEmbeddings
        from qdrant_client import models as qdrant_models
        import hashlib

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents([doc])

        # Embed synchronously via Ollama
        embeddings_model = OllamaEmbeddings(model=settings.embedding_model)
        texts = [c.page_content for c in chunks]
        vectors = embeddings_model.embed_documents(texts)

        # Upsert directly via raw qdrant client (synchronous)
        points = []
        for chunk, vector in zip(chunks, vectors):
            point_id = hashlib.md5(
                (chunk.metadata.get("source", "") + chunk.page_content).encode("utf-8")
            ).hexdigest()
            points.append(qdrant_models.PointStruct(
                id=point_id,
                vector=vector,
                payload={"page_content": chunk.page_content, "metadata": chunk.metadata}
            ))
        vector_store.client.upsert(collection_name="documents", points=points)
        return f"✅ Remembered ({len(chunks)} chunk(s)): {content[:120]}..."
    except Exception as e:
        return f"Failed to save memory: {e}"


# ─── System Prompt ────────────────────────────────────────────────

JARVIS_SYSTEM_PROMPT = """You are JARVIS — Manvith's personal AI assistant, built with complete knowledge of his work, projects, research, and thinking patterns.

You have access to:
- ALL his chat histories (Cursor, ChatGPT, Gemini, Claude)
- ALL his code projects and files
- His M.Tech thesis work (QuadFusion-AD, WaveDroughtNet)
- His research at Data Conquest (medical AI, clinical datasets)
- His job applications and documents

Your personality:
- Confident, direct, and technically sharp — like a senior engineer who knows his codebase
- Reference past work when relevant ("In your WaveDroughtNet project, you used...")
- Proactively surface patterns and connections across his work
- Concise by default; expand only when depth is needed

Decision flow for every message:
1. ALWAYS call search_my_knowledge first to get relevant context
2. If context is thin, search additional sources or use web search
3. Connect dots across different sources and time periods
4. Be specific — mention exact files, projects, dates when known
5. If genuinely unknown, say so and offer to search the web

You are Manvith's second brain. Act like it."""


# ─── LLM Factory ────────────────────────────────────────────────

def get_llm():
    if settings.llm_provider == "claude":
        return ChatAnthropic(
            model=settings.llm_model,
            api_key=settings.anthropic_api_key,
            max_tokens=4096
        )
    return ChatOpenAI(
        model=settings.llm_model,
        api_key=settings.openai_api_key,
        max_tokens=4096
    )


# ─── LangGraph Agent ────────────────────────────────────────────

TOOLS = [
    search_my_knowledge,
    read_file,
    list_directory,
    run_terminal_command,
    search_web,
    remember_this,
]


def build_jarvis_agent():
    llm = get_llm()
    llm_with_tools = llm.bind_tools(TOOLS)
    tool_map = {t.name: t for t in TOOLS}
    memory = MemorySaver()

    def agent_node(state: JarvisState) -> JarvisState:
        messages = [SystemMessage(content=JARVIS_SYSTEM_PROMPT)] + state["messages"]
        response = llm_with_tools.invoke(messages)
        return {"messages": [response], "retrieved_context": "", "tool_results": []}

    def tool_node(state: JarvisState) -> JarvisState:
        last_msg = state["messages"][-1]
        tool_messages = []

        for tool_call in last_msg.tool_calls:
            tool_fn = tool_map.get(tool_call["name"])
            if tool_fn:
                try:
                    result = tool_fn.invoke(tool_call["args"])
                except Exception as e:
                    result = f"Tool error ({tool_call['name']}): {e}"
            else:
                result = f"Unknown tool: {tool_call['name']}"

            tool_messages.append(
                ToolMessage(
                    content=str(result),
                    tool_call_id=tool_call["id"],
                    name=tool_call["name"]
                )
            )

        return {"messages": tool_messages, "retrieved_context": "", "tool_results": []}

    def should_continue(state: JarvisState) -> str:
        last = state["messages"][-1]
        if hasattr(last, "tool_calls") and last.tool_calls:
            return "tools"
        return "end"

    graph = StateGraph(JarvisState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)
    graph.set_entry_point("agent")
    graph.add_conditional_edges(
        "agent", should_continue, {"tools": "tools", "end": END}
    )
    graph.add_edge("tools", "agent")

    return graph.compile(checkpointer=memory)


# Singleton — built once at import time
jarvis_agent = build_jarvis_agent()


# ─── Public API ─────────────────────────────────────────────────

async def chat(message: str, session_id: str = "default") -> str:
    """Single-turn non-streaming chat."""
    config = {"configurable": {"thread_id": session_id}}
    result = await jarvis_agent.ainvoke(
        {
            "messages": [HumanMessage(content=message)],
            "retrieved_context": "",
            "tool_results": []
        },
        config=config
    )
    return result["messages"][-1].content


async def stream_chat(message: str, session_id: str = "default"):
    """Streaming chat — async generator that yields text chunks."""
    config = {"configurable": {"thread_id": session_id}}
    async for event in jarvis_agent.astream_events(
        {
            "messages": [HumanMessage(content=message)],
            "retrieved_context": "",
            "tool_results": []
        },
        config=config,
        version="v2"
    ):
        if event["event"] == "on_chat_model_stream":
            chunk = event["data"]["chunk"]
            if hasattr(chunk, "content") and chunk.content:
                yield chunk.content
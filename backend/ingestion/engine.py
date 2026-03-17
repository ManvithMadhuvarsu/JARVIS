"""
Ingestion Engine — parses and embeds all your data sources.

Supported:
  - Cursor chat history   (~/.cursor/logs/ or chat export)
  - ChatGPT export        (conversations.json)
  - Gemini export         (Google Takeout)
  - Claude export         (claude_conversations.json)
  - Local files           (.py, .ts, .md, .pdf, .txt, .docx, .json)
  - Git repos             (commit history + code)
"""
import json
import asyncio
import logging
from pathlib import Path
from datetime import datetime
from typing import AsyncGenerator

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader, TextLoader, UnstructuredMarkdownLoader
)

from config import settings, COLLECTION_ROUTING
from memory.vector_store import vector_store

logger = logging.getLogger(__name__)

# ─── Chunkers ─────────────────────────────────────────────────────
CODE_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=800, chunk_overlap=100,
    separators=["\nclass ", "\ndef ", "\n\n", "\n", " "]
)
TEXT_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=512, chunk_overlap=64,
    separators=["\n\n", "\n", ".", " "]
)
CHAT_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=1024, chunk_overlap=128,
    separators=["\n\n", "\n", ". "]
)


# ──────────────────────────────────────────────────────────────────
# CHAT HISTORY PARSERS
# ──────────────────────────────────────────────────────────────────

def parse_chatgpt_export(export_path: str) -> list[Document]:
    """Parse ChatGPT's conversations.json export."""
    docs = []
    data = json.loads(Path(export_path).read_text())

    for conv in data:
        title = conv.get("title", "Untitled")
        create_time = conv.get("create_time", 0)
        date = datetime.fromtimestamp(create_time).isoformat() if create_time else ""

        messages = []
        for node in conv.get("mapping", {}).values():
            msg = node.get("message")
            if not msg:
                continue
            role = msg.get("author", {}).get("role", "")
            parts = msg.get("content", {}).get("parts", [])
            text = " ".join(str(p) for p in parts if isinstance(p, str))
            if text.strip() and role in ("user", "assistant"):
                messages.append(f"[{role.upper()}]: {text.strip()}")

        if messages:
            full_text = f"# ChatGPT: {title}\n\n" + "\n\n".join(messages)
            chunks = CHAT_SPLITTER.create_documents(
                [full_text],
                metadatas=[{
                    "source": export_path,
                    "source_type": "chatgpt",
                    "title": title,
                    "date": date,
                    "platform": "chatgpt"
                }]
            )
            docs.extend(chunks)

    logger.info(f"Parsed {len(data)} ChatGPT conversations → {len(docs)} chunks")
    return docs


def parse_cursor_export(cursor_dir: str) -> list[Document]:
    """Parse Cursor AI chat history from workspace storage."""
    docs = []
    cursor_path = Path(cursor_dir)

    # Cursor stores chats in SQLite or JSON depending on version
    # Try JSON files first
    json_files = list(cursor_path.rglob("*.json"))

    for jf in json_files:
        try:
            data = json.loads(jf.read_text())
            # Cursor format: list of {role, content} or nested
            if isinstance(data, list):
                messages = []
                for item in data:
                    if isinstance(item, dict):
                        role = item.get("role", item.get("type", ""))
                        content = item.get("content", item.get("text", ""))
                        if content and role:
                            messages.append(f"[{role.upper()}]: {content}")

                if messages:
                    full_text = "\n\n".join(messages)
                    chunks = CHAT_SPLITTER.create_documents(
                        [full_text],
                        metadatas=[{
                            "source": str(jf),
                            "source_type": "cursor",
                            "date": datetime.fromtimestamp(jf.stat().st_mtime).isoformat(),
                            "platform": "cursor"
                        }]
                    )
                    docs.extend(chunks)
        except Exception as e:
            logger.debug(f"Skipping {jf}: {e}")

    logger.info(f"Parsed Cursor history → {len(docs)} chunks")
    return docs


def parse_claude_export(export_path: str) -> list[Document]:
    """Parse Claude.ai conversation export (JSON)."""
    docs = []
    data = json.loads(Path(export_path).read_text())

    conversations = data if isinstance(data, list) else [data]

    for conv in conversations:
        title = conv.get("name", conv.get("title", "Untitled"))
        messages = []

        for msg in conv.get("chat_messages", conv.get("messages", [])):
            role = msg.get("sender", msg.get("role", ""))
            # Content can be string or list of content blocks
            content = msg.get("text", msg.get("content", ""))
            if isinstance(content, list):
                content = " ".join(
                    c.get("text", "") for c in content
                    if isinstance(c, dict) and c.get("type") == "text"
                )
            if content and role:
                messages.append(f"[{role.upper()}]: {str(content)[:2000]}")

        if messages:
            full_text = f"# Claude: {title}\n\n" + "\n\n".join(messages)
            chunks = CHAT_SPLITTER.create_documents(
                [full_text],
                metadatas=[{
                    "source": export_path,
                    "source_type": "claude",
                    "title": title,
                    "platform": "claude"
                }]
            )
            docs.extend(chunks)

    logger.info(f"Parsed Claude export → {len(docs)} chunks")
    return docs


def parse_gemini_export(export_dir: str) -> list[Document]:
    """Parse Google Takeout Gemini AI data."""
    docs = []
    gemini_path = Path(export_dir)

    for json_file in gemini_path.rglob("*.json"):
        try:
            data = json.loads(json_file.read_text())
            if "conversations" in data:
                for conv in data["conversations"]:
                    messages = []
                    for turn in conv.get("turns", []):
                        role = turn.get("role", "")
                        text = turn.get("parts", [{}])[0].get("text", "")
                        if text:
                            messages.append(f"[{role.upper()}]: {text}")

                    if messages:
                        chunks = CHAT_SPLITTER.create_documents(
                            ["\n\n".join(messages)],
                            metadatas=[{
                                "source": str(json_file),
                                "source_type": "gemini",
                                "platform": "gemini"
                            }]
                        )
                        docs.extend(chunks)
        except Exception as e:
            logger.debug(f"Skipping {json_file}: {e}")

    logger.info(f"Parsed Gemini export → {len(docs)} chunks")
    return docs


# ──────────────────────────────────────────────────────────────────
# FILE INGESTION
# ──────────────────────────────────────────────────────────────────

EXCLUDE = {
    "node_modules", "__pycache__", ".git", "dist", "build",
    ".next", "venv", ".venv", "env", ".env", "coverage"
}

TEXT_EXTENSIONS = {
    ".py", ".js", ".ts", ".tsx", ".jsx", ".go", ".rs", ".java",
    ".cpp", ".c", ".h", ".cs", ".rb", ".php", ".swift", ".kt",
    ".md", ".txt", ".yaml", ".yml", ".toml", ".json", ".env",
    ".sh", ".bash", ".zsh", ".fish", ".html", ".css", ".scss",
    ".sql", ".graphql", ".proto", ".conf", ".ini", ".cfg"
}

MAX_BYTES = settings.max_file_size_mb * 1024 * 1024


def should_ingest(path: Path) -> bool:
    """Check if file should be ingested."""
    if any(ex in path.parts for ex in EXCLUDE):
        return False
    if path.stat().st_size > MAX_BYTES:
        return False
    return path.suffix.lower() in TEXT_EXTENSIONS or path.suffix.lower() == ".pdf"


async def ingest_file(file_path: str) -> int:
    """Ingest a single file. Returns number of chunks added."""
    path = Path(file_path)
    if not path.exists() or not path.is_file():
        return 0

    suffix = path.suffix.lower()
    collection = COLLECTION_ROUTING.get(suffix, "documents")

    try:
        if suffix == ".pdf":
            loader = PyPDFLoader(str(path))
            raw_docs = loader.load()
        elif suffix == ".md":
            loader = UnstructuredMarkdownLoader(str(path))
            raw_docs = loader.load()
        else:
            loader = TextLoader(str(path), autodetect_encoding=True)
            raw_docs = loader.load()

        # Choose splitter
        splitter = CODE_SPLITTER if collection == "code_files" else TEXT_SPLITTER

        for doc in raw_docs:
            doc.metadata.update({
                "source": str(path),
                "source_type": "local_file",
                "file_type": suffix,
                "project": _detect_project(path),
                "date": datetime.fromtimestamp(path.stat().st_mtime).isoformat(),
            })

        chunks = splitter.split_documents(raw_docs)
        count = await vector_store.add_documents(collection, chunks)
        return count

    except Exception as e:
        logger.error(f"Failed to ingest {path}: {e}")
        return 0


async def ingest_directory(dir_path: str, progress_callback=None) -> dict:
    """Walk a directory and ingest all eligible files."""
    root = Path(dir_path)
    stats = {"files": 0, "chunks": 0, "errors": 0}

    eligible = [f for f in root.rglob("*") if f.is_file() and should_ingest(f)]
    total = len(eligible)

    for i, file_path in enumerate(eligible):
        try:
            chunks = await ingest_file(str(file_path))
            stats["chunks"] += chunks
            stats["files"] += 1
        except Exception:
            stats["errors"] += 1

        if progress_callback:
            await progress_callback(i + 1, total, str(file_path))

    return stats


def _detect_project(path: Path) -> str:
    """Walk up to find project root (git repo or package.json)."""
    for parent in path.parents:
        if (parent / ".git").exists():
            return parent.name
        if (parent / "package.json").exists():
            return parent.name
        if (parent / "pyproject.toml").exists():
            return parent.name
    return "unknown"


# ──────────────────────────────────────────────────────────────────
# MASTER INGEST FUNCTION
# ──────────────────────────────────────────────────────────────────

async def ingest_all_chat_exports(exports_config: dict) -> dict:
    """
    Ingest all chat exports.

    exports_config = {
        "chatgpt": "/path/to/conversations.json",
        "cursor":  "/path/to/.cursor/logs",
        "claude":  "/path/to/claude_export.json",
        "gemini":  "/path/to/takeout/Gemini",
    }
    """
    stats = {}

    if path := exports_config.get("chatgpt"):
        docs = parse_chatgpt_export(path)
        count = await vector_store.add_documents("chat_history", docs)
        stats["chatgpt"] = count

    if path := exports_config.get("cursor"):
        docs = parse_cursor_export(path)
        count = await vector_store.add_documents("chat_history", docs)
        stats["cursor"] = count

    if path := exports_config.get("claude"):
        docs = parse_claude_export(path)
        count = await vector_store.add_documents("chat_history", docs)
        stats["claude"] = count

    if path := exports_config.get("gemini"):
        docs = parse_gemini_export(path)
        count = await vector_store.add_documents("chat_history", docs)
        stats["gemini"] = count

    return stats
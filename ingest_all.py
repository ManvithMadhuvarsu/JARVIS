#!/usr/bin/env python3
"""
JARVIS Full Ingestion Script
=============================
Ingests ALL personal data into a local Qdrant vector store using Ollama embeddings.
No Docker needed. No paid API keys needed for ingestion.

Sources:
  1. Claude conversation exports
  2. Projects directory
  3. Resume PDF
"""

import json
import os
import sys
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional
import uuid
import hashlib

import ollama
from qdrant_client import QdrantClient, models

# ─── Configuration ────────────────────────────────────────────────────
# FIX #4: All paths now use env vars with cross-platform defaults.
# Override by setting env vars before running: QDRANT_PATH, CLAUDE_DATA_DIR, etc.
import os as _os
from pathlib import Path as _Path
_root = _Path(__file__).resolve().parent  # project root

QDRANT_PATH     = _os.getenv("QDRANT_PATH",     str(_root / "qdrant_data"))
EMBEDDING_MODEL = _os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
EMBEDDING_DIM   = 768  # nomic-embed-text dimension

CLAUDE_DATA_DIR = _os.getenv("CLAUDE_DATA_DIR", str(_root / "data" / "claude"))
PROJECTS_DIR    = _os.getenv("PROJECTS_DIR",    str(_Path.home() / "Projects"))
RESUME_PATH     = _os.getenv("RESUME_PATH",     str(_Path.home() / "Documents" / "Resume.pdf"))

# Collections
COLLECTIONS = {
    "chat_history":  "All LLM chat conversations (Claude exports)",
    "code_files":    "Source code from projects",
    "documents":     "Markdown, PDF, text documents, resume",
}

# File routing
CODE_EXTENSIONS = {
    ".py", ".js", ".ts", ".tsx", ".jsx", ".go", ".rs", ".java",
    ".cpp", ".c", ".h", ".cs", ".rb", ".php", ".swift", ".kt",
}
DOC_EXTENSIONS = {
    ".md", ".txt", ".pdf", ".docx", ".json", ".yaml", ".yml",
    ".toml", ".ini", ".cfg", ".env", ".sh", ".bash", ".html",
    ".css", ".scss", ".sql", ".graphql",
}
EXCLUDE_DIRS = {
    "node_modules", "__pycache__", ".git", "dist", "build",
    ".next", "venv", ".venv", "env", ".env", "coverage",
    "qdrant_data", ".ollama", "ollama_models", ".cache",
    "__MACOSX", ".idea", ".vscode", "data",
}

MAX_FILE_BYTES = 10 * 1024 * 1024  # 10 MB
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("jarvis-ingest")


# ─── Embedding ────────────────────────────────────────────────────────

def get_embedding(text: str) -> list[float]:
    """Get embedding vector from Ollama."""
    response = ollama.embed(model=EMBEDDING_MODEL, input=text)
    return response["embeddings"][0]


def get_embeddings_batch(texts: list[str]) -> list[list[float]]:
    """Get embeddings for a batch of texts with retry."""
    try:
        response = ollama.embed(model=EMBEDDING_MODEL, input=texts)
        return response["embeddings"]
    except Exception:
        # If batch fails, fall back to one-by-one
        results = []
        for t in texts:
            try:
                r = ollama.embed(model=EMBEDDING_MODEL, input=t)
                results.append(r["embeddings"][0])
            except Exception:
                # Return zero vector as fallback
                results.append([0.0] * EMBEDDING_DIM)
        return results


# ─── Text Chunking ────────────────────────────────────────────────────

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks."""
    if len(text) <= chunk_size:
        return [text] if text.strip() else []

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size

        # Try to break at a natural boundary
        if end < len(text):
            for sep in ["\n\n", "\n", ". ", " "]:
                last_sep = text.rfind(sep, start + chunk_size // 2, end)
                if last_sep != -1:
                    end = last_sep + len(sep)
                    break

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        start = end - overlap
        if start >= len(text):
            break

    return chunks


# ─── Qdrant Client ───────────────────────────────────────────────────

class LocalVectorStore:
    """Manages local Qdrant storage with Ollama embeddings."""

    def __init__(self, path: str):
        os.makedirs(path, exist_ok=True)
        self.client = QdrantClient(path=path)
        self.stats = {name: 0 for name in COLLECTIONS}
        self._ensure_collections()

    def _ensure_collections(self):
        existing = {c.name for c in self.client.get_collections().collections}
        for name in COLLECTIONS:
            if name not in existing:
                self.client.create_collection(
                    collection_name=name,
                    vectors_config=models.VectorParams(
                        size=EMBEDDING_DIM,
                        distance=models.Distance.COSINE
                    ),
                )
                logger.info(f"Created collection: {name}")
            else:
                info = self.client.get_collection(name)
                count = info.points_count or 0
                self.stats[name] = count
                logger.info(f"Collection '{name}' exists with {count:,} points")

    def upsert_chunks(self, collection: str, chunks: list[dict]) -> int:
        """Upsert chunks with embeddings into Qdrant. Returns count."""
        if not chunks:
            return 0

        # Batch embed all chunk texts (small batches for memory)
        texts = [c["text"] for c in chunks]
        BATCH_SIZE = 4
        all_embeddings = []

        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i:i + BATCH_SIZE]
            embeddings = get_embeddings_batch(batch)
            all_embeddings.extend(embeddings)

        points = []
        for i, (chunk, embedding) in enumerate(zip(chunks, all_embeddings)):
            # Generate stable ID for idempotency (hash of source + text)
            content_id = f"{chunk.get('source', '')}_{chunk['text']}"
            point_id = hashlib.md5(content_id.encode("utf-8")).hexdigest()
            
            points.append(models.PointStruct(
                id=point_id,
                vector=embedding,
                payload={
                    "text": chunk["text"],
                    "source": chunk.get("source", ""),
                    "source_index": i,  # To help order if needed
                    "source_type": chunk.get("source_type", ""),
                    "title": chunk.get("title", ""),
                    "date": chunk.get("date", ""),
                    "project": chunk.get("project", ""),
                    "platform": chunk.get("platform", ""),
                    "file_type": chunk.get("file_type", ""),
                }
            ))

        # Upsert in batches
        for i in range(0, len(points), 100):
            batch = points[i:i + 100]
            self.client.upsert(collection_name=collection, points=batch)

        self.stats[collection] = self.stats.get(collection, 0) + len(points)
        return len(points)

    def search(self, query: str, collection: str = None, top_k: int = 5) -> list[dict]:
        """Search across collections."""
        query_embedding = get_embedding(query)
        collections = [collection] if collection else list(COLLECTIONS.keys())

        results = []
        for col in collections:
            try:
                hits = self.client.query_points(
                    collection_name=col,
                    query=query_embedding,
                    limit=top_k,
                )
                for hit in hits.points:
                    results.append({
                        "score": hit.score,
                        "text": hit.payload.get("text", ""),
                        "source": hit.payload.get("source", ""),
                        "source_type": hit.payload.get("source_type", ""),
                        "title": hit.payload.get("title", ""),
                        "collection": col,
                    })
            except Exception as e:
                logger.warning(f"Search failed on '{col}': {e}")

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def get_stats(self) -> dict:
        stats = {}
        for name in COLLECTIONS:
            try:
                info = self.client.get_collection(name)
                stats[name] = info.points_count or 0
            except:
                stats[name] = 0
        return stats


# ─── CLAUDE PARSER ───────────────────────────────────────────────────

def parse_claude_conversations(export_path: str) -> list[dict]:
    """Parse a Claude conversations.json export into chunks."""
    chunks = []
    try:
        data = json.loads(Path(export_path).read_text(encoding="utf-8"))
    except Exception as e:
        logger.error(f"Failed to read {export_path}: {e}")
        return []

    conversations = data if isinstance(data, list) else [data]

    for conv in conversations:
        title = conv.get("name", conv.get("title", "Untitled"))
        created = conv.get("created_at", conv.get("create_time", ""))
        date = ""
        if created:
            try:
                if isinstance(created, (int, float)):
                    date = datetime.fromtimestamp(created).isoformat()
                else:
                    date = str(created)[:19]
            except:
                date = str(created)[:19]

        messages = []
        for msg in conv.get("chat_messages", conv.get("messages", [])):
            role = msg.get("sender", msg.get("role", ""))
            content = msg.get("text", msg.get("content", ""))
            if isinstance(content, list):
                content = " ".join(
                    c.get("text", "") for c in content
                    if isinstance(c, dict) and c.get("type") == "text"
                )
            if content and role:
                messages.append(f"[{role.upper()}]: {str(content)[:3000]}")

        if messages:
            full_text = f"# Claude: {title}\n\n" + "\n\n".join(messages)
            text_chunks = chunk_text(full_text, chunk_size=1024, overlap=128)
            for tc in text_chunks:
                chunks.append({
                    "text": tc,
                    "source": export_path,
                    "source_type": "claude",
                    "title": title,
                    "date": date,
                    "platform": "claude",
                })

    return chunks


# ─── FILE INGESTION ──────────────────────────────────────────────────

def should_ingest(path: Path) -> bool:
    """Check if a file should be ingested."""
    if any(ex in path.parts for ex in EXCLUDE_DIRS):
        return False
    try:
        if path.stat().st_size > MAX_FILE_BYTES:
            return False
        if path.stat().st_size == 0:
            return False
    except:
        return False
    suffix = path.suffix.lower()
    return suffix in CODE_EXTENSIONS or suffix in DOC_EXTENSIONS or suffix == ".pdf"


def read_file_text(path: Path) -> str:
    """Read file content as text."""
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        try:
            from pypdf import PdfReader
            reader = PdfReader(str(path))
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""
                text += "\n\n"
            return text
        except Exception as e:
            logger.warning(f"Failed to read PDF {path}: {e}")
            return ""

    # Text files
    for encoding in ["utf-8", "utf-8-sig", "latin-1", "cp1252"]:
        try:
            return path.read_text(encoding=encoding)
        except (UnicodeDecodeError, ValueError):
            continue

    return ""


def detect_project(path: Path) -> str:
    """Walk up to find project root."""
    for parent in path.parents:
        if (parent / ".git").exists():
            return parent.name
        if (parent / "package.json").exists():
            return parent.name
        if (parent / "pyproject.toml").exists():
            return parent.name
    return path.parts[0] if path.parts else "unknown"


def ingest_file(store: LocalVectorStore, file_path: Path) -> int:
    """Ingest a single file. Returns chunk count."""
    text = read_file_text(file_path)
    if not text.strip():
        return 0

    suffix = file_path.suffix.lower()
    if suffix in CODE_EXTENSIONS:
        collection = "code_files"
        text_chunks = chunk_text(text, chunk_size=800, overlap=100)
    else:
        collection = "documents"
        text_chunks = chunk_text(text, chunk_size=512, overlap=64)

    chunks = []
    for tc in text_chunks:
        chunks.append({
            "text": tc,
            "source": str(file_path),
            "source_type": "local_file",
            "title": file_path.name,
            "date": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat(),
            "project": detect_project(file_path),
            "file_type": suffix,
        })

    return store.upsert_chunks(collection, chunks)


# ─── MAIN INGESTION ─────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("JARVIS - Full Knowledge Base Ingestion")
    print("=" * 60)
    print()

    store = LocalVectorStore(QDRANT_PATH)

    # Show current stats
    stats = store.get_stats()
    print("Current knowledge base:")
    for col, count in stats.items():
        print(f"   {col}: {count:,} chunks")
    print()

    total_chunks = 0
    start_time = time.time()

    # -- 1. CLAUDE CONVERSATIONS -------------------------------------
    print("=" * 60)
    print("Phase 1: Ingesting Claude Conversation Exports")
    print("=" * 60)

    claude_dir = Path(CLAUDE_DATA_DIR)
    if claude_dir.exists():
        for folder in sorted(claude_dir.iterdir()):
            conv_file = folder / "conversations.json"
            if conv_file.exists():
                file_size_mb = conv_file.stat().st_size / (1024 * 1024)
                print(f"\n   Folder {folder.name}/conversations.json ({file_size_mb:.1f} MB)")

                chunks = parse_claude_conversations(str(conv_file))
                print(f"      Parsed {len(chunks)} chunks")

                if chunks:
                    count = store.upsert_chunks("chat_history", chunks)
                    total_chunks += count
                    print(f"      Indexed {count} chunks")

            # Also ingest memories.json and projects.json if present
            for extra_file in ["memories.json", "projects.json"]:
                extra_path = folder / extra_file
                if extra_path.exists() and extra_path.stat().st_size > 10:
                    try:
                        data = json.loads(extra_path.read_text(encoding="utf-8"))
                        if data:
                            text = json.dumps(data, indent=2)
                            text_chunks = chunk_text(text, chunk_size=512, overlap=64)
                            doc_chunks = [{
                                "text": tc,
                                "source": str(extra_path),
                                "source_type": "claude",
                                "title": extra_file,
                                "platform": "claude",
                            } for tc in text_chunks]
                            count = store.upsert_chunks("documents", doc_chunks)
                            total_chunks += count
                            print(f"      {extra_file}: {count} chunks")
                    except Exception as e:
                        logger.debug(f"Skipped {extra_path}: {e}")

    print(f"\n   Claude total: {total_chunks:,} chunks")

    # -- 2. RESUME ---------------------------------------------------
    print()
    print("=" * 60)
    print("Phase 2: Ingesting Resume")
    print("=" * 60)

    resume = Path(RESUME_PATH)
    if resume.exists():
        count = ingest_file(store, resume)
        total_chunks += count
        print(f"   Resume: {count} chunks indexed")
    else:
        print(f"   Resume not found at {RESUME_PATH}")

    # -- 3. PROJECT FILES --------------------------------------------
    print()
    print("=" * 60)
    print("Phase 3: Ingesting Project Files from D:\\Projects")
    print("=" * 60)

    projects_root = Path(PROJECTS_DIR)
    if projects_root.exists():
        # Get all eligible files
        eligible_files = []
        for f in projects_root.rglob("*"):
            if f.is_file() and should_ingest(f):
                eligible_files.append(f)

        print(f"\n   Found {len(eligible_files):,} eligible files")
        
        project_stats = {}
        processed = 0
        errors = 0

        for i, file_path in enumerate(eligible_files):
            try:
                count = ingest_file(store, file_path)
                if count > 0:
                    project = detect_project(file_path)
                    project_stats[project] = project_stats.get(project, 0) + count
                    total_chunks += count
                    processed += 1
            except Exception as e:
                errors += 1
                logger.debug(f"Error ingesting {file_path}: {e}")

            # Progress update every 50 files
            if (i + 1) % 50 == 0 or i + 1 == len(eligible_files):
                elapsed = time.time() - start_time
                print(f"   [{i+1}/{len(eligible_files)}] "
                      f"Processed: {processed}, Errors: {errors}, "
                      f"Time: {elapsed:.0f}s")

        print(f"\n   Project stats:")
        for project, count in sorted(project_stats.items(), key=lambda x: -x[1]):
            print(f"      {project}: {count:,} chunks")

    # -- FINAL STATS -------------------------------------------------
    print()
    print("=" * 60)
    print("INGESTION COMPLETE!")
    print("=" * 60)
    elapsed = time.time() - start_time
    final_stats = store.get_stats()
    print(f"\n   Total time: {elapsed:.1f}s")
    print(f"   Knowledge base:")
    for col, count in final_stats.items():
        print(f"      {col}: {count:,} chunks")
    print(f"   Total: {sum(final_stats.values()):,} chunks")

    # -- Quick test search -------------------------------------------
    print()
    print("=" * 60)
    print("Quick Test Search: 'what projects has Manvith worked on?'")
    print("=" * 60)
    results = store.search("what projects has Manvith worked on?", top_k=5)
    for i, r in enumerate(results, 1):
        print(f"\n   [{i}] Score: {r['score']:.3f} | {r['source_type']} | {r['collection']}")
        print(f"       {r['text'][:150]}...")

    print("\nJARVIS knowledge base is ready!")
    print(f"   Stored at: {QDRANT_PATH}")


if __name__ == "__main__":
    main()

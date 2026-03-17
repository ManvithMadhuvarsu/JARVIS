# JARVIS — Personal RAG AI Assistant

> Your personal AI that knows everything you've ever built, written, and thought.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     JARVIS System Overview                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│   ┌──────────────┐    ┌────────────────────────────────────┐    │
│   │   FRONTEND   │    │         INGESTION PIPELINE         │    │
│   │  (React UI)  │    │                                    │    │
│   │  - Chat      │    │  ┌──────────┐  ┌────────────────┐  │    │
│   │  - Memory    │    │  │ Watchers │  │  Data Sources  │  │    │
│   │  - File Mgmt │    │  │ (inotify)│  │  - Cursor Chat │  │    │
│   └──────┬───────┘    │  │ (cron)   │  │  - GPT History │  │    │
│          │            │  └────┬─────┘  │  - Gemini      │  │    │
│   ┌──────▼───────┐    │       │        │  - Local Files  │  │    │
│   │  FastAPI     │    │       ▼        │  - Code Repos  │  │    │
│   │  Backend     │    │  ┌─────────────▼──────────────┐  │    │
│   │              │    │  │   Ingestion Engine          │  │    │
│   │  /chat       │    │  │   - Chunker (semantic)      │  │    │
│   │  /search     │    │  │   - Embedder (OpenAI/local) │  │    │
│   │  /memory     │    │  │   - Deduplicator            │  │    │
│   │  /ingest     │    │  └────────────┬───────────────┘  │    │
│   │  /files      │    └───────────────┼────────────────────┘    │
│   └──────┬───────┘                    │                          │
│          │                            ▼                          │
│   ┌──────▼────────────────────────────────────┐                 │
│   │              VECTOR STORE (Qdrant)         │                 │
│   │  Collections: chat_history | code | files  │                 │
│   │  Metadata: source | date | project | type  │                 │
│   └──────┬─────────────────────────────────────┘                │
│          │                                                        │
│   ┌──────▼──────────────────────────────────┐                   │
│   │           LANGGRAPH AGENT                │                   │
│   │                                          │                   │
│   │  Query → Rewrite → Hybrid Retrieve       │                   │
│   │        → Rerank → Fuse Context           │                   │
│   │        → Reason → Stream Response        │                   │
│   │                                          │                   │
│   │  Tools: file_read | system_exec |        │                   │
│   │         web_search | code_run |          │                   │
│   │         memory_store | calendar          │                   │
│   └──────────────────────────────────────────┘                   │
└─────────────────────────────────────────────────────────────────┘
```

## Quick Start

```bash
# 1. Clone and setup
cp .env.example .env
# Fill in your API keys

# 2. Start all services
docker-compose up -d

# 3. Run initial ingestion
python scripts/ingest_all.py --path ~/

# 4. Open UI
open http://localhost:3000
```

## Data Sources Supported

| Source | What's Ingested |
|--------|-----------------|
| Cursor | `.cursor/chat_history/` JSON conversations |
| ChatGPT | Exported `conversations.json` |
| Gemini | Google Takeout AI data |
| Claude | Exported conversation JSON |
| Local Files | Code, docs, markdown, PDFs |
| Git Repos | Commit history, README, code |
| Browser | Bookmarks, history (optional) |

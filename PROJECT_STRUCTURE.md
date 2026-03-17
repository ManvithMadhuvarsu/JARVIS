# JARVIS — Project Structure

```
JARVIS/
│
├── .env.example                    ← Copy to .env and fill API keys
├── docker-compose.yml              ← Start everything: docker-compose up -d
├── README.md
│
├── backend/                        ← Python FastAPI + LangGraph
│   ├── Dockerfile                  ← Includes PYTHONPATH=/app fix
│   ├── requirements.txt            ← Deduplicated, pinned versions
│   ├── config.py                   ← Settings (reads from .env)
│   │
│   ├── api/
│   │   ├── __init__.py             ← Makes this a Python package
│   │   └── main.py                 ← FastAPI app: /chat /search /ingest /files
│   │
│   ├── agents/
│   │   ├── __init__.py
│   │   └── jarvis_agent.py         ← LangGraph agent + all tools (FIXED)
│   │
│   ├── memory/
│   │   ├── __init__.py
│   │   └── vector_store.py         ← Qdrant multi-collection manager (FIXED)
│   │
│   └── ingestion/
│       ├── __init__.py
│       ├── engine.py               ← ChatGPT/Cursor/Claude/Gemini parsers
│       └── watcher.py              ← Auto-ingest on file changes
│
├── frontend/                       ← React + Vite + Tailwind
│   ├── Dockerfile
│   ├── nginx.conf                  ← NEW: was missing, caused build failure
│   ├── index.html
│   ├── package.json
│   ├── vite.config.ts
│   ├── tailwind.config.js
│   ├── postcss.config.js           ← NEW: was missing, Tailwind won't work without
│   ├── tsconfig.json               ← NEW: was missing, TypeScript won't compile
│   ├── tsconfig.node.json          ← NEW: required by tsconfig.json
│   └── src/
│       ├── main.tsx
│       ├── index.css
│       └── App.tsx                 ← Full JARVIS UI (Chat / Search / Files / Memory)
│
└── scripts/
    ├── setup.py                    ← Interactive setup wizard
    └── ingest.py                   ← CLI for manual data ingestion
```

## Quick Start

```bash
# 1. Setup
cp .env.example .env
# Edit .env with your API keys

# 2. Start infrastructure
docker-compose up -d qdrant redis

# 3. Install backend deps (local dev)
cd backend && pip install -r requirements.txt

# 4. Ingest your data
cd .. && python scripts/ingest.py --all

# 5. Start everything
docker-compose up -d

# 6. Open
open http://localhost:3000
```

## Export Your Chat Data

| Platform | How to Export |
|---|---|
| ChatGPT | Settings → Data Controls → Export → `conversations.json` |
| Claude | Settings → Privacy → Export → `claude_conversations.json` |
| Gemini | takeout.google.com → Select Gemini AI |
| Cursor | `~/.cursor/logs/` (auto-detected) |

Then run:
```bash
python scripts/ingest.py \
  --chatgpt ~/Downloads/conversations.json \
  --claude ~/Downloads/claude_conversations.json \
  --cursor ~/.cursor/logs
```

#!/usr/bin/env python3
"""
JARVIS Setup Script — run this to configure and start everything.
"""
import os
import sys
import subprocess
import json
from pathlib import Path

HOME = Path.home()
ROOT = Path(__file__).parent.parent  # Project root (one level above scripts/)

def detect_chat_exports():
    """Auto-detect common chat export locations."""
    found = {}

    # Cursor locations (macOS, Linux, Windows)
    cursor_paths = [
        HOME / ".cursor" / "logs",
        HOME / "Library" / "Application Support" / "Cursor" / "User" / "workspaceStorage",
        HOME / ".config" / "Cursor" / "User" / "workspaceStorage",
    ]
    for p in cursor_paths:
        if p.exists():
            found["cursor"] = str(p)
            print(f"✅ Found Cursor history: {p}")
            break

    # ChatGPT export
    gpt_paths = [
        HOME / "Downloads" / "conversations.json",
        HOME / "Desktop" / "conversations.json",
    ]
    for p in gpt_paths:
        if p.exists():
            found["chatgpt"] = str(p)
            print(f"✅ Found ChatGPT export: {p}")
            break

    # Claude export
    claude_paths = [
        HOME / "Downloads" / "claude_conversations.json",
        HOME / "Desktop" / "claude_conversations.json",
    ]
    for p in claude_paths:
        if p.exists():
            found["claude"] = str(p)
            print(f"✅ Found Claude export: {p}")
            break

    return found


def create_env():
    """Interactive .env setup."""
    env_path = ROOT / ".env"  # Place .env at the project root, not scripts/
    if env_path.exists():
        print("⚠️  .env already exists. Skipping...")
        return

    print("\n📋 JARVIS Setup — API Keys\n")

    anthropic_key = input("Anthropic API Key (sk-ant-...): ").strip()
    openai_key = input("OpenAI API Key (sk-...): ").strip()

    llm_choice = input("Primary LLM [claude/openai] (default: claude): ").strip() or "claude"

    env_content = f"""ANTHROPIC_API_KEY={anthropic_key}
OPENAI_API_KEY={openai_key}
LLM_PROVIDER={llm_choice}
LLM_MODEL={"claude-sonnet-4-5" if llm_choice == "claude" else "gpt-4o"}
# FIX #10: was hardcoded to openai — system uses local Ollama embeddings
EMBEDDING_PROVIDER=local
EMBEDDING_MODEL=nomic-embed-text
QDRANT_URL=http://localhost:6333
REDIS_URL=redis://localhost:6379
USER_HOME={HOME}
ENABLE_SYSTEM_EXEC=true
ENABLE_FILE_WRITE=false
ENABLE_WEB_SEARCH=true
ENABLE_AUTO_INGEST=true
"""
    env_path.write_text(env_content)
    print(f"✅ Created .env at {env_path}")


def main():
    print("🚀 JARVIS Personal AI Assistant — Setup\n")
    print("=" * 50)

    # Step 1: Create .env
    create_env()

    # Step 2: Detect exports
    print("\n🔍 Scanning for chat exports...")
    exports = detect_chat_exports()

    if not exports:
        print("\n💡 No chat exports auto-detected.")
        print("   Export instructions:")
        print("   - ChatGPT: Settings → Data controls → Export data")
        print("   - Claude:  Settings → Privacy → Export data")
        print("   - Gemini:  takeout.google.com → Select Gemini")
        print("   Then place files in ~/Downloads/ and re-run this script.")

    # Step 3: Start Docker services
    print("\n🐳 Starting Docker services...")
    result = subprocess.run(
        ["docker-compose", "up", "-d", "qdrant", "redis"],
        cwd=ROOT,  # Must run from project root where docker-compose.yml lives
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print("❌ Docker failed. Make sure Docker is running.")
        print(result.stderr[:500])
        sys.exit(1)
    print("✅ Qdrant + Redis started")

    # Step 4: Install backend deps
    print("\n📦 Installing backend dependencies...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", "backend/requirements.txt", "-q"])
    print("✅ Backend dependencies installed")

    # Step 5: Run initial ingestion via subprocess (avoids exec() anti-pattern)
    if exports:
        print(f"\n📚 Running initial ingestion from {len(exports)} sources...")
        ingest_args = []
        if "chatgpt" in exports:
            ingest_args += ["--chatgpt", exports["chatgpt"]]
        if "cursor" in exports:
            ingest_args += ["--cursor", exports["cursor"]]
        if "claude" in exports:
            ingest_args += ["--claude", exports["claude"]]
        if "gemini" in exports:
            ingest_args += ["--gemini", exports["gemini"]]
        ingest_args += ["--all"]
        subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "ingest.py")] + ingest_args,
            cwd=ROOT
        )

    # Step 6: Launch everything
    print("\n🌐 Starting JARVIS...")
    print("\n" + "=" * 50)
    print("✅ JARVIS is ready!")
    print(f"   UI:      http://localhost:3000")
    print(f"   API:     http://localhost:8000")
    print(f"   Docs:    http://localhost:8000/docs")
    print("=" * 50)
    print("\nRun: docker-compose up -d")


if __name__ == "__main__":
    main()

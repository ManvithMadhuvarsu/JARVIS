#!/usr/bin/env python3
"""
JARVIS Ingest CLI — manually trigger ingestion of specific paths.

Usage:
    python scripts/ingest.py --chatgpt ~/Downloads/conversations.json
    python scripts/ingest.py --cursor ~/.cursor/logs
    python scripts/ingest.py --dir ~/Projects
    python scripts/ingest.py --all   # ingest everything auto-detected
"""
import asyncio
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from ingestion.engine import (
    ingest_directory, ingest_all_chat_exports,
    parse_chatgpt_export, parse_cursor_export,
    parse_claude_export, parse_gemini_export
)
from memory.vector_store import vector_store
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

console = Console()


async def main():
    parser = argparse.ArgumentParser(description="JARVIS Data Ingestion")
    parser.add_argument("--chatgpt", help="Path to ChatGPT conversations.json")
    parser.add_argument("--cursor", help="Path to Cursor chat directory")
    parser.add_argument("--claude", help="Path to Claude export JSON")
    parser.add_argument("--gemini", help="Path to Gemini Takeout directory")
    parser.add_argument("--dir", help="Directory to ingest (files + code)")
    parser.add_argument("--all", action="store_true", help="Auto-detect and ingest everything")
    args = parser.parse_args()

    console.print("[bold cyan]JARVIS Ingestion CLI[/bold cyan]\n")

    # Show current stats
    stats = vector_store.get_stats()
    console.print("[dim]Current knowledge base:[/dim]")
    for col, info in stats.items():
        console.print(f"  {col}: [yellow]{info['count']:,}[/yellow] chunks")
    console.print()

    exports = {}
    if args.chatgpt:
        exports["chatgpt"] = args.chatgpt
    if args.cursor:
        exports["cursor"] = args.cursor
    if args.claude:
        exports["claude"] = args.claude
    if args.gemini:
        exports["gemini"] = args.gemini

    if args.all:
        home = Path.home()
        cursor_paths = [
            home / ".cursor" / "logs",
            home / "Library" / "Application Support" / "Cursor" / "User" / "workspaceStorage",
        ]
        for p in cursor_paths:
            if p.exists():
                exports["cursor"] = str(p)
                break

        for name in ["conversations.json"]:
            for loc in [home / "Downloads" / name, home / "Desktop" / name]:
                if loc.exists():
                    exports["chatgpt"] = str(loc)

    if exports:
        console.print(f"[bold]Ingesting chat exports:[/bold] {list(exports.keys())}")
        with console.status("Processing chat histories..."):
            result = await ingest_all_chat_exports(exports)
        for source, count in result.items():
            console.print(f"  ✅ {source}: [green]{count}[/green] chunks indexed")

    if args.dir or args.all:
        dirs = [args.dir] if args.dir else []
        if args.all:
            home = Path.home()
            dirs.extend([
                str(home / d) for d in ["Projects", "Code", "repos", "Documents", "Desktop"]
                if (home / d).exists()
            ])

        for d in dirs:
            console.print(f"\n[bold]Indexing directory:[/bold] {d}")
            total_files = sum(1 for _ in Path(d).rglob("*") if _.is_file())
            processed = 0

            async def progress_cb(current, total, path):
                nonlocal processed
                processed = current
                if current % 50 == 0:
                    console.print(f"  [{current}/{total}] {Path(path).name}", highlight=False)

            result = await ingest_directory(d, progress_callback=progress_cb)
            console.print(f"  ✅ Files: [green]{result['files']}[/green], Chunks: [green]{result['chunks']}[/green]")

    # Final stats
    console.print("\n[bold cyan]Knowledge base updated:[/bold cyan]")
    stats = vector_store.get_stats()
    for col, info in stats.items():
        console.print(f"  {col}: [green]{info['count']:,}[/green] chunks")


if __name__ == "__main__":
    asyncio.run(main())

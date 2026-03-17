"""
File Watcher — monitors your filesystem and auto-ingests changed files.
Uses watchdog for cross-platform file system events.
"""
import asyncio
import logging
import time
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileModifiedEvent, FileCreatedEvent

from config import settings
from ingestion.engine import ingest_file, should_ingest

logger = logging.getLogger(__name__)

DEBOUNCE_SECONDS = 5  # Wait N seconds after last change before ingesting


class JarvisFileHandler(FileSystemEventHandler):
    """Handles file system events and queues files for ingestion."""

    def __init__(self):
        self.pending: dict[str, float] = {}  # path -> last_modified_time

    def on_modified(self, event):
        if not event.is_directory:
            self._queue(event.src_path)

    def on_created(self, event):
        if not event.is_directory:
            self._queue(event.src_path)

    def _queue(self, path: str):
        p = Path(path)
        if p.is_file() and should_ingest(p):
            self.pending[path] = time.time()
            logger.debug(f"Queued: {path}")

    async def process_pending(self):
        """Process files that haven't changed in DEBOUNCE_SECONDS."""
        now = time.time()
        to_process = [
            path for path, ts in list(self.pending.items())
            if now - ts >= DEBOUNCE_SECONDS
        ]
        for path in to_process:
            del self.pending[path]
            try:
                count = await ingest_file(path)
                if count:
                    logger.info(f"Auto-ingested: {path} ({count} chunks)")
            except Exception as e:
                logger.error(f"Failed to auto-ingest {path}: {e}")


async def run_watcher():
    """Start file system watcher on user's home directory."""
    if not settings.enable_auto_ingest:
        logger.info("Auto-ingest disabled. Watcher not started.")
        return

    handler = JarvisFileHandler()
    observer = Observer()

    watch_paths = [
        settings.user_home,
    ]

    for path in watch_paths:
        if Path(path).exists():
            observer.schedule(handler, path, recursive=True)
            logger.info(f"Watching: {path}")

    observer.start()
    logger.info("File watcher started.")

    try:
        while True:
            await asyncio.sleep(2)
            await handler.process_pending()
    except KeyboardInterrupt:
        observer.stop()
    observer.join()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    asyncio.run(run_watcher())
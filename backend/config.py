from pydantic_settings import BaseSettings
from pathlib import Path
import os


class Settings(BaseSettings):
    # LLM
    anthropic_api_key: str = ""
    openai_api_key: str = ""
    llm_provider: str = "claude"
    llm_model: str = "claude-sonnet-4-5"

    # Embeddings
    # FIX: defaults changed to "local"/"nomic-embed-text" to match .env and ingest_all.py
    # FIX: removed unused embedding_dim field — vector size derived dynamically below
    embedding_provider: str = "local"
    embedding_model: str = "nomic-embed-text"

    # Infrastructure
    # qdrant_url can be http URL OR local path — vector_store.py handles both
    qdrant_url: str = "http://localhost:6333"
    redis_url: str = "redis://localhost:6379"

    # System
    user_home: str = str(Path.home())
    exclude_patterns: str = "node_modules,__pycache__,.git,*.pyc"
    max_file_size_mb: int = 10

    # Feature flags
    enable_system_exec: bool = True
    enable_file_write: bool = False
    enable_web_search: bool = True
    enable_auto_ingest: bool = True

    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()


def get_vector_size() -> int:
    provider = os.getenv("EMBEDDING_PROVIDER", settings.embedding_provider).lower()
    return 768 if provider == "local" else 1536


VECTOR_SIZE = get_vector_size()

COLLECTIONS = {
    "chat_history": {"description": "All LLM chat conversations", "vector_size": VECTOR_SIZE},
    "code_files":   {"description": "Source code from projects",  "vector_size": VECTOR_SIZE},
    "documents":    {"description": "Markdown, PDF, text docs",   "vector_size": VECTOR_SIZE},
    "web_research": {"description": "Bookmarks, saved pages",     "vector_size": VECTOR_SIZE},
}

COLLECTION_ROUTING = {
    ".py": "code_files", ".js": "code_files", ".ts": "code_files",
    ".tsx": "code_files", ".jsx": "code_files", ".go": "code_files",
    ".rs": "code_files", ".java": "code_files", ".cpp": "code_files",
    ".c": "code_files", ".md": "documents", ".txt": "documents",
    ".pdf": "documents", ".docx": "documents", ".json": "documents",
}
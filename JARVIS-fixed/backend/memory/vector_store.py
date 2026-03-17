"""
Vector store manager — handles all Qdrant collections for JARVIS.
"""
from __future__ import annotations
from typing import Optional
from qdrant_client import QdrantClient, models
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document
from config import settings, COLLECTIONS
# Ollama embeddings will be imported lazily if needed
import logging

logger = logging.getLogger(__name__)


class VectorStoreManager:
    """Manages all Qdrant collections for JARVIS."""

    def __init__(self):
        # Handle local Qdrant path
        # Detect a local path only if it is NOT an HTTP(S) URL
        if ("/" in settings.qdrant_url or "\\" in settings.qdrant_url) and \
           not settings.qdrant_url.lower().startswith(("http://", "https://")):
            self.client = QdrantClient(path=settings.qdrant_url)
            logger.info(f"Using local Qdrant at {settings.qdrant_url}")
        else:
            self.client = QdrantClient(url=settings.qdrant_url, timeout=30)
            logger.info(f"Using remote Qdrant at {settings.qdrant_url}")

        if settings.embedding_provider.lower() == "local":
            try:
                from langchain_ollama import OllamaEmbeddings
            except Exception as e:
                raise ImportError("Ollama embeddings requested but langchain_ollama is not installed.") from e
            self.embeddings = OllamaEmbeddings(model=settings.embedding_model)
            logger.info(f"Using local Ollama embeddings ({settings.embedding_model})")
        else:
            self.embeddings = OpenAIEmbeddings(
                model=settings.embedding_model,
                api_key=settings.openai_api_key
            )
            logger.info(f"Using OpenAI embeddings ({settings.embedding_model})")
        
        self._stores: dict[str, QdrantVectorStore] = {}
        self._ensure_collections()

    def _ensure_collections(self):
        """Create collections if they don't exist."""
        try:
            existing = {c.name for c in self.client.get_collections().collections}
        except Exception as e:
            logger.error(f"Cannot connect to Qdrant at {settings.qdrant_url}: {e}")
            return

        for name, cfg in COLLECTIONS.items():
            if name not in existing:
                self.client.create_collection(
                    collection_name=name,
                    vectors_config=models.VectorParams(
                        size=cfg["vector_size"],
                        distance=models.Distance.COSINE
                    ),
                    hnsw_config=models.HnswConfigDiff(m=16, ef_construct=100),
                )
                # Payload indexes — must use metadata.<field> prefix to match stored docs
                for field in ["source", "source_type", "project", "platform", "date"]:
                    try:
                        self.client.create_payload_index(
                            collection_name=name,
                            field_name=f"metadata.{field}",
                            field_schema=models.PayloadSchemaType.KEYWORD
                        )
                    except Exception:
                        pass  # Index may already exist
                logger.info(f"Created collection: {name}")
            else:
                logger.debug(f"Collection already exists: {name}")

    def get_store(self, collection: str) -> QdrantVectorStore:
        if collection not in self._stores:
            self._stores[collection] = QdrantVectorStore(
                client=self.client,
                collection_name=collection,
                embedding=self.embeddings
            )
        return self._stores[collection]

    async def add_documents(self, collection: str, docs: list[Document]) -> int:
        """Batch upsert documents with deduplication by source path."""
        if not docs:
            return 0

        store = self.get_store(collection)

        # Deduplicate: delete existing chunks from same source
        sources = {doc.metadata.get("source") for doc in docs if doc.metadata.get("source")}
        for source in sources:
            self.client.delete(
                collection_name=collection,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[models.FieldCondition(
                            key="metadata.source",
                            match=models.MatchValue(value=source)
                        )]
                    )
                )
            )

        await store.aadd_documents(docs)
        return len(docs)

    def get_retriever(self, collections: Optional[list[str]] = None, top_k: int = 10):
        """Returns a multi-collection retriever."""
        from langchain.retrievers import MergerRetriever

        if collections is None:
            collections = list(COLLECTIONS.keys())

        retrievers = [
            self.get_store(c).as_retriever(search_kwargs={"k": top_k})
            for c in collections
        ]
        return MergerRetriever(retrievers=retrievers)

    def search(self, query: str, collections: Optional[list[str]] = None,
               top_k: int = 8, filters: Optional[dict] = None) -> list[Document]:
        """Synchronous search across specified collections."""
        if collections is None:
            collections = list(COLLECTIONS.keys())

        qdrant_filter = None
        if filters:
            conditions = [
                models.FieldCondition(key=f"metadata.{k}", match=models.MatchValue(value=v))
                for k, v in filters.items()
            ]
            qdrant_filter = models.Filter(must=conditions)

        all_docs = []
        for col in collections:
            store = self.get_store(col)
            results = store.similarity_search_with_score(
                query, k=top_k,
                filter=qdrant_filter
            )
            for doc, score in results:
                doc.metadata["_score"] = score
                doc.metadata["_collection"] = col
                all_docs.append(doc)

        # Sort by score descending
        all_docs.sort(key=lambda d: d.metadata.get("_score", 0), reverse=True)
        return all_docs[:top_k]

    def get_stats(self) -> dict:
        """Return collection stats."""
        stats = {}
        for name in COLLECTIONS:
            try:
                info = self.client.get_collection(name)
                stats[name] = {
                    "count": info.points_count or 0,
                    "status": info.status.value if hasattr(info.status, "value") else str(info.status)
                }
            except Exception as e:
                stats[name] = {"count": 0, "status": "error", "error": str(e)}
        return stats


# Singleton
vector_store = VectorStoreManager()
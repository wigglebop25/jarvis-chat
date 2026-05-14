"""Vector store module for semantic search and RAG."""

from .store import VectorStore, get_vector_store
from .errors import VectorStoreError

__all__ = [
    "VectorStore",
    "get_vector_store",
    "VectorStoreError",
]

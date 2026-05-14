"""Vector store facade for backward compatibility."""

from .vector.store import VectorStore, get_vector_store

__all__ = ["VectorStore", "get_vector_store"]

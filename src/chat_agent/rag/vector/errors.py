"""Vector store exceptions."""


class VectorStoreError(Exception):
    """Base exception for vector store operations."""
    pass


class VectorStoreDatabaseError(VectorStoreError):
    """Raised when database operation fails."""
    pass


class VectorStoreEmbeddingError(VectorStoreError):
    """Raised when embedding operation fails."""
    pass


class VectorStoreSearchError(VectorStoreError):
    """Raised when search operation fails."""
    pass

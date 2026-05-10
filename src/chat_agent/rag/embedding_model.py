"""Lazy-loaded embedding model using Sentence-Transformers."""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

_model = None
_model_name = "all-MiniLM-L6-v2"


def get_embedding_model():
    """Get or load the embedding model (lazy-loaded on first use)."""
    global _model
    if _model is None:
        logger.info(f"Loading embedding model: {_model_name}")
        try:
            from sentence_transformers import SentenceTransformer
            _model = SentenceTransformer(_model_name)
            logger.info(f"Embedding model loaded successfully (dimension: {_model.get_sentence_embedding_dimension()})")
        except ImportError:
            logger.error("sentence-transformers not installed. Install with: uv pip install sentence-transformers")
            raise
    return _model


def embed_text(text: str) -> Optional[list[float]]:
    """
    Embed a single text string into a vector.
    
    Args:
        text: Text to embed
        
    Returns:
        List of floats representing the embedding, or None if embedding fails
    """
    try:
        model = get_embedding_model()
        embedding = model.encode(text, convert_to_tensor=False)
        return embedding.tolist()
    except Exception as e:
        logger.warning(f"Failed to embed text: {e}")
        return None


def embed_texts_batch(texts: list[str]) -> Optional[list[list[float]]]:
    """
    Embed multiple texts efficiently in batch.
    
    Args:
        texts: List of texts to embed
        
    Returns:
        List of embeddings, or None if embedding fails
    """
    try:
        if not texts:
            return []
        model = get_embedding_model()
        embeddings = model.encode(texts, convert_to_tensor=False)
        return embeddings.tolist()
    except Exception as e:
        logger.warning(f"Failed to batch embed texts: {e}")
        return None


def get_embedding_dimension() -> int:
    """Get the dimension of embeddings from this model."""
    model = get_embedding_model()
    # Explicitly check/assert to satisfy return type requirements
    if model is None:
        return 384 # Default for MiniLM
    return int(model.get_sentence_embedding_dimension())

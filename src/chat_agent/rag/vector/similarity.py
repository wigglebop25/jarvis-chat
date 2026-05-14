"""Similarity metrics for vector search."""

import math
from typing import Optional


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    dot_product = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    
    if mag_a == 0 or mag_b == 0:
        return 0.0
    
    return dot_product / (mag_a * mag_b)


def euclidean_distance(a: list[float], b: list[float]) -> float:
    """Calculate Euclidean distance between two vectors."""
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


class SimilarityThreshold:
    """Configurable similarity thresholds."""
    
    DEFAULT_THRESHOLD = 0.3
    MIN_THRESHOLD = 0.0
    MAX_THRESHOLD = 1.0
    
    @staticmethod
    def validate(threshold: Optional[float]) -> float:
        """Validate and return similarity threshold."""
        if threshold is None:
            return SimilarityThreshold.DEFAULT_THRESHOLD
        
        if not (SimilarityThreshold.MIN_THRESHOLD <= threshold <= SimilarityThreshold.MAX_THRESHOLD):
            raise ValueError(f"Threshold must be between {SimilarityThreshold.MIN_THRESHOLD} and {SimilarityThreshold.MAX_THRESHOLD}")
        
        return threshold

"""Search functionality for vector store."""

import json
import logging
import sqlite3
from pathlib import Path
from typing import Any, Optional

from .similarity import cosine_similarity, SimilarityThreshold
from .errors import VectorStoreSearchError

logger = logging.getLogger(__name__)


class VectorSearch:
    """Handles semantic search over vectors."""
    
    def __init__(self, db_path: Path):
        """Initialize vector search."""
        self.db_path = db_path
    
    def semantic_search(
        self,
        query_embedding: list[float],
        entity_type: Optional[str] = None,
        top_k: int = 3,
        threshold: Optional[float] = None,
    ) -> list[dict[str, Any]]:
        """
        Semantic search using vector similarity.
        
        Args:
            query_embedding: Embedding vector of query
            entity_type: Filter by entity type (optional)
            top_k: Number of top results to return
            threshold: Similarity threshold (default: 0.3)
            
        Returns:
            List of matching entities sorted by similarity
        """
        try:
            threshold = SimilarityThreshold.validate(threshold)
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            if entity_type:
                cursor.execute("""
                    SELECT id, entity_type, entity_name, text_content, embedding, metadata, updated_at
                    FROM embeddings
                    WHERE entity_type = ?
                    ORDER BY updated_at DESC
                """, (entity_type,))
            else:
                cursor.execute("""
                    SELECT id, entity_type, entity_name, text_content, embedding, metadata, updated_at
                    FROM embeddings
                    ORDER BY updated_at DESC
                """)
            
            rows = cursor.fetchall()
            conn.close()
            
            scored_results = []
            for row in rows:
                try:
                    stored_embedding = json.loads(row[4].decode())
                    similarity = cosine_similarity(query_embedding, stored_embedding)
                    
                    if similarity > threshold:
                        scored_results.append({
                            'id': row[0],
                            'entity_type': row[1],
                            'entity_name': row[2],
                            'text_content': row[3],
                            'similarity_score': similarity,
                            'metadata': json.loads(row[5]),
                            'updated_at': row[6],
                        })
                except (json.JSONDecodeError, UnicodeDecodeError):
                    logger.warning(f"Failed to decode embedding for {row[0]}")
                    continue
            
            scored_results.sort(key=lambda x: x['similarity_score'], reverse=True)
            return scored_results[:top_k]
            
        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            raise VectorStoreSearchError(f"Search failed: {e}")
    
    def search_with_threshold(
        self,
        query_embedding: list[float],
        threshold: float = 0.5,
    ) -> list[dict[str, Any]]:
        """
        Search for all results above threshold.
        
        Args:
            query_embedding: Embedding vector of query
            threshold: Minimum similarity threshold
            
        Returns:
            All matching entities above threshold
        """
        try:
            threshold = SimilarityThreshold.validate(threshold)
            results = self.semantic_search(query_embedding, threshold=threshold, top_k=999)
            return results
            
        except Exception as e:
            logger.error(f"Threshold search failed: {e}")
            raise VectorStoreSearchError(f"Threshold search failed: {e}")

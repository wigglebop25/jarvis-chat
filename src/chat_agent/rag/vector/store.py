"""Core vector store implementation."""

import json
import logging
import sqlite3
from pathlib import Path
from typing import Any, Optional

from ..embedding_model import embed_text
from .errors import VectorStoreEmbeddingError, VectorStoreDatabaseError
from .indexing import IndexManager
from .search import VectorSearch

logger = logging.getLogger(__name__)

CACHE_DIR = Path.home() / ".jarvis" / "cache"
VECTOR_DB_PATH = CACHE_DIR / "vector_store.db"


def _ensure_db_exists():
    """Ensure vector store database exists."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    
    conn = sqlite3.connect(str(VECTOR_DB_PATH))
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS embeddings (
            id TEXT PRIMARY KEY,
            entity_type TEXT NOT NULL,
            entity_name TEXT,
            text_content TEXT NOT NULL,
            embedding BLOB NOT NULL,
            metadata TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            ttl_seconds INTEGER
        )
    """)
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS user_actions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            query TEXT NOT NULL,
            tool_name TEXT,
            result_type TEXT,
            keywords TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS mood_correlations (
            id TEXT PRIMARY KEY,
            mood_keyword TEXT NOT NULL,
            correlated_entity_id TEXT,
            entity_type TEXT,
            confidence REAL DEFAULT 0.5,
            sample_count INTEGER DEFAULT 1,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    conn.commit()
    conn.close()
    
    IndexManager(VECTOR_DB_PATH).create_indexes()
    logger.info(f"Vector store initialized at {VECTOR_DB_PATH}")


class VectorStore:
    """Local vector store for semantic search."""
    
    def __init__(self):
        """Initialize the vector store."""
        _ensure_db_exists()
        self.search = VectorSearch(VECTOR_DB_PATH)
        self.index = IndexManager(VECTOR_DB_PATH)
    
    def embed_and_cache(
        self,
        entity_id: str,
        entity_type: str,
        text_content: str,
        entity_name: Optional[str] = None,
        metadata: Optional[dict] = None,
        ttl_hours: int = 1,
    ) -> bool:
        """Embed text and cache with metadata."""
        try:
            embedding = embed_text(text_content)
            if embedding is None:
                logger.warning(f"Failed to embed {entity_type}:{entity_id}")
                return False
            
            embedding_bytes = json.dumps(embedding).encode()
            meta_str = json.dumps(metadata or {})
            ttl_seconds = ttl_hours * 3600 if ttl_hours > 0 else None
            
            conn = sqlite3.connect(str(VECTOR_DB_PATH))
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT OR REPLACE INTO embeddings 
                (id, entity_type, entity_name, text_content, embedding, metadata, ttl_seconds, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """, (entity_id, entity_type, entity_name, text_content, embedding_bytes, meta_str, ttl_seconds))
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            logger.error(f"Failed to embed and cache: {e}")
            raise VectorStoreEmbeddingError(f"Embedding failed: {e}")
    
    def semantic_search(
        self,
        query: str,
        entity_type: Optional[str] = None,
        top_k: int = 3,
    ) -> list[dict[str, Any]]:
        """Semantic search using vector similarity."""
        try:
            query_embedding = embed_text(query)
            if query_embedding is None:
                logger.warning(f"Failed to embed query: {query}")
                return []
            
            return self.search.semantic_search(query_embedding, entity_type, top_k)
            
        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            return []
    
    def clear_stale(self) -> int:
        """Remove embeddings older than their TTL."""
        try:
            conn = sqlite3.connect(str(VECTOR_DB_PATH))
            cursor = conn.cursor()
            
            cursor.execute("""
                DELETE FROM embeddings
                WHERE ttl_seconds IS NOT NULL
                AND datetime(updated_at, '+' || ttl_seconds || ' seconds') < CURRENT_TIMESTAMP
            """)
            
            deleted_count = cursor.rowcount
            conn.commit()
            conn.close()
            
            if deleted_count > 0:
                logger.info(f"Cleared {deleted_count} stale embeddings")
            
            return deleted_count
            
        except Exception as e:
            logger.error(f"Failed to clear stale embeddings: {e}")
            raise VectorStoreDatabaseError(f"Cleanup failed: {e}")
    
    def get_by_type(self, entity_type: str) -> list[dict[str, Any]]:
        """Get all embeddings of a specific type."""
        try:
            conn = sqlite3.connect(str(VECTOR_DB_PATH))
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT id, entity_name, text_content, metadata, updated_at
                FROM embeddings
                WHERE entity_type = ?
                AND (ttl_seconds IS NULL OR datetime(updated_at, '+' || ttl_seconds || ' seconds') >= CURRENT_TIMESTAMP)
                ORDER BY updated_at DESC
            """, (entity_type,))
            
            rows = cursor.fetchall()
            conn.close()
            
            return [
                {
                    'id': row[0],
                    'entity_name': row[1],
                    'text_content': row[2],
                    'metadata': json.loads(row[3]),
                    'updated_at': row[4],
                }
                for row in rows
            ]
            
        except Exception as e:
            logger.error(f"Failed to get embeddings by type: {e}")
            return []
    
    def log_user_action(
        self,
        query: str,
        tool_name: Optional[str] = None,
        result_type: Optional[str] = None,
        keywords: Optional[list[str]] = None,
    ) -> bool:
        """Log user action for mood analysis."""
        try:
            keywords_str = ",".join(keywords) if keywords else None
            
            conn = sqlite3.connect(str(VECTOR_DB_PATH))
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO user_actions (query, tool_name, result_type, keywords)
                VALUES (?, ?, ?, ?)
            """, (query, tool_name, result_type, keywords_str))
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            logger.error(f"Failed to log user action: {e}")
            return False


# Singleton instance
_vector_store = None


def get_vector_store() -> VectorStore:
    """Get or create the singleton vector store instance."""
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
    return _vector_store

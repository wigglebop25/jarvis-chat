"""Cache warming service for RAG system startup."""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class CacheWarmer:
    """Warm up cache with frequently-used data on startup."""
    
    @staticmethod
    async def warm_cache(custom_playlists: Optional[list] = None) -> int:
        """
        Populate cache with common queries on startup.
        
        Args:
            custom_playlists: Optional list of (id, name, desc) tuples
            
        Returns:
            Count of items cached
        """
        try:
            from ..retriever import get_rag_retriever
            
            retriever = get_rag_retriever()
            
            playlists = custom_playlists or [
                ("spotify:pl:1", "My Top Tracks", "Your top 50 most-played songs"),
                ("spotify:pl:2", "Discover Weekly", "Personalized recommendations"),
                ("spotify:pl:3", "Release Radar", "New releases from followed artists"),
            ]
            
            for pl_id, pl_name, description in playlists:
                try:
                    retriever.cache_playlist(pl_id, pl_name, description, ttl_hours=24)
                except Exception as e:
                    logger.debug(f"Failed to cache {pl_name}: {e}")
            
            logger.info(f"Cache warmed with {len(playlists)} playlists")
            return len(playlists)
            
        except Exception as e:
            logger.debug(f"Cache warming failed (non-critical): {e}")
            return 0
    
    @staticmethod
    async def warm_common_queries() -> int:
        """Pre-compute embeddings for common queries."""
        try:
            from ..embedding_model import embed_texts_batch
            
            common_queries = [
                "show my playlists",
                "what's playing",
                "play music",
                "pause",
                "next track",
                "previous track",
                "shuffle",
                "repeat",
            ]
            
            embed_texts_batch(common_queries)
            logger.info(f"Pre-computed embeddings for {len(common_queries)} queries")
            return len(common_queries)
            
        except Exception as e:
            logger.debug(f"Query warming failed: {e}")
            return 0

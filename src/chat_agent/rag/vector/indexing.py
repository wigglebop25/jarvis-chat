"""Index management for vector store."""

import logging
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)


class IndexManager:
    """Manages vector store indexes."""
    
    def __init__(self, db_path: Path):
        """Initialize index manager."""
        self.db_path = db_path
    
    def create_indexes(self) -> None:
        """Create all necessary indexes."""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            cursor.execute("""CREATE INDEX IF NOT EXISTS idx_entity_type ON embeddings(entity_type)""")
            cursor.execute("""CREATE INDEX IF NOT EXISTS idx_updated_at ON embeddings(updated_at)""")
            
            conn.commit()
            conn.close()
            logger.info("Indexes created successfully")
            
        except Exception as e:
            logger.error(f"Failed to create indexes: {e}")
            raise
    
    def rebuild_index(self) -> None:
        """Rebuild all indexes."""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            cursor.execute("REINDEX")
            conn.commit()
            conn.close()
            
            logger.info("Indexes rebuilt successfully")
            
        except Exception as e:
            logger.error(f"Failed to rebuild indexes: {e}")
            raise
    
    def optimize_index(self) -> None:
        """Optimize indexes for better performance."""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            cursor.execute("VACUUM")
            cursor.execute("PRAGMA optimize")
            
            conn.commit()
            conn.close()
            
            logger.info("Indexes optimized successfully")
            
        except Exception as e:
            logger.error(f"Failed to optimize indexes: {e}")
            raise

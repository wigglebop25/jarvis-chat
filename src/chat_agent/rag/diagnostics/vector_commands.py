"""Vector store statistics and diagnostics."""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class VectorStoreCommands:
    """Commands for vector store diagnostics."""
    
    @staticmethod
    def show_vector_store_stats() -> str:
        """Show vector store statistics."""
        try:
            import sqlite3
            
            db_path = Path.home() / ".jarvis" / "cache" / "vector_store.db"
            
            if not db_path.exists():
                return "Vector store not initialized"
            
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()
            
            lines = ["Vector Store Statistics:", "=" * 60]
            
            # Count embeddings by type
            cursor.execute(
                "SELECT entity_type, COUNT(*) as count FROM embeddings "
                "GROUP BY entity_type"
            )
            type_counts = cursor.fetchall()
            
            total_embeddings = 0
            for entity_type, count in type_counts:
                lines.append(f"  {entity_type:20s}: {count:5d} embeddings")
                total_embeddings += count
            
            # Get user actions count
            cursor.execute("SELECT COUNT(*) FROM user_actions")
            action_count = cursor.fetchone()[0]
            
            # Get mood correlations count
            cursor.execute("SELECT COUNT(*) FROM mood_correlations")
            mood_count = cursor.fetchone()[0]
            
            conn.close()
            
            lines.append("-" * 60)
            lines.append(f"Total Embeddings: {total_embeddings}")
            lines.append(f"User Actions Logged: {action_count}")
            lines.append(f"Mood Correlations: {mood_count}")
            
            return "\n".join(lines)
            
        except Exception as e:
            logger.error(f"Failed to get vector store stats: {e}")
            return f"Error: {e}"
    
    @staticmethod
    def show_recent_queries(limit: int = 10) -> str:
        """Show recently logged user queries."""
        try:
            import sqlite3
            
            db_path = Path.home() / ".jarvis" / "cache" / "vector_store.db"
            
            if not db_path.exists():
                return "Vector store not initialized"
            
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()
            
            cursor.execute(
                "SELECT query, tool_name, timestamp FROM user_actions "
                "ORDER BY timestamp DESC LIMIT ?",
                (limit,)
            )
            
            results = cursor.fetchall()
            conn.close()
            
            if not results:
                return "No recent queries found"
            
            lines = [f"Recent Queries (last {limit}):", "=" * 80]
            
            for query, tool_name, timestamp in results:
                lines.append(f"[{timestamp}] Q: {query}")
                if tool_name:
                    lines.append(f"           T: {tool_name}")
            
            return "\n".join(lines)
            
        except Exception as e:
            logger.error(f"Failed to get recent queries: {e}")
            return f"Error: {e}"

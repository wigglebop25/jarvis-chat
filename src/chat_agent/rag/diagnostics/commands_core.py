"""Core RAG diagnostic commands."""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class RAGCoreCommands:
    """Core RAG diagnostic commands."""
    
    @staticmethod
    def show_cache_stats() -> str:
        """Show tool cache statistics."""
        try:
            from ...tools.tool_cache import get_cache_stats
            
            stats = get_cache_stats()
            
            if not stats:
                return "Cache is empty"
            
            lines = ["Tool Cache Statistics:", "=" * 60]
            
            total_entries = 0
            total_hits = 0
            
            for tool_name, stats_dict in stats.items():
                entries = stats_dict.get('entries', 0)
                hits = stats_dict.get('hits', 0)
                total_entries += entries
                total_hits += hits
                hit_rate = (hits / (hits + 1)) * 100 if hits else 0
                
                lines.append(
                    f"{tool_name:30s} | Entries: {entries:3d} | "
                    f"Hits: {hits:5d} ({hit_rate:.1f}%)"
                )
            
            lines.append("-" * 60)
            lines.append(f"{'TOTAL':30s} | Entries: {total_entries:3d} | Hits: {total_hits:5d}")
            
            return "\n".join(lines)
            
        except Exception as e:
            logger.error(f"Failed to get cache stats: {e}")
            return f"Error: {e}"
    
    @staticmethod
    def clear_cache(tool_name: Optional[str] = None) -> str:
        """Clear tool cache."""
        try:
            from ...tools.tool_cache import clear_tool_cache
            
            deleted = clear_tool_cache(tool_name)
            
            if tool_name:
                return f"✓ Cleared {deleted} cache entries for {tool_name}"
            else:
                return f"✓ Cleared {deleted} total cache entries"
                
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")
            return f"✗ Error: {e}"

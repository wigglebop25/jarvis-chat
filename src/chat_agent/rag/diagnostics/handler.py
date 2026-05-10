"""RAG diagnostic command handler."""

import logging
from typing import Optional

from .commands_core import RAGCoreCommands
from .mood_commands import MoodCommands
from .vector_commands import VectorStoreCommands

logger = logging.getLogger(__name__)


async def handle_rag_command(command: str, args: Optional[list[str]] = None) -> str:
    """
    Handle RAG diagnostic commands.
    
    Usage:
        /rag cache-stats              - Show cache statistics
        /rag vector-stats             - Show vector store statistics
        /rag mood-analysis            - Run mood analysis
        /rag clear-cache [tool]       - Clear cache
        /rag recent-queries [n]       - Show recent queries
        /rag mood-correlations        - Show discovered moods
        /rag mood-stats               - Show mood statistics
    
    Args:
        command: Command name
        args: Optional command arguments
        
    Returns:
        Command result string
    """
    args = args or []
    
    if command == "cache-stats":
        return RAGCoreCommands.show_cache_stats()
    
    elif command == "vector-stats":
        return VectorStoreCommands.show_vector_store_stats()
    
    elif command == "mood-analysis":
        return await MoodCommands.run_mood_analysis()
    
    elif command == "clear-cache":
        tool_name: Optional[str] = args[0] if args else None
        return RAGCoreCommands.clear_cache(tool_name)
    
    elif command == "recent-queries":
        limit = int(args[0]) if args else 10
        return VectorStoreCommands.show_recent_queries(limit)
    
    elif command == "mood-correlations":
        return MoodCommands.show_mood_correlations()
    
    elif command == "mood-stats":
        return MoodCommands.show_mood_stats()
    
    else:
        return f"Unknown RAG command: {command}"


class RAGCommandHandler:
    """Unified interface for RAG diagnostics."""
    
    @staticmethod
    async def execute(command: str, args: Optional[list[str]] = None) -> str:
        """Execute RAG command."""
        return await handle_rag_command(command, args)
    
    @staticmethod
    def get_help() -> str:
        """Get RAG command help."""
        return """
RAG Diagnostic Commands:
  /rag cache-stats         - Show tool cache statistics
  /rag vector-stats        - Show vector store statistics
  /rag mood-analysis       - Run mood correlation analysis
  /rag clear-cache [tool]  - Clear cache (optionally for specific tool)
  /rag recent-queries [n]  - Show recent queries (default: 10)
  /rag mood-correlations   - Show discovered mood patterns
  /rag mood-stats          - Show mood analysis statistics

Examples:
  /rag cache-stats
  /rag clear-cache getMyPlaylists
  /rag recent-queries 20
  /rag mood-correlations
        """

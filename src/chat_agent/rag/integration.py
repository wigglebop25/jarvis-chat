"""Phase 6: Integration - Parallel RAG execution and service management.

This module provides high-level integration functions for RAG system.
Actual implementations are in the services/ submodule.
"""

# Re-export for backward compatibility
from .services import (
    CacheWarmer,
    MoodAnalysisBackgroundTask,
    ParallelRAGExecutor,
    get_rag_manager,
    initialize_rag_services,
    shutdown_rag_services,
)

__all__ = [
    "ParallelRAGExecutor",
    "MoodAnalysisBackgroundTask",
    "CacheWarmer",
    "get_rag_manager",
    "initialize_rag_services",
    "shutdown_rag_services",
]


"""RAG (Retrieval-Augmented Generation) module for context-aware responses."""

from .embedding_model import embed_text, embed_texts_batch, get_embedding_model
from .mood_analyzer import MoodAnalyzer, get_mood_analyzer, run_mood_correlation_analysis
from .retriever import RagRetriever, get_rag_retriever
from .vector_store import VectorStore, get_vector_store

# Services: background tasks, execution, lifecycle management
from .services import (
    CacheWarmer,
    MoodAnalysisBackgroundTask,
    MoodAnalysisTask,
    ParallelRAGExecutor,
    get_rag_manager,
    initialize_rag_services,
    shutdown_rag_services,
)

# Diagnostics: monitoring and debugging
from .diagnostics import (
    RAGCommandHandler,
    RAGCoreCommands,
    handle_rag_command,
)

# Tests: verification suite
from .tests import (
    RAGTestSuite,
    run_rag_verification_tests,
)

__all__ = [
    # Core RAG components
    "VectorStore",
    "get_vector_store",
    "RagRetriever",
    "get_rag_retriever",
    "MoodAnalyzer",
    "get_mood_analyzer",
    "run_mood_correlation_analysis",
    "embed_text",
    "embed_texts_batch",
    "get_embedding_model",
    # Services
    "ParallelRAGExecutor",
    "MoodAnalysisBackgroundTask",
    "MoodAnalysisTask",
    "CacheWarmer",
    "get_rag_manager",
    "initialize_rag_services",
    "shutdown_rag_services",
    # Diagnostics
    "RAGCommandHandler",
    "RAGCoreCommands",
    "handle_rag_command",
    # Tests
    "RAGTestSuite",
    "run_rag_verification_tests",
]

"""Classify intent and retrieve context."""

import logging
from typing import Any, Dict, Optional
from langchain_core.runnables import RunnableConfig

from ..state import AgentState
from .helpers import get_transcript_from_state

logger = logging.getLogger(__name__)


async def classify_intent_node(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    """
    Node that retrieves RAG context based on the current intent and query.
    
    This represents the "Retrieval" and "Augmentation" steps of the RAG pipeline.
    The retrieved context is added to the state and later injected into the prompt.
    """
    logger.info("[LangGraph] Entering classify_intent_node")
    configurable = config.get("configurable", {})
    from ...rag.retriever import RagRetriever
    rag_retriever: Optional[RagRetriever] = configurable.get("rag_retriever")
    
    if not rag_retriever:
        logger.warning("[LangGraph] RAG retriever not found in config")
        return {"rag_context": ""}
        
    transcript = get_transcript_from_state(state)
    intent = state.get("intent")
    intent_type = intent.get("type") if intent else None
    
    logger.info(f"[LangGraph] Retrieving context for intent: {intent_type}")
    context_data = rag_retriever.retrieve_context(transcript, intent=intent_type)
    formatted_context = rag_retriever.format_context_for_prompt(context_data)
    logger.info(f"[LangGraph] Formatted context length: {len(formatted_context)}")
    
    return {"rag_context": formatted_context}

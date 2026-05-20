"""
rag/augmentation.py
────────────────────
RAG-based message augmentation for LLM requests.
Safely integrates RAG without breaking if vector store fails.
"""
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def augment_messages_with_rag(
    messages: list[dict[str, str]],
    detected_intent: Optional[str] = None,
) -> list[dict[str, str]]:
    """
    Inject RAG context into the system message before the LLM call.
    Returns original messages unchanged if retrieval fails or finds nothing.
    """
    try:
        from . import get_rag_retriever
        retriever = get_rag_retriever()

        # Use the last user message as the retrieval query
        query = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                query = msg.get("content", "")
                break

        if not query:
            logger.debug("No user query found, skipping RAG augmentation")
            return messages

        context = retriever.retrieve_context(query, intent=detected_intent, top_k=1)
        if not any(context.values()):
            logger.debug("No RAG context retrieved")
            return messages

        context_str = retriever.format_context_for_prompt(context).strip()
        if not context_str:
            logger.debug("Failed to format RAG context")
            return messages

        augmented: list[dict[str, str]] = []
        system_found = False
        for msg in messages:
            if msg.get("role") == "system" and not system_found:
                augmented.append({
                    "role": "system",
                    "content": f"{msg.get('content', '')}\n\n[CONTEXT FROM MEMORY]\n{context_str}",
                })
                system_found = True
            else:
                augmented.append(msg)

        if not system_found:
            augmented.insert(0, {
                "role": "system",
                "content": f"[CONTEXT FROM MEMORY]\n{context_str}",
            })

        logger.debug(f"RAG augmented prompt with {len(context_str)} chars of context")
        return augmented

    except Exception as e:
        logger.debug(f"RAG augmentation failed (graceful fallback): {e}")
        return messages


async def augment_messages_with_rag_async(
    messages: list[dict[str, str]],
    detected_intent: Optional[str] = None,
) -> list[dict[str, str]]:
    """Async shim — delegates to sync version (RAG is fast: 5–10 ms)."""
    return augment_messages_with_rag(messages, detected_intent)

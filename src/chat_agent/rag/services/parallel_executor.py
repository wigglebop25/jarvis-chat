"""Parallel execution of RAG retrieval and tool operations."""

import asyncio
import logging
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


class ParallelRAGExecutor:
    """Execute RAG context retrieval and tool execution concurrently."""
    
    @staticmethod
    async def execute_with_rag_context(
        user_query: str,
        tool_executor: Callable[..., Any],
        detected_intent: Optional[str] = None,
        tool_name: Optional[str] = None,
        tool_args: Optional[dict] = None,
    ) -> tuple[dict[str, Any], Any]:
        """
        Execute tool while retrieving RAG context in parallel.
        
        Rationale:
            Sequential: context (10ms) + tool (500ms) = 510ms
            Parallel:  max(context 10ms, tool 500ms) = 500ms
        
        Args:
            user_query: User's original query
            tool_executor: Async function to execute
            detected_intent: Intent for context filtering
            tool_name: Tool being executed
            tool_args: Tool arguments
            
        Returns:
            Tuple of (rag_context, tool_result)
        """
        try:
            context, tool_result = await asyncio.gather(
                _retrieve_rag_context_async(user_query, detected_intent),
                tool_executor(**(tool_args or {})),
                return_exceptions=True,
            )
            
            if isinstance(context, Exception):
                logger.debug(f"RAG context retrieval failed: {context}")
                context = {}
            
            if isinstance(tool_result, Exception):
                logger.error(f"Tool execution failed: {tool_result}")
                raise tool_result
            
            return context, tool_result
            
        except Exception as e:
            logger.error(f"Parallel execution failed: {e}")
            raise
    
    @staticmethod
    async def execute_multiple_tools(
        tool_calls: list[tuple[str, dict]],
        executor_func: Callable,
    ) -> list[tuple[str, Any]]:
        """
        Execute multiple tools in parallel.
        
        Args:
            tool_calls: List of (tool_name, args) tuples
            executor_func: Async function to execute single tool
            
        Returns:
            List of (tool_name, result) tuples
        """
        try:
            tasks = [
                executor_func(tool_name, **args)
                for tool_name, args in tool_calls
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            return [
                (tool_name, result if not isinstance(result, Exception) else None)
                for (tool_name, _), result in zip(tool_calls, results)
            ]
            
        except Exception as e:
            logger.error(f"Batch execution failed: {e}")
            raise


async def _retrieve_rag_context_async(
    user_query: str,
    detected_intent: Optional[str] = None,
) -> dict[str, Any]:
    """Retrieve RAG context asynchronously."""
    try:
        from ..retriever import get_rag_retriever
        
        retriever = get_rag_retriever()
        
        loop = asyncio.get_event_loop()
        context = await loop.run_in_executor(
            None,
            lambda: retriever.retrieve_context(user_query, detected_intent)
        )
        
        return context
        
    except Exception as e:
        logger.debug(f"RAG context retrieval failed: {e}")
        return {}

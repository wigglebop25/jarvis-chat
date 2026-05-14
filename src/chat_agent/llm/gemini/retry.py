"""Retry logic with exponential backoff for Gemini API calls."""

import asyncio
import logging
from typing import Callable, TypeVar, Any

logger = logging.getLogger(__name__)

T = TypeVar("T")


class MaxRetriesExceeded(Exception):
    """Raised when maximum number of retries is exceeded."""


async def retry_with_backoff(
    func: Callable[..., Any],
    max_retries: int = 3,
    initial_delay: float = 0.5,
    max_delay: float = 30.0,
    backoff_multiplier: float = 2.0,
    retryable_exceptions: tuple = (Exception,),
) -> Any:
    """
    Retry an async function with exponential backoff.
    
    Args:
        func: Async function to retry
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay in seconds
        max_delay: Maximum delay in seconds
        backoff_multiplier: Multiply delay by this after each retry
        retryable_exceptions: Tuple of exceptions to retry on
    
    Returns:
        Result from func
        
    Raises:
        MaxRetriesExceeded: If all retries are exhausted
        Last exception encountered if not retryable
    """
    delay = initial_delay
    
    for attempt in range(max_retries + 1):
        try:
            return await func()
        except retryable_exceptions as e:
            if attempt >= max_retries:
                logger.error(f"Max retries ({max_retries}) exceeded. Last error: {e}")
                raise MaxRetriesExceeded(f"Failed after {max_retries} retries: {e}") from e
            
            delay = min(delay * backoff_multiplier, max_delay)
            logger.warning(f"Retry attempt {attempt + 1}/{max_retries} after {delay:.1f}s. Error: {e}")
            await asyncio.sleep(delay)
        except Exception as e:
            logger.error(f"Non-retryable exception: {e}")
            raise

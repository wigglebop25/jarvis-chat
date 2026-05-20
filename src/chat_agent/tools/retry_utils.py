"""
Retry utilities for handling transient failures with exponential backoff.

Provides common retry logic for handling rate limits, timeouts, and
connection errors with configurable exponential backoff strategy.
"""

import asyncio
import logging
import random
from typing import Awaitable, Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Common exceptions that should trigger retry
RETRIABLE_EXCEPTIONS = (
    asyncio.TimeoutError,
    ConnectionError,
    TimeoutError,
)


class RateLimitError(RuntimeError):
    """Raised when rate limit (429) is encountered."""
    pass


class RetryExceededError(RuntimeError):
    """Raised when maximum retries are exceeded."""
    pass


class MCPRateLimitError(RateLimitError):
    """Raised when MCP rate limit is encountered."""
    pass


class MCPTimeoutError(RuntimeError):
    """Raised when an MCP call times out."""
    pass


async def async_retry_with_backoff(
    coro_func: Callable[..., Awaitable[T]],
    *args,
    max_retries: int = 3,
    base_delay: float = 1.0,
    backoff_factor: float = 2.0,
    jitter: bool = True,
    retryable_exceptions: tuple = (RateLimitError, asyncio.TimeoutError, ConnectionError, TimeoutError),
    **kwargs,
) -> T:
    """
    Execute an async coroutine with exponential backoff retry logic.
    
    Handles transient failures like timeouts and connection errors.
    Uses exponential backoff: delay = base_delay * (backoff_factor ** attempt)
    
    Args:
        coro_func: Async function to call
        *args: Positional arguments for coro_func
        max_retries: Maximum number of retries (default 3)
        base_delay: Initial delay in seconds (default 1.0)
        backoff_factor: Multiplier for exponential backoff (default 2.0)
        jitter: Whether to add random jitter to delay
        retryable_exceptions: Tuple of exceptions that should trigger a retry
        **kwargs: Keyword arguments for coro_func
    
    Returns:
        Result from successful call to coro_func
    """
    last_exception = None
    
    for attempt in range(max_retries + 1):
        try:
            if attempt > 0:
                logger.debug(
                    f"Calling {coro_func.__name__} (attempt {attempt + 1}/{max_retries + 1})"
                )
            return await coro_func(*args, **kwargs)
        
        except retryable_exceptions as e:
            last_exception = e
            if attempt < max_retries:
                wait_seconds = base_delay * (backoff_factor ** attempt)
                if jitter:
                    wait_seconds += random.uniform(0, wait_seconds * 0.1)
                
                logger.warning(
                    f"Retryable error {type(e).__name__} hit on attempt {attempt + 1}/{max_retries + 1}, "
                    f"waiting {wait_seconds:.2f}s before retry: {e}"
                )
                await asyncio.sleep(wait_seconds)
                continue
            else:
                logger.error(
                    f"Max retries ({max_retries}) exceeded for {coro_func.__name__}. Last error: {e}"
                )
                # Task 13: Generic error for new callers, but maintain specific 
                # error raising for existing tests/logic that expect them.
                if isinstance(e, MCPRateLimitError):
                    raise e
                if isinstance(e, (asyncio.TimeoutError, TimeoutError)):
                    raise MCPTimeoutError(f"Call to {coro_func.__name__} timed out after {max_retries} retries") from e
                
                raise RetryExceededError(f"Failed after {max_retries} retries: {e}") from e
        
        except Exception as e:
            # Other exceptions: log and re-raise immediately
            logger.error(
                f"Non-retriable error in {coro_func.__name__}: {type(e).__name__}: {e}"
            )
            raise
    
    # Fallback
    if last_exception:
        raise last_exception
    raise RuntimeError(f"Unexpected error: no exception to raise after {max_retries} attempts")

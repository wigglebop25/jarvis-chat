"""
Retry utilities for handling transient failures with exponential backoff.

Provides common retry logic for handling rate limits, timeouts, and
connection errors with configurable exponential backoff strategy.
"""

import asyncio
import logging
from typing import Awaitable, Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Common exceptions that should trigger retry
RETRIABLE_EXCEPTIONS = (
    asyncio.TimeoutError,
    ConnectionError,
    TimeoutError,
)


class MCPRateLimitError(RuntimeError):
    """Raised when rate limit (429) is encountered."""
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
        **kwargs: Keyword arguments for coro_func
    
    Returns:
        Result from successful call to coro_func
    
    Raises:
        MCPRateLimitError: If rate limit exceeded after all retries
        MCPTimeoutError: If timeout occurs
        Exception: Any other exception that occurs on the final attempt
    
    Example:
        >>> async def fetch_data(url):
        ...     return await client.get(url)
        >>> result = await async_retry_with_backoff(
        ...     fetch_data,
        ...     "https://api.example.com/data",
        ...     max_retries=4,
        ...     base_delay=1.0,
        ... )
    """
    last_exception = None
    
    for attempt in range(max_retries):
        try:
            logger.debug(
                f"Calling {coro_func.__name__} (attempt {attempt + 1}/{max_retries})"
            )
            result = await coro_func(*args, **kwargs)
            
            # Reset logger if we succeeded after retries
            if attempt > 0:
                logger.debug(
                    f"Successfully called {coro_func.__name__} after {attempt} retries"
                )
            
            return result
        
        except MCPRateLimitError as e:
            # Rate limit errors should always retry (unless final attempt)
            last_exception = e
            if attempt < max_retries - 1:
                wait_seconds = base_delay * (backoff_factor ** attempt)
                logger.warning(
                    f"Rate limit hit on attempt {attempt + 1}/{max_retries}, "
                    f"waiting {wait_seconds}s before retry"
                )
                await asyncio.sleep(wait_seconds)
                continue
            else:
                logger.error(
                    f"Rate limit exceeded after {max_retries} retries for "
                    f"{coro_func.__name__}"
                )
                raise
        
        except (asyncio.TimeoutError, TimeoutError) as e:
            # Timeout errors should retry
            last_exception = e
            if attempt < max_retries - 1:
                wait_seconds = base_delay * (backoff_factor ** attempt)
                logger.warning(
                    f"Timeout on attempt {attempt + 1}/{max_retries}, "
                    f"waiting {wait_seconds}s before retry"
                )
                await asyncio.sleep(wait_seconds)
                continue
            else:
                logger.error(
                    f"Timeout exceeded for {coro_func.__name__} after {max_retries} retries"
                )
                raise MCPTimeoutError(
                    f"Call to {coro_func.__name__} timed out after {max_retries} retries"
                ) from e
        
        except ConnectionError as e:
            # Connection errors should retry
            last_exception = e
            if attempt < max_retries - 1:
                wait_seconds = base_delay * (backoff_factor ** attempt)
                logger.warning(
                    f"Connection error on attempt {attempt + 1}/{max_retries}, "
                    f"waiting {wait_seconds}s before retry: {e}"
                )
                await asyncio.sleep(wait_seconds)
                continue
            else:
                logger.error(
                    f"Connection failed for {coro_func.__name__} after {max_retries} retries: {e}"
                )
                raise
        
        except Exception as e:
            # Other exceptions: log and re-raise immediately
            logger.error(
                f"Non-retriable error in {coro_func.__name__}: {type(e).__name__}: {e}"
            )
            raise
    
    # Fallback (should not reach here, but just in case)
    if last_exception:
        raise last_exception
    raise RuntimeError(f"Unexpected error: no exception to raise after {max_retries} attempts")


def create_retry_decorator(
    max_retries: int = 3,
    base_delay: float = 1.0,
    backoff_factor: float = 2.0,
):
    """
    Create a decorator for retrying async functions with exponential backoff.
    
    Args:
        max_retries: Maximum number of retries
        base_delay: Initial delay in seconds
        backoff_factor: Multiplier for exponential backoff
    
    Returns:
        Decorator function
    
    Example:
        >>> @create_retry_decorator(max_retries=4, base_delay=0.5)
        ... async def fetch_data(url):
        ...     return await client.get(url)
    """
    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        async def wrapper(*args, **kwargs) -> T:
            return await async_retry_with_backoff(
                func,
                *args,
                max_retries=max_retries,
                base_delay=base_delay,
                backoff_factor=backoff_factor,
                **kwargs,
            )
        return wrapper
    return decorator

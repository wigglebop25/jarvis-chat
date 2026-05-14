"""Circuit breaker pattern for fault tolerance."""

import logging
import time
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


class CircuitBreakerError(Exception):
    """Raised when circuit breaker is open."""
    pass


class CircuitBreaker:
    """
    Circuit breaker implementation for preventing cascading failures.
    
    States:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Failing, requests fail fast
    - HALF_OPEN: Testing if service recovered
    """
    
    def __init__(
        self,
        failure_threshold: int = 5,
        success_threshold: int = 2,
        timeout_seconds: int = 60,
        name: str = "CircuitBreaker",
    ):
        """
        Initialize circuit breaker.
        
        Args:
            failure_threshold: Failures to trip breaker
            success_threshold: Successes to close breaker from half-open
            timeout_seconds: Time before attempting recovery
            name: Circuit breaker name for logging
        """
        self.failure_threshold = failure_threshold
        self.success_threshold = success_threshold
        self.timeout_seconds = timeout_seconds
        self.name = name
        
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[float] = None
        self.state = "closed"  # closed | open | half_open
    
    async def call(
        self,
        func: Callable,
        *args,
        **kwargs,
    ) -> Any:
        """
        Execute function with circuit breaker protection.
        
        Args:
            func: Async function to call
            *args: Function positional arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Result from function
            
        Raises:
            CircuitBreakerError: If breaker is open
        """
        if self.state == "open":
            if self._should_attempt_reset():
                self.state = "half_open"
                self.success_count = 0
                logger.info(f"{self.name} entering half-open state")
            else:
                raise CircuitBreakerError(
                    f"{self.name} is open (will retry in "
                    f"{self._time_until_retry()}s)"
                )
        
        try:
            result = await func(*args, **kwargs)
            await self._on_success()
            return result
            
        except Exception:
            await self._on_failure()
            raise
    
    async def _on_success(self) -> None:
        """Handle successful call."""
        if self.state == "half_open":
            self.success_count += 1
            if self.success_count >= self.success_threshold:
                self.state = "closed"
                self.failure_count = 0
                logger.info(f"{self.name} closed (recovered)")
        else:
            self.failure_count = 0
    
    async def _on_failure(self) -> None:
        """Handle failed call."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.state == "closed":
            if self.failure_count >= self.failure_threshold:
                self.state = "open"
                logger.error(
                    f"{self.name} opened after {self.failure_count} failures"
                )
        elif self.state == "half_open":
            self.state = "open"
            logger.warning(f"{self.name} re-opened during recovery attempt")
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time passed to attempt recovery."""
        if self.last_failure_time is None:
            return True
        return time.time() - self.last_failure_time >= self.timeout_seconds
    
    def _time_until_retry(self) -> int:
        """Seconds until next retry attempt."""
        if self.last_failure_time is None:
            return 0
        elapsed = time.time() - self.last_failure_time
        remaining = int(self.timeout_seconds - elapsed)
        return max(0, remaining)
    
    def reset(self) -> None:
        """Manually reset circuit breaker."""
        self.state = "closed"
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        logger.info(f"{self.name} manually reset")
    
    def get_state(self) -> dict[str, Any]:
        """Get circuit breaker state."""
        return {
            "name": self.name,
            "state": self.state,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "last_failure_time": self.last_failure_time,
            "time_until_retry": self._time_until_retry() if self.state == "open" else 0,
        }

"""Session cleanup and management."""

import asyncio
import logging
import time
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class SessionCleanupManager:
    """Manages session lifecycle and cleanup."""
    
    def __init__(self, ttl_seconds: int = 86400):
        """Initialize session cleanup manager (24h TTL default)."""
        self.ttl_seconds = ttl_seconds
        self.sessions: Dict[str, float] = {}
        self._cleanup_task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()
    
    async def register_session(self, session_id: str) -> None:
        """Register a new session."""
        async with self._lock:
            self.sessions[session_id] = time.time()
            logger.debug(f"Session registered: {session_id}")
    
    async def touch_session(self, session_id: str) -> None:
        """Update session last-accessed time."""
        async with self._lock:
            if session_id in self.sessions:
                self.sessions[session_id] = time.time()
    
    async def cleanup_expired(self) -> int:
        """Remove sessions older than TTL. Returns count cleaned up."""
        now = time.time()
        expired = []
        
        async with self._lock:
            for session_id, created_at in list(self.sessions.items()):
                if now - created_at > self.ttl_seconds:
                    expired.append(session_id)
                    del self.sessions[session_id]
        
        if expired:
            logger.info(f"Cleaned up {len(expired)} expired sessions")
        
        return len(expired)
    
    async def start_background_cleanup(self, interval_seconds: int = 3600) -> None:
        """Start background cleanup task (runs every hour)."""
        if self._cleanup_task is not None:
            logger.warning("Background cleanup already running")
            return
        
        async def _cleanup_loop():
            try:
                while True:
                    await asyncio.sleep(interval_seconds)
                    await self.cleanup_expired()
            except asyncio.CancelledError:
                logger.info("Background cleanup stopped")
                raise
        
        self._cleanup_task = asyncio.create_task(_cleanup_loop())
        logger.info(f"Started background cleanup (interval: {interval_seconds}s)")
    
    async def stop_background_cleanup(self) -> None:
        """Stop background cleanup task."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            self._cleanup_task = None
            logger.info("Background cleanup stopped")
    
    async def get_active_sessions(self) -> list[str]:
        """Get list of active sessions."""
        async with self._lock:
            now = time.time()
            return [
                sid
                for sid, created_at in self.sessions.items()
                if now - created_at <= self.ttl_seconds
            ]

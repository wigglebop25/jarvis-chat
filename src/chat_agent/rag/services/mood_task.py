"""Background mood correlation analysis task."""

import asyncio
import logging

logger = logging.getLogger(__name__)


class MoodAnalysisBackgroundTask:
    """Background task for periodic mood correlation analysis."""
    
    def __init__(self, interval_seconds: int = 300):
        """
        Initialize background task.
        
        Args:
            interval_seconds: Run interval (default: 5 mins)
        """
        self.interval = interval_seconds
        self.task = None
        self.is_running = False
    
    async def start(self):
        """Start the background task."""
        if self.is_running:
            logger.warning("Mood analysis task already running")
            return
        
        self.is_running = True
        self.task = asyncio.create_task(self._run_loop())
        logger.info(f"Mood analysis started (interval: {self.interval}s)")
    
    async def stop(self):
        """Stop the background task."""
        self.is_running = False
        if self.task:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
        logger.info("Mood analysis stopped")
    
    async def _run_loop(self):
        """Run the analysis loop."""
        while self.is_running:
            try:
                await asyncio.sleep(self.interval)
                
                if not self.is_running:
                    break
                
                from ..mood_integration import periodically_update_mood_correlations
                
                logger.debug("Running mood correlation analysis...")
                await periodically_update_mood_correlations()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Mood analysis error: {e}")


class MoodAnalysisTask:
    """Wrapper for mood analysis task with lifecycle management."""
    
    def __init__(self, interval_seconds: int = 300):
        """Initialize wrapper."""
        self._task = MoodAnalysisBackgroundTask(interval_seconds)
    
    async def __aenter__(self):
        """Context manager entry."""
        await self._task.start()
        return self._task
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        await self._task.stop()
    
    async def start(self):
        """Start the task."""
        return await self._task.start()
    
    async def stop(self):
        """Stop the task."""
        return await self._task.stop()
    
    @property
    def is_running(self) -> bool:
        """Check if task is running."""
        return self._task.is_running

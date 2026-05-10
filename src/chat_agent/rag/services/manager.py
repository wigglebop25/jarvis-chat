"""RAG service lifecycle management."""

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


class RAGServiceManager:
    """Manage RAG background services lifecycle."""
    
    def __init__(self):
        """Initialize service manager."""
        self.services = {}
        self.is_initialized = False
    
    async def initialize(
        self,
        enable_mood_analysis: bool = True,
        enable_cache_warming: bool = True,
        mood_analysis_interval: int = 300,
    ) -> dict[str, Any]:
        """
        Initialize all RAG background services.
        
        Args:
            enable_mood_analysis: Enable periodic mood analysis
            enable_cache_warming: Warm cache on startup
            mood_analysis_interval: Seconds between analysis runs
            
        Returns:
            Dict with active services
        """
        try:
            if enable_cache_warming:
                logger.info("Warming RAG cache on startup...")
                from .cache_warmer import CacheWarmer
                await CacheWarmer.warm_cache()
                await CacheWarmer.warm_common_queries()
            
            if enable_mood_analysis:
                from .mood_task import MoodAnalysisBackgroundTask
                
                mood_task = MoodAnalysisBackgroundTask(
                    interval_seconds=mood_analysis_interval
                )
                await mood_task.start()
                self.services['mood_analysis'] = mood_task
            
            self.is_initialized = True
            logger.info(f"RAG services initialized: {list(self.services.keys())}")
            
        except Exception as e:
            logger.error(f"Failed to initialize RAG services: {e}")
        
        return self.services
    
    async def shutdown(self):
        """Shutdown all services."""
        try:
            for service_name, service in self.services.items():
                if hasattr(service, 'stop'):
                    logger.info(f"Stopping RAG service: {service_name}")
                    await service.stop()
            
            self.services = {}
            self.is_initialized = False
            logger.info("RAG services shutdown complete")
            
        except Exception as e:
            logger.error(f"Failed to shutdown RAG services: {e}")
    
    def get_service(self, name: str) -> Any:
        """Get a service by name."""
        return self.services.get(name)
    
    async def __aenter__(self):
        """Context manager entry."""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        await self.shutdown()


# Global instance
_rag_manager: Optional[RAGServiceManager] = None


async def get_rag_manager() -> RAGServiceManager:
    """Get or create global RAG service manager."""
    global _rag_manager
    if _rag_manager is None:
        _rag_manager = RAGServiceManager()
    return _rag_manager


async def initialize_rag_services(**kwargs) -> dict[str, Any]:
    """Initialize RAG background services."""
    manager = await get_rag_manager()
    return await manager.initialize(**kwargs)


async def shutdown_rag_services():
    """Shutdown RAG background services."""
    manager = await get_rag_manager()
    await manager.shutdown()

import pytest  # type: ignore
from unittest.mock import MagicMock, AsyncMock
from chat_agent.llm.base import LLMProvider
from chat_agent.mcp.multi_endpoint_router import MultiEndpointMCPRouter
from chat_agent.config import AgentConfig

@pytest.fixture
def mock_llm_provider():
    provider = MagicMock(spec=LLMProvider)
    provider.complete = AsyncMock()
    provider.supports_tools = True
    return provider

@pytest.fixture
def mock_mcp_router():
    router = MagicMock(spec=MultiEndpointMCPRouter)
    router.execute_tool = AsyncMock()
    router.route_and_call = AsyncMock()
    router.list_tools = AsyncMock(return_value=[])
    return router

@pytest.fixture
def agent_config():
    return AgentConfig()

# Lightweight test fixtures to allow running RAG tests offline.
@pytest.fixture
def vector_store():
    class DummyVectorStore:
        def semantic_search(self, query, namespace, top_k=3):
            # Return a high-similarity result for positive tests, low otherwise
            if "sad" in query or "test" in query:
                return [{"similarity": 0.9, "id": "1", "metadata": {}}]
            return [{"similarity": 0.1, "id": "0", "metadata": {}}]
    return DummyVectorStore()

@pytest.fixture
def embeddings():
    return {"dummy": [0.1, 0.2, 0.3]}

@pytest.fixture
def mood_analyzer():
    class DummyMoodAnalyzer:
        def extract_mood_keywords(self, text: str):
            lower = text.lower()
            if "sad" in lower:
                return ["sad"]
            if "workout" in lower:
                return ["workout"]
            return ["chill"]

        def analyze_correlations(self, min_samples=1):
            return {"sad": [{"confidence": 0.8}]}
    return DummyMoodAnalyzer()

import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock, patch
from chat_agent.agent_builder import ChatAgentBuilder
from chat_agent.config import AgentConfig

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("E2E-Test")

async def test_agent_integration():
    logger.info("Starting E2E Integration Test...")
    
    # Mock embeddings to avoid sentence-transformers dependency
    with patch("chat_agent.rag.embedding_model.embed_text", return_value=[0.1] * 384):
        # 1. Setup Config
        config = AgentConfig()
        config.agent_type = "langgraph"
        config.mcp.multi_endpoint_enabled = True
        
        # 2. Mock dependencies
        mock_llm = MagicMock()
        # Simulate a response that calls a tool
        mock_llm.complete = AsyncMock()
        mock_llm.supports_tools = True
        
        # Mocking a realistic ReAct sequence:
        # 1st call: Returns a Tool Call
        # 2nd call: Returns a final text response
        from chat_agent.llm.base import LLMResponse, ToolCall
        mock_llm.complete.side_effect = [
            LLMResponse(
                text="Sure, I'll search for that jazz track.",
                tool_calls=[ToolCall(id="call_1", name="searchSpotify", arguments={"query": "jazz", "type": "track"})],
                usage={"prompt_tokens": 10, "completion_tokens": 10}
            ),
            LLMResponse(
                text="I've found 'Kind of Blue' and started playing it for you.",
                tool_calls=[],
                usage={"prompt_tokens": 20, "completion_tokens": 5}
            )
        ]
        
        mock_mcp = AsyncMock()
        # Mock tools/list for dynamic discovery
        mock_mcp.call.side_effect = lambda method, params, timeout=None: {
            "tools/list": {"tools": [{"name": "searchSpotify", "description": "Search Spotify", "inputSchema": {}}]},
            "jarvis/route_and_call": {"intent": "music_control", "confidence": 0.9, "should_execute": False},
            "tools/call": {"result": {"uri": "spotify:track:123"}}
        }.get(method, {})

        # 3. Build Agent
        builder = ChatAgentBuilder(config)
        agent = builder.build_langgraph_agent()
        agent.llm_provider = mock_llm
        assert agent.mcp_router is not None, "mcp_router should be initialized"
        agent.mcp_router.mcp_client = mock_mcp
        
        # 4. Process a transcript
        transcript = "play some jazz"
        logger.info(f"User Input: {transcript}")
        
        response = await agent.process_transcript(transcript)
        
        logger.info(f"Agent Response: {response}")
        
        # 5. Assertions
        history = agent.get_conversation_history()
        logger.info(f"History length: {len(history)}")
        
        for msg in history:
            logger.info(f"[{msg.role}] {msg.content[:100]}...")

        logger.info("Test completed successfully!")

if __name__ == "__main__":
    asyncio.run(test_agent_integration())

import pytest  # type: ignore
from chat_agent.models import IntentType

def test_intent_type_enum():
    assert IntentType.MUSIC_CONTROL.value == "music_control"
    assert IntentType.UNKNOWN.value == "unknown"

@pytest.mark.anyio
async def test_mock_fixtures(mock_llm_provider, mock_mcp_router):
    assert mock_llm_provider.supports_tools is True
    await mock_mcp_router.list_tools()
    mock_mcp_router.list_tools.assert_called_once()

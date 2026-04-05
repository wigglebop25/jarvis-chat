# JARVIS Chat Agent

Modular LLM-powered chat agent with swappable providers.

## Features

- **Swappable LLM Providers**: Ollama (free, local), OpenAI, Google Gemini, GitHub Copilot SDK
- **Tool Integration**: Execute system tools via MCP Server
- **Multi-turn Conversations**: Maintains conversation context
- **Streaming Support**: Real-time response streaming

## Installation

```bash
cd jarvis-chat
uv sync
```

## Quick Start

```python
from chat_agent import ChatAgent, AgentConfig, LLMConfig

# Create with Ollama (free, local)
config = AgentConfig(
    llm=LLMConfig(provider="ollama", model="llama3")
)
agent = ChatAgent(config=config, llm_config=config.llm)

# Process a transcript
response = agent.process_transcript("What is the system CPU usage?")
print(response)
```

## Configuration

Set environment variables for paid providers:

```bash
export OPENAI_API_KEY=sk-...
export GEMINI_API_KEY=...
```

## Providers

| Provider | Type | Cost | Setup |
|----------|------|------|-------|
| Ollama | Local | Free | `ollama pull llama3` |
| OpenAI | API | Paid | OPENAI_API_KEY |
| Gemini | API | Free tier | GEMINI_API_KEY |
| Copilot SDK | GitHub | Free with Copilot | GitHub CLI auth |

## License

MIT

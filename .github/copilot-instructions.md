# JARVIS Chat Agent - Copilot Instructions

## Quick Reference

### Build & Dependencies
- **Install dependencies**: `uv sync`
- **Package manager**: UV (fast Python package manager)
- **Python version**: 3.13+

### Run Commands
- **Interactive CLI**: `uv run main.py`
- **Single command**: `uv run main.py "your command"`
- **HTTP server mode**: `uv run main.py --server` (default port 8000)
- **Custom MCP URL**: `uv run main.py --mcp-url http://custom-host:port`
- **Benchmarking**: `python benchmark_google_api.py --model gemini-1.5-flash --warmup --repeats 1`

### Linting & Type Checking
- **Pylint**: Uses 120 char line length, disables docstring requirements. Config: `.pylintrc`
- **Pyright**: Set to basic type checking mode. Config: `pyrightconfig.json`
- Run linting: `pylint src/` (not yet configured as script)
- Run type check: `pyright` (not yet configured as script)

## Architecture

### High-Level Flow
The JARVIS Chat Agent optimizes command latency through a **Rust-first deterministic hot-path** and **LLM response caching**:

```
Voice Transcript (input)
    ↓
[MCP Router] → [Rust MCP Router] (Fast-Path Match?) → Yes → [Execution]
    ↓ No
[LLM Response Cache] (Cache Hit?) → Yes → [Response]
    ↓ No
[LLM Provider] (gemma-4-31b-it) → [Tool Selection]
    ↓
[MCP Router] → [Rust MCP Server]
```

### Key Components

#### 1. **Rust Router Hot-Path** (`jarvis-skills/rust-mcp-server`)
- **Deterministic Recognition**: Ported from Python to Rust for ultra-low latency.
- **RPC Methods**: `jarvis/route_and_call` handles intent detection, parameter extraction, and execution in one round-trip.
- **Linux Compatibility**: Hardened for `/mnt` and `/media` paths and Linux-native utilities (`nmcli`, `pactl`).

#### 2. **LLM Response Cache** (`src/chat_agent/response_cache.py`)
- **Phase 2 Implementation**: Caches general queries to bypass LLM latency.
- **Safety Guardrails**: Automatically skips caching for stateful commands (volume, wifi), file paths, or freshness-sensitive queries ("now", "status").
- **Persistence**: Managed via `.cache/llm-response-cache.json`.

#### 3. **LLM Provider Abstraction** (`src/chat_agent/llm/`)
- **Pluggable providers**: Ollama (local, free), OpenAI (paid), Google Gemini (free tier), GitHub Copilot SDK
- **Selection**: Via `LLMConfig(provider="ollama"|"openai"|"gemini"|"copilot")`
- **Base class**: `LLMProvider` with sync/async completion methods
- **Tool support**: Some providers support function calling (OpenAI, Gemini), others don't (Ollama default)
- **Configuration**: Each provider reads from env vars (e.g., `OPENAI_API_KEY`, `GEMINI_API_KEY`)

#### 2. **Intent Recognition** (`src/chat_agent/intent.py`)
- **Pattern-based matching**: Regex patterns defined per `IntentType` enum (SYSTEM_INFO, VOLUME_CONTROL, MUSIC_CONTROL, DIRECTORY_LIST, etc.)
- **Intent → Tool mapping**: `get_tool_name_for_intent()` maps recognized intent to MCP tool name
- **Parameter extraction**: `map_intent_params_to_tool()` extracts arguments from matched text

#### 3. **Tool Execution via MCP** (`src/chat_agent/mcp/`)
- **MCPRouter**: Routes all tool calls to remote MCP server (default: `http://127.0.0.1:5050`)
- **Protocol**: JSON-RPC calls with `tools/call` method
- **Execution model**: Strict MCP-only execution (no local tool fallback)
- **MCP Server**: Hosted in separate `jarvis-skills/rust-mcp-server` repository (Tokio + Axum + Hyper stack)

#### 4. **Context & Caching** (`src/chat_agent/context_cache.py`)
- **Session cache**: Persists conversation history with KV-style storage
- **Token budgeting**: Token estimation for context assembly (4 chars ≈ 1 token)
- **Summary compaction**: Optional context summary refresh when cache grows
- **DType conversion pipeline**: Numeric context can be converted to fp16/fp8 for efficiency
- **Configuration**: `CONTEXT_CACHE_ENABLED=true`, `CONTEXT_DTYPE=fp16|fp8|fp32`, `CONTEXT_CACHE_PATH=.cache/context-cache.json`

#### 5. **Agent Lifecycle** (`src/chat_agent/agent.py`)
- **Backward-compatible wrapper** (`_CompatChatAgent`): Maintains synchronous API for legacy code
- **Modern async API**: `ChatAgent.process_transcript()` (async) for new code
- **Context persistence**: `ConversationContext` tracks message history with role/content
- **Tool call generation**: Structured tool calls from intent or LLM function calling

## Key Conventions

### Configuration Management
- **Environment-driven**: Use `.env` file for development, env vars for deployment
- **Pydantic models**: All config is type-safe via `pydantic.BaseModel`
- **Defaults**: Ollama provider and no context cache unless explicitly enabled
- **Validation**: Provider names, temperature ranges, token limits validated at config creation

### Error Handling
- **LLMProviderError** hierarchy: Base exception with `LLMConfigurationError` for missing API keys
- **MCP failures**: Return `MCPToolResult` with `is_error=True` rather than raising
- **MCP requirement**: Tool execution requires MCP connectivity to the Rust server

### Async/Sync Duality
- **Main entry (main.py)**: Uses persistent event loop (`get_event_loop()`) to avoid loop conflicts
- **Agent API**: Core async methods, wrapped in `_CompatChatAgent` for sync compatibility
- **Tool routing**: MCP calls are async over HTTP JSON-RPC

### Message Format
- **Chat completions**: OpenAI-style format `[{"role": "user"/"assistant", "content": "..."}]`
- **Tool calls**: JSON objects with `name`, `arguments` (dict), `id` (for tracking)
- **Tool definitions**: JSON Schema-compatible (used by OpenAI, Gemini; Ollama ignores)

### Code Organization
- **Module structure**: Each provider/component gets its own module with clear imports
- **Provider imports**: Lazy loaded via `llm/registry.py` to avoid hard dependencies
- **Type hints**: Full type coverage expected (Pyright basic mode enforced)

## Common Tasks

### Adding a New LLM Provider
1. Create `src/chat_agent/llm/my_provider.py` inheriting from `LLMProvider`
2. Implement: `name`, `supports_tools`, `is_configured()`, `complete_sync()`, `complete_async()`
3. Register in `src/chat_agent/llm/registry.py` in `create_provider()`
4. Add env var config to `src/chat_agent/config.py` if needed
5. Update provider list in README.md

### Adding a New Intent Type
1. Add to `IntentType` enum in `src/chat_agent/models.py`
2. Add regex patterns to `INTENT_PATTERNS` in `src/chat_agent/intent.py`
3. Map to tool in `INTENT_TO_TOOL` in `intent.py`
4. Define tool schema in `src/chat_agent/tools/definitions.py`

### Adding a New Tool
1. Define schema in `src/chat_agent/tools/definitions.py` in `get_tool_definitions()`
2. Implement handler in remote MCP server (jarvis-skills/rust-mcp-server)
3. Optional: Add intent pattern if voice-driven

### Testing Provider-Specific Code
- No formal test suite yet; use interactive mode: `uv run main.py`
- Set provider via env: `LLM_PROVIDER=openai LLM_MODEL=gpt-4o uv run main.py`
- Test streaming: Use async completion methods directly in Python REPL

## Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `LLM_PROVIDER` | `ollama` | Provider selection (ollama, openai, gemini, copilot) |
| `LLM_MODEL` | `llama3` | Model name (provider-specific) |
| `OPENAI_API_KEY` | - | OpenAI auth (required if provider=openai) |
| `GEMINI_API_KEY` | - | Google Gemini auth (required if provider=gemini) |
| `MCP_SERVER_URL` | `http://127.0.0.1:5050` | MCP server endpoint |
| `CHAT_SESSION_ID` | `default` | Session identifier for context caching |
| `CONTEXT_CACHE_ENABLED` | `false` | Enable persistent context cache |
| `CONTEXT_DTYPE` | `fp32` | Numeric context precision (fp32, fp16, fp8) |
| `CONTEXT_CACHE_PATH` | `.cache/context-cache.json` | Cache storage location |

## Dependencies

- **google-generativeai**: Gemini API client
- **openai**: OpenAI API client
- **httpx**: Async HTTP client (MCP communication)
- **pydantic**: Config validation and serialization
- **python-dotenv**: Environment variable loading
- **asyncio**: Async runtime (stdlib)

No heavy ML frameworks; this is a **chat coordination layer** that delegates to external LLMs and tools.

## Related Repositories

- **jarvis-skills/rust-mcp-server**: Implements MCP server with available tools (GitHub: Tokio + Axum + Hyper)
- Connect via `MCP_SERVER_URL` environment variable

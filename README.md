# JARVIS Chat Agent

A Python-based chat runtime for JARVIS. It handles natural language processing, tool orchestration (ReAct), and routes all hardware/system commands to the Rust MCP server.

## Overview

- **Runtime:** Python 3.12+ (managed via `uv`)
- **Backend:** Communicates with `jarvis-skills/rust-mcp-server` via JSON-RPC.
- **Tools:** Dynamic loading from MCP `tools/list` with smart local caching.

## Setup

1. **Install dependencies:**
   ```bash
   cd jarvis-chat
   uv sync
   ```

2. **Configure environment:**
   Create a `.env` file from the example and add your keys:
   ```bash
   cp .env.example .env
   ```

## Running the Agent

### 1. Start the Rust MCP Server
(From the `jarvis-skills` directory)
```bash
cd rust-mcp-server
cargo run --release
```

### 2. Launch the Chat Agent
```bash
cd jarvis-chat
python main.py
```

- **Interactive mode:** Just run `python main.py` and start typing.
- **Server mode:** Run `python main.py --server --port 8000` to expose the HTTP API.

## Verification

Check if everything is connected correctly:

1. **Health Check:** `curl http://127.0.0.1:5050/health`
2. **Tool Discovery:** `curl http://127.0.0.1:5050/tools`
3. **Smoke Test:** `python main.py "get system info"`

## Notes

- **Caching:** LLM responses are cached for 3 minutes to save on API costs and latency.
- **Routing:** Complex hardware intents (Spotify, Volume, WiFi) are routed directly to Rust for native execution.
- **Requirements:** Ensure the Rust MCP server is running before starting the Python agent.

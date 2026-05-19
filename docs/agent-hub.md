# Agent Hub: Sub-Agent Orchestrator

Agent Hub is the FastAPI service that sits in front of all local LLMs on the cluster. Its job is to route every incoming query to the smallest model capable of answering it, apply token-saving transforms before forwarding, and maintain a unified API surface regardless of which backends happen to be online.

Source: `/home/merry/agent-hub/main.py`
Service port: **8000** (via uvicorn, configured at the bottom of main.py as port 7860 in `__main__` — override with the systemd unit's `--port 8000` if applicable)

---

## 1. Overview

Agent Hub sits above three backends:

| Backend | Port | Models served |
|---|---|---|
| Ollama | 11434 | phi3.5, llava-phi3, qwen2.5-coder:7b |
| llama.cpp (llama-server) | 8080 | mistral-small (Mistral 22B distributed across aipi+jolly) |
| ChromaDB | 8001 | Vector store (no model, just retrieval) |

All backend communication uses the OpenAI-compatible `/v1` API format. Ollama exposes this natively; llama-server implements it via its built-in HTTP server. This means the same `AsyncOpenAI` client class is used for both, configured with different `base_url` values.

The central design principle: **phi3.5 handles all cheap tasks** (routing, summarisation, RAG synthesis, general chat when Mistral is down). **Mistral 22B is invoked only for general/voice queries and synthesis of parallel sub-answers** — the two tasks that actually benefit from the larger context window and stronger reasoning. Code generation always uses qwen2.5-coder:7b regardless of Mistral's availability.

---

## 2. Sub-Agent Roster

The six agents are defined as `async` functions in main.py. Each has a dedicated system prompt constant and calls one specific model.

### Router — `agent_router(query: str)`
- **Model**: phi3.5 via Ollama
- **Max tokens generated**: 64
- **Temperature**: 0.0 (deterministic)
- **Purpose**: Classifies the incoming query into one of five categories (`general`, `code`, `vision`, `voice`, `rag`) and sets a `multi_question` boolean. Returns JSON.
- **Output format**: `{"category": "...", "multi_question": true|false, "confidence": 0.0–1.0}`
- **Failure mode**: If phi3.5 returns non-JSON, defaults to `{"category": "general", "multi_question": false, "confidence": 0.5}` and logs a warning.

### Summariser — `agent_summariser(text: str)`
- **Model**: phi3.5 via Ollama
- **Max tokens generated**: 700
- **Temperature**: 0.2
- **Threshold**: 2000 tokens (approximated as `word_count × 1.3`)
- **Purpose**: Compresses long inputs before passing them to heavyweight models. Called automatically by `agent_chat` and `agent_coder` before dispatching to their respective models. If input is under 2000 tokens, returns the input unchanged without making an LLM call.
- **Target output**: Dense summary under 500 words preserving all key facts, numbers, names, and intent.

### Chat — `agent_chat(query: str)`
- **Model**: Mistral 22B via llama-server (falls back to phi3.5 if Mistral is down)
- **Max tokens generated**: 1024
- **Temperature**: 0.3
- **Purpose**: General-purpose reasoning and conversation. Invoked for `general` and `voice` category queries. Calls `agent_summariser` first if input exceeds the 2000-token threshold.

### Coder — `agent_coder(query: str)`
- **Model**: qwen2.5-coder:7b via Ollama
- **Max tokens generated**: 2048
- **Temperature**: 0.2
- **Purpose**: Code generation, debugging, review, and technical how-to. Invoked for `code` category queries and directly via the `/code` endpoint. Calls `agent_summariser` first if input is long. No Mistral fallback — if Ollama is down, the endpoint raises an error.

### RAG — `agent_rag(query: str)`
- **Model**: phi3.5 via Ollama
- **Max tokens generated**: 1024
- **Temperature**: 0.2
- **Purpose**: Retrieves up to 5 document excerpts from ChromaDB, then synthesises an answer using only the retrieved content. If ChromaDB is unavailable or returns no results, falls back to phi3.5 answering from parametric knowledge (with a `[Note: knowledge base is unavailable]` prefix injected into the prompt).
- **ChromaDB collection**: `agent_hub_knowledge`

### Summariser — `agent_synthesiser(original_query, partial_answers)`
- **Model**: Mistral 22B via llama-server (falls back to phi3.5)
- **Max tokens generated**: 2048
- **Purpose**: Merges the list of parallel sub-answers into one coherent response. Only invoked when the router's `multi_question` flag is true and the query successfully splits into 2+ parts.

**Note on vision and voice**: The router can classify queries into `vision` and `voice` categories, and both model names (`llava-phi3`, `MODEL_PHI35`) are defined as constants in main.py. However, the current `_route_single()` dispatch function maps both `vision` and `voice` (and any unrecognised category) to `agent_chat()`. Dedicated `agent_vision()` and `agent_voice()` functions are not yet implemented — they are stub categories in the router.

---

## 3. Token Maximisation Strategy

The pipeline is designed to spend the minimum tokens necessary to answer each query. The execution path:

```
Incoming query
    │
    ├─ SHA-256 cache check  →  cache hit? return immediately (0 LLM tokens)
    │
    ├─ agent_router (phi3.5, max_tokens=64)  ≈200 input + 64 output tokens
    │
    ├─ approx_tokens(query) > 2000?
    │       yes → agent_summariser (phi3.5, max_tokens=700) before forwarding
    │
    ├─ multi_question == true?
    │       yes → _split_questions (phi3.5, max_tokens=256)
    │             → asyncio.gather(*[_route_single(q) for q in sub_queries])  ← parallel
    │             → agent_synthesiser (Mistral 22B, max_tokens=2048)
    │
    └─ single question dispatch:
            category == "code"    → agent_coder (qwen2.5-coder:7b, max_tokens=2048)
            category == "rag"     → agent_rag (phi3.5, max_tokens=1024)
            everything else       → agent_chat (Mistral 22B, max_tokens=1024)
```

**Key threshold values from the code**:

| Constant | Value | Purpose |
|---|---|---|
| `SUMMARISE_THRESHOLD` | 2000 tokens | Trigger for summariser (approx, word × 1.3) |
| `CTX_PHI35` | 4096 | phi3.5 context window (reported in /status) |
| `CTX_MISTRAL` | 8192 | Mistral context window (reported in /status) |
| Router `max_tokens` | 64 | Hard cap on router output — it only needs to emit JSON |
| Summariser `max_tokens` | 700 | Summary capped at ~500 words with headroom |

**Escalation to Mistral** only happens for two reasons:
1. The router classifies the query as `general` or `voice`
2. The query has `multi_question: true` (synthesiser always uses Mistral)

Everything else — RAG synthesis, code generation, routing, summarisation — stays on phi3.5 or qwen2.5-coder.

---

## 4. API Endpoints

### POST /chat

Auto-routed endpoint. Accepts any query and dispatches through the full orchestration pipeline.

**Request:**
```json
{
  "query": "Explain the Fourier transform simply.",
  "system_prompt": null
}
```

`system_prompt` is optional. If provided, it is prepended to the query as `[SYSTEM]\n<prompt>\n\n[USER]\n<query>` before routing.

**Response:**
```json
{
  "response": "The Fourier transform decomposes a signal into its constituent frequencies...",
  "category": "general",
  "model_used": "mistral-small",
  "approx_input_tokens": 8,
  "cached": false,
  "elapsed_ms": 18342.5
}
```

```bash
curl -s -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "Explain the Fourier transform simply."}' | python3 -m json.tool
```

### POST /code

Bypasses the router and sends directly to qwen2.5-coder:7b. Use this when you already know the query is a code task and want to skip the router overhead.

**Request:**
```json
{
  "query": "Write a Python async HTTP client with retry logic."
}
```

**Response:**
```json
{
  "response": "```python\nimport asyncio\nimport httpx\n...",
  "approx_input_tokens": 11,
  "elapsed_ms": 24610.0
}
```

```bash
curl -s -X POST http://localhost:8000/code \
  -H "Content-Type: application/json" \
  -d '{"query": "Write a Python function to parse ISO 8601 timestamps."}' | python3 -m json.tool
```

The cache key for `/code` is prefixed with `"code:"` to prevent collisions with identical queries that came through `/chat` and were routed differently.

### POST /rag/query

Bypasses the router and sends directly to the RAG agent. Allows specifying `n_results` (default: 5).

**Request:**
```json
{
  "query": "What does my project doc say about authentication?",
  "n_results": 5
}
```

**Response:**
```json
{
  "response": "According to [auth-design.md]: authentication uses JWT tokens with a 24-hour expiry...",
  "excerpts_found": 5,
  "elapsed_ms": 3820.1
}
```

```bash
curl -s -X POST http://localhost:8000/rag/query \
  -H "Content-Type: application/json" \
  -d '{"query": "How does the ingest pipeline handle disambiguation errors?", "n_results": 3}' | python3 -m json.tool
```

`excerpts_found` reports how many ChromaDB results were retrieved. If ChromaDB was unavailable, this will be 0 and the response will be from parametric memory.

### GET /status

Runs a live health check against all three backends and returns their current state. Also reports cache size and token budget constants.

```bash
curl -s http://localhost:8000/status | python3 -m json.tool
```

**Response:**
```json
{
  "backends": {
    "ollama": {
      "online": true,
      "url": "http://localhost:11434/v1",
      "models": ["phi3.5", "llava-phi3", "qwen2.5-coder:7b"]
    },
    "mistral": {
      "online": true,
      "url": "http://localhost:8080/v1",
      "models": ["mistral-small"],
      "note": "Falls back to phi3.5 when offline"
    },
    "chromadb": {
      "online": true,
      "url": "http://localhost:8001",
      "collection": "agent_hub_knowledge"
    }
  },
  "cache": {
    "entries": 14
  },
  "token_budget": {
    "phi35_ctx": 4096,
    "mistral_ctx": 8192,
    "summarise_threshold": 2000
  }
}
```

### GET /agents

Returns the current sub-agent roster with live model assignments (accounts for Mistral fallback state).

```bash
curl -s http://localhost:8000/agents | python3 -m json.tool
```

When Mistral is offline, the `chat` and `synthesiser` entries will show `"model": "phi3.5 (fallback)"` and `"backend": "ollama (fallback)"`.

---

## 5. Backend Health Checking

Agent Hub runs a periodic health check every **30 seconds** via `_periodic_health_check()`, which is spawned as an `asyncio` background task at startup via `@app.on_event("startup")`.

The three ping functions:

| Function | Endpoint probed | Timeout |
|---|---|---|
| `_ping_ollama()` | `GET http://localhost:11434/api/tags` | 3.0 seconds |
| `_ping_mistral()` | `GET http://localhost:8080/health` (falls back to `/v1/models`) | 3.0 seconds each |
| `_ping_chromadb()` | `GET http://localhost:8001/api/v2/heartbeat` | 3.0 seconds |

All three pings run concurrently via `asyncio.gather()`. Results are written into the `_backend_status` dict which all agent functions read before deciding which model to call.

**Automatic fallback**: The `_mistral_complete()` function checks `_backend_status["mistral"]` before every call. If false, it immediately calls `_ollama_complete(MODEL_PHI35, ...)` instead — no error is raised, the caller receives a response from phi3.5 transparently. A warning is logged: `Mistral backend down — falling back to phi3.5`.

Fallback latency: because the health check runs every 30 seconds, there is a window of up to 30 seconds between Mistral going down and Agent Hub detecting it. Queries arriving in that window may time out waiting for Mistral before the next health cycle marks it offline.

**GET /status** bypasses the 30-second cycle: it calls `refresh_backend_status()` directly and returns a live reading.

---

## 6. ChromaDB Integration

ChromaDB is accessed via `chromadb.AsyncHttpClient` on port 8001. The collection used by Agent Hub is `agent_hub_knowledge` (defined as `CHROMA_COLLECTION_NAME`).

**Note on collection name mismatch**: The ingest script (`/home/merry/rag-ingest/ingest.py`) writes to a collection named `knowledge_base`. Agent Hub reads from `agent_hub_knowledge`. These are two separate collections. If you want Agent Hub's RAG agent to see the ingested data, you must either (a) change `CHROMA_COLLECTION_NAME` in main.py to `"knowledge_base"`, or (b) change `COLLECTION_NAME` in ingest.py to `"agent_hub_knowledge"`. They currently do not share data unless one is updated.

**RAG query flow** inside `_chroma_query(query_text, n_results=5)`:

1. Obtain the collection handle via `_get_chroma_collection()` (cached after first successful connection)
2. Call `collection.query(query_texts=[query_text], n_results=n_results, include=["documents", "metadatas"])`
3. ChromaDB performs embedding internally using the collection's configured embedding function (sentence-transformers all-MiniLM-L6-v2 on the server side), then HNSW approximate nearest-neighbour search with cosine distance
4. Results are formatted as `[source_name]\n<chunk_text>` and joined with `\n\n---\n\n` to form the context block
5. The context block is prepended to the user question as `RETRIEVED EXCERPTS:\n\n<context>\n\nUSER QUESTION:\n<query>` before passing to phi3.5

The collection is created with `{"hnsw:space": "cosine"}` metadata, consistent with the ingest script's configuration.

---

## 7. Adding New Agents

To add a new specialist agent — for example, a summarisation-focused agent backed by a different model:

**Step 1**: Define a system prompt constant near the top of the file with the other `*_SYSTEM` constants:

```python
NEWSUMMARISER_SYSTEM = """You are a specialist document summariser.
Focus on extracting the three most important points.
Use bullet format."""
```

**Step 2**: Write an async agent function following the existing pattern:

```python
async def agent_newsummariser(query: str) -> str:
    """Dedicated summarisation agent using phi3.5."""
    messages = [
        {"role": "system", "content": NEWSUMMARISER_SYSTEM},
        {"role": "user", "content": query},
    ]
    return await _ollama_complete(MODEL_PHI35, messages, max_tokens=512, temperature=0.2)
```

**Step 3**: Add a case to `_route_single()` so the router can dispatch to it:

```python
async def _route_single(query: str, category: str) -> str:
    if category == "code":
        return await agent_coder(query)
    if category == "rag":
        return await agent_rag(query)
    if category == "newsummary":        # new
        return await agent_newsummariser(query)
    return await agent_chat(query)
```

**Step 4**: Add the new category string to the router's system prompt (`ROUTER_SYSTEM`) so phi3.5 knows to classify queries into it:

```python
ROUTER_SYSTEM = """...
- newsummary : requests to summarise a document or article
..."""
```

**Step 5**: Register the new agent in the `GET /agents` response inside `get_agents()`:

```python
{
    "name": "newsummariser",
    "description": "Extracts three key points from a document",
    "model": MODEL_PHI35,
    "backend": "ollama",
    "ctx_window": CTX_PHI35,
},
```

No restart of other services is needed — Agent Hub is stateless between requests.

---

## 8. Example Usage

**General question (routed to Mistral 22B):**
```bash
curl -s -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the difference between TCP and UDP?"}' | python3 -m json.tool
```
Expected: `"category": "general"`, `"model_used": "mistral-small"`, response time ~15–30 seconds on cold model.

**Code question (direct to qwen2.5-coder):**
```bash
curl -s -X POST http://localhost:8000/code \
  -H "Content-Type: application/json" \
  -d '{"query": "Write a bash script that monitors disk usage and emails an alert if any partition exceeds 90%."}' | python3 -m json.tool
```

**RAG lookup:**
```bash
curl -s -X POST http://localhost:8000/rag/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What arXiv papers were ingested about reinforcement learning?", "n_results": 3}' | python3 -m json.tool
```

**Second identical query (cache hit):**
```bash
curl -s -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the difference between TCP and UDP?"}' | python3 -m json.tool
```
Expected: `"cached": true`, `"model_used": "(cached)"`, `elapsed_ms` under 5 ms.

**Check backend health:**
```bash
curl -s http://localhost:8000/status | python3 -m json.tool
```

**List agents with live fallback state:**
```bash
curl -s http://localhost:8000/agents | python3 -m json.tool
```

**Multi-question input (parallel execution):**
```bash
curl -s -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What is HNSW indexing? Also, how does cosine similarity work?"}' | python3 -m json.tool
```
Expected: router sets `multi_question: true`, both sub-questions run in parallel via `asyncio.gather`, synthesiser merges the answers. `"category"` will reflect the primary classification (likely `"general"`).

**With a custom system prompt:**
```bash
curl -s -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "Summarise what you know about attention mechanisms.", "system_prompt": "Respond as if explaining to a high school student."}' | python3 -m json.tool
```
The system prompt is prepended as `[SYSTEM]\n...\n\n[USER]\n...` before routing. This adds token overhead — keep custom system prompts concise.

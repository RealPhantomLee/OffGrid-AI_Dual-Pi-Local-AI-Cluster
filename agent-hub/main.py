"""
Agent Hub — FastAPI sub-agent orchestrator for Raspberry Pi 5
Routes queries to the smallest capable local LLM, maximising tokenisation efficiency.

Model topology
--------------
Ollama  (port 11434): phi3.5, llava-phi3, qwen2.5-coder:7b
llama.cpp (port 8080): Mistral Small 22B  (the big distributed model)
ChromaDB  (port 8001): vector store for RAG

Design rules
------------
- phi3.5 (fast, small) handles routing, RAG synthesis, summarisation, and general chat
- Escalate to Mistral 22B only when the router decides or phi3.5 is unavailable
- Summarise any input > 2 000 tokens before forwarding to a specialist
- Split multi-question inputs and run sub-queries in parallel, then synthesise
- In-memory dict cache (query_hash -> response) — good enough for a Pi
- Mistral down? fall back to phi3.5 for every agent that normally uses it
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import time
from typing import Any

import chromadb
import httpx
from fastapi import FastAPI, HTTPException
from openai import AsyncOpenAI
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
log = logging.getLogger("agent-hub")

# ---------------------------------------------------------------------------
# Backend URLs / model names
# ---------------------------------------------------------------------------
OLLAMA_BASE_URL = "http://localhost:11434/v1"
LLAMA_CPP_BASE_URL = "http://localhost:8080/v1"
CHROMA_HOST = "localhost"
CHROMA_PORT = 8001

MODEL_PHI35 = "phi3.5"
MODEL_LLAVA = "llava-phi3"
MODEL_QWEN_CODER = "qwen2.5-coder:7b"
MODEL_MISTRAL = "mistral-small"  # served by llama.cpp; name may vary

# Context windows
CTX_PHI35 = 4096
CTX_MISTRAL = 8192

# Token approximation threshold
SUMMARISE_THRESHOLD = 2000  # tokens (approx)

# ---------------------------------------------------------------------------
# OpenAI-compatible clients (all local, no cloud calls)
# ---------------------------------------------------------------------------
ollama_client = AsyncOpenAI(
    base_url=OLLAMA_BASE_URL,
    api_key="ollama",
)

mistral_client = AsyncOpenAI(
    base_url=LLAMA_CPP_BASE_URL,
    api_key="none",
)

# ---------------------------------------------------------------------------
# In-memory response cache
# ---------------------------------------------------------------------------
_cache: dict[str, dict[str, Any]] = {}


def _cache_key(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _cache_get(key: str) -> dict[str, Any] | None:
    return _cache.get(key)


def _cache_set(key: str, value: dict[str, Any]) -> None:
    _cache[key] = value


# ---------------------------------------------------------------------------
# Token approximation
# ---------------------------------------------------------------------------
def approx_tokens(text: str) -> int:
    """Word-count × 1.3 — fast, no tiktoken required."""
    return int(len(text.split()) * 1.3)


# ---------------------------------------------------------------------------
# Backend health state (refreshed by health-check task)
# ---------------------------------------------------------------------------
_backend_status: dict[str, bool] = {
    "ollama": False,
    "mistral": False,
    "chromadb": False,
}


async def _ping_ollama() -> bool:
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            r = await client.get("http://localhost:11434/api/tags")
            return r.status_code == 200
    except Exception:
        return False


async def _ping_mistral() -> bool:
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            r = await client.get("http://localhost:8080/health")
            return r.status_code == 200
    except Exception:
        # Try /v1/models as a fallback probe
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                r = await client.get("http://localhost:8080/v1/models")
                return r.status_code == 200
        except Exception:
            return False


async def _ping_chromadb() -> bool:
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            r = await client.get(f"http://{CHROMA_HOST}:{CHROMA_PORT}/api/v2/heartbeat")
            return r.status_code == 200
    except Exception:
        return False


async def refresh_backend_status() -> dict[str, bool]:
    results = await asyncio.gather(
        _ping_ollama(), _ping_mistral(), _ping_chromadb(), return_exceptions=True
    )
    _backend_status["ollama"] = results[0] is True
    _backend_status["mistral"] = results[1] is True
    _backend_status["chromadb"] = results[2] is True
    return dict(_backend_status)


# ---------------------------------------------------------------------------
# Periodic health-check background task
# ---------------------------------------------------------------------------
async def _periodic_health_check() -> None:
    while True:
        await refresh_backend_status()
        await asyncio.sleep(30)


# ---------------------------------------------------------------------------
# Low-level completion helpers
# ---------------------------------------------------------------------------
async def _ollama_complete(
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int = 1024,
    temperature: float = 0.3,
) -> str:
    """Call Ollama-hosted model; returns the assistant text."""
    response = await ollama_client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return response.choices[0].message.content or ""


async def _mistral_complete(
    messages: list[dict[str, str]],
    max_tokens: int = 2048,
    temperature: float = 0.3,
) -> str:
    """Call Mistral 22B via llama.cpp; falls back to phi3.5 if Mistral is down."""
    if not _backend_status["mistral"]:
        log.warning("Mistral backend down — falling back to phi3.5")
        return await _ollama_complete(MODEL_PHI35, messages, max_tokens, temperature)
    response = await mistral_client.chat.completions.create(
        model=MODEL_MISTRAL,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return response.choices[0].message.content or ""


# ---------------------------------------------------------------------------
# AGENT 1 — Router (phi3.5)
# ---------------------------------------------------------------------------
ROUTER_SYSTEM = """You are a query classifier for a local AI assistant running on a Raspberry Pi.

Classify the user message into EXACTLY ONE of these categories:
- general   : casual conversation, factual questions, opinion
- code      : programming, debugging, code review, scripts, technical how-to
- vision    : image analysis, OCR, visual questions (only when an image is attached)
- voice     : text-to-speech, speech-to-text, audio processing
- rag       : questions that need retrieval from a knowledge base ("according to...", "what does my document say", "find in my notes")

ALSO decide:
- multi_question: true if the message contains more than one distinct question that can be answered independently

Respond with ONLY valid JSON, no markdown, no explanation:
{"category": "<category>", "multi_question": <true|false>, "confidence": <0.0-1.0>}"""


async def agent_router(query: str) -> dict[str, Any]:
    """Classify the query. Returns dict with category, multi_question, confidence."""
    messages = [
        {"role": "system", "content": ROUTER_SYSTEM},
        {"role": "user", "content": query},
    ]
    raw = await _ollama_complete(MODEL_PHI35, messages, max_tokens=64, temperature=0.0)
    # Strip any accidental markdown fences
    raw = re.sub(r"```[a-z]*\n?", "", raw).strip()
    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        log.warning("Router returned non-JSON: %r — defaulting to general", raw)
        result = {"category": "general", "multi_question": False, "confidence": 0.5}
    return result


# ---------------------------------------------------------------------------
# AGENT 2 — Summariser (phi3.5)
# Compresses long context before escalation to a heavyweight model
# ---------------------------------------------------------------------------
SUMMARISER_SYSTEM = """You are a precise text summariser.
Produce a dense summary that preserves all key facts, numbers, names, and intent.
The summary must be short enough to fit in 500 words.
Do not add commentary — only the summary."""


async def agent_summariser(text: str) -> str:
    """Summarise text if it exceeds the token threshold."""
    if approx_tokens(text) <= SUMMARISE_THRESHOLD:
        return text
    log.info(
        "Input ~%d tokens — running summariser first", approx_tokens(text)
    )
    messages = [
        {"role": "system", "content": SUMMARISER_SYSTEM},
        {"role": "user", "content": text},
    ]
    return await _ollama_complete(
        MODEL_PHI35, messages, max_tokens=700, temperature=0.2
    )


# ---------------------------------------------------------------------------
# AGENT 3 — Chat (Mistral 22B, falls back to phi3.5)
# ---------------------------------------------------------------------------
CHAT_SYSTEM = """You are a knowledgeable, concise AI assistant running locally on a Raspberry Pi 5.
Answer clearly and accurately. Never mention any cloud service or external API."""


async def agent_chat(query: str) -> str:
    """General-purpose reasoning. Uses Mistral 22B (or phi3.5 if Mistral is down)."""
    condensed = await agent_summariser(query)
    messages = [
        {"role": "system", "content": CHAT_SYSTEM},
        {"role": "user", "content": condensed},
    ]
    return await _mistral_complete(messages, max_tokens=1024)


# ---------------------------------------------------------------------------
# AGENT 4 — Coder (qwen2.5-coder:7b)
# ---------------------------------------------------------------------------
CODER_SYSTEM = """You are an expert software engineer and code reviewer.
Write clean, well-commented, production-ready code.
When reviewing code, identify bugs, suggest improvements, and explain reasoning.
Always specify the language/runtime at the top of code blocks."""


async def agent_coder(query: str) -> str:
    """Code generation and review via qwen2.5-coder:7b."""
    condensed = await agent_summariser(query)
    messages = [
        {"role": "system", "content": CODER_SYSTEM},
        {"role": "user", "content": condensed},
    ]
    return await _ollama_complete(
        MODEL_QWEN_CODER, messages, max_tokens=2048, temperature=0.2
    )


# ---------------------------------------------------------------------------
# AGENT 5 — RAG (phi3.5 + ChromaDB)
# ---------------------------------------------------------------------------
RAG_SYSTEM = """You are a precise retrieval assistant.
You have been given document excerpts retrieved from a knowledge base.
Answer the user question using ONLY the information in the provided excerpts.
If the excerpts do not contain enough information to answer, say so explicitly.
Cite the source document name when available."""

_chroma_client: chromadb.AsyncHttpClient | None = None
_chroma_collection: Any | None = None

CHROMA_COLLECTION_NAME = "agent_hub_knowledge"


async def _get_chroma_collection() -> Any | None:
    global _chroma_client, _chroma_collection
    if not _backend_status["chromadb"]:
        return None
    if _chroma_collection is not None:
        return _chroma_collection
    try:
        _chroma_client = await chromadb.AsyncHttpClient(
            host=CHROMA_HOST, port=CHROMA_PORT
        )
        _chroma_collection = await _chroma_client.get_or_create_collection(
            name=CHROMA_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        return _chroma_collection
    except Exception as exc:
        log.error("ChromaDB connection failed: %s", exc)
        return None


async def _chroma_query(query_text: str, n_results: int = 5) -> list[str]:
    """Return top-N document excerpts from ChromaDB."""
    collection = await _get_chroma_collection()
    if collection is None:
        return []
    try:
        results = await collection.query(
            query_texts=[query_text],
            n_results=n_results,
            include=["documents", "metadatas"],
        )
        docs: list[str] = []
        for i, doc in enumerate(results["documents"][0]):
            meta = results["metadatas"][0][i] if results["metadatas"] else {}
            source = meta.get("source", f"doc_{i+1}")
            docs.append(f"[{source}]\n{doc}")
        return docs
    except Exception as exc:
        log.error("ChromaDB query failed: %s", exc)
        return []


async def agent_rag(query: str) -> str:
    """Retrieve relevant context from ChromaDB then synthesise with phi3.5."""
    excerpts = await _chroma_query(query)
    if not excerpts:
        # Graceful degradation: answer from parametric memory
        log.warning("ChromaDB unavailable or empty — answering from parametric knowledge")
        return await _ollama_complete(
            MODEL_PHI35,
            [
                {"role": "system", "content": CHAT_SYSTEM},
                {
                    "role": "user",
                    "content": (
                        f"[Note: knowledge base is unavailable]\n\n{query}"
                    ),
                },
            ],
            max_tokens=800,
        )

    context_block = "\n\n---\n\n".join(excerpts)
    prompt = f"RETRIEVED EXCERPTS:\n\n{context_block}\n\nUSER QUESTION:\n{query}"
    messages = [
        {"role": "system", "content": RAG_SYSTEM},
        {"role": "user", "content": prompt},
    ]
    return await _ollama_complete(
        MODEL_PHI35, messages, max_tokens=1024, temperature=0.2
    )


# ---------------------------------------------------------------------------
# AGENT 6 — Synthesiser (Mistral 22B)
# Merges parallel agent outputs into a coherent final response
# ---------------------------------------------------------------------------
SYNTHESISER_SYSTEM = """You are a response synthesiser.
You receive several partial answers to different parts of a compound question.
Merge them into a single, coherent, well-structured response.
Remove redundancy, fix contradictions (prefer the more specific answer), and
ensure the final reply directly addresses the original user question.
Do not add meta-commentary about the synthesis process."""


async def agent_synthesiser(
    original_query: str, partial_answers: list[str]
) -> str:
    """Merge multiple partial answers into one coherent reply using Mistral 22B."""
    numbered = "\n\n".join(
        f"[Part {i+1}]\n{ans}" for i, ans in enumerate(partial_answers)
    )
    prompt = (
        f"ORIGINAL QUESTION:\n{original_query}\n\n"
        f"PARTIAL ANSWERS:\n{numbered}"
    )
    messages = [
        {"role": "system", "content": SYNTHESISER_SYSTEM},
        {"role": "user", "content": prompt},
    ]
    return await _mistral_complete(messages, max_tokens=2048)


# ---------------------------------------------------------------------------
# Parallel chunking helper
# Splits a multi-question query, runs each through the appropriate agent,
# then synthesises the results.
# ---------------------------------------------------------------------------
_SPLITTER_SYSTEM = """Split the following compound message into its individual questions or tasks.
Return ONLY a JSON array of strings. Example: ["What is X?", "How do I do Y?"]
Do not add any explanation or markdown."""


async def _split_questions(query: str) -> list[str]:
    messages = [
        {"role": "system", "content": _SPLITTER_SYSTEM},
        {"role": "user", "content": query},
    ]
    raw = await _ollama_complete(MODEL_PHI35, messages, max_tokens=256, temperature=0.0)
    raw = re.sub(r"```[a-z]*\n?", "", raw).strip()
    try:
        parts = json.loads(raw)
        if isinstance(parts, list) and len(parts) > 1:
            return [str(p) for p in parts]
    except json.JSONDecodeError:
        pass
    return [query]


async def _route_single(query: str, category: str) -> str:
    """Send a single (already-classified) query to the right specialist agent."""
    if category == "code":
        return await agent_coder(query)
    if category == "rag":
        return await agent_rag(query)
    return await agent_chat(query)


async def _orchestrate(query: str) -> str:
    """
    Core orchestration logic:
    1. Check cache
    2. Route to classify
    3. Summarise if too long
    4. Split if multi-question, run in parallel, synthesise
    5. Otherwise dispatch to single specialist
    """
    cache_key = _cache_key(query)
    cached = _cache_get(cache_key)
    if cached:
        log.info("Cache hit for query hash %s", cache_key[:8])
        return cached["response"]

    # Step 1: classify
    route = await agent_router(query)
    category: str = route.get("category", "general")
    is_multi: bool = bool(route.get("multi_question", False))
    log.info(
        "Router → category=%s  multi=%s  confidence=%.2f  tokens≈%d",
        category,
        is_multi,
        route.get("confidence", 0.0),
        approx_tokens(query),
    )

    # Step 2: if multi-question, split and run in parallel
    if is_multi:
        sub_queries = await _split_questions(query)
        if len(sub_queries) > 1:
            log.info("Running %d sub-queries in parallel", len(sub_queries))
            tasks = [_route_single(q, category) for q in sub_queries]
            partial_answers: list[str] = await asyncio.gather(*tasks)
            response = await agent_synthesiser(query, list(partial_answers))
            _cache_set(cache_key, {"response": response, "category": category})
            return response

    # Step 3: single-question dispatch
    response = await _route_single(query, category)
    _cache_set(cache_key, {"response": response, "category": category})
    return response


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Agent Hub",
    description="Local LLM sub-agent orchestrator — Raspberry Pi 5",
    version="1.0.0",
)


@app.on_event("startup")
async def _startup() -> None:
    log.info("Agent Hub starting — running initial health check")
    status = await refresh_backend_status()
    log.info("Backend status: %s", status)
    asyncio.create_task(_periodic_health_check())


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------
class ChatRequest(BaseModel):
    query: str
    system_prompt: str | None = None  # optional override

    class Config:
        json_schema_extra = {
            "example": {"query": "Explain the Fourier transform simply."}
        }


class ChatResponse(BaseModel):
    response: str
    category: str
    model_used: str
    approx_input_tokens: int
    cached: bool
    elapsed_ms: float


class CodeRequest(BaseModel):
    query: str

    class Config:
        json_schema_extra = {
            "example": {"query": "Write a Python async HTTP client with retry logic."}
        }


class CodeResponse(BaseModel):
    response: str
    approx_input_tokens: int
    elapsed_ms: float


class RAGRequest(BaseModel):
    query: str
    n_results: int = 5

    class Config:
        json_schema_extra = {
            "example": {"query": "What does my project doc say about authentication?"}
        }


class RAGResponse(BaseModel):
    response: str
    excerpts_found: int
    elapsed_ms: float


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.post("/chat", response_model=ChatResponse, summary="Auto-routed chat")
async def post_chat(req: ChatRequest) -> ChatResponse:
    """
    Main entry point. Automatically routes the query to the smallest capable
    model. Applies summarisation and parallel chunking where appropriate.
    """
    t0 = time.monotonic()
    cache_key = _cache_key(req.query)
    was_cached = bool(_cache_get(cache_key))

    # Optional system prompt injection: prepend to query text
    effective_query = req.query
    if req.system_prompt:
        effective_query = f"[SYSTEM]\n{req.system_prompt}\n\n[USER]\n{req.query}"

    # Classify to determine which model label to report
    route = await agent_router(effective_query)
    category: str = route.get("category", "general")

    if was_cached:
        cached_data = _cache_get(cache_key)
        return ChatResponse(
            response=cached_data["response"],
            category=cached_data.get("category", category),
            model_used="(cached)",
            approx_input_tokens=approx_tokens(effective_query),
            cached=True,
            elapsed_ms=round((time.monotonic() - t0) * 1000, 1),
        )

    response = await _orchestrate(effective_query)

    # Determine which model was actually used for reporting
    if category == "code":
        model_used = MODEL_QWEN_CODER
    elif category in ("general", "voice"):
        model_used = MODEL_MISTRAL if _backend_status["mistral"] else MODEL_PHI35
    else:
        model_used = MODEL_PHI35

    return ChatResponse(
        response=response,
        category=category,
        model_used=model_used,
        approx_input_tokens=approx_tokens(effective_query),
        cached=False,
        elapsed_ms=round((time.monotonic() - t0) * 1000, 1),
    )


@app.post("/code", response_model=CodeResponse, summary="Direct coder agent")
async def post_code(req: CodeRequest) -> CodeResponse:
    """
    Bypass the router and send directly to the qwen2.5-coder:7b agent.
    Summarises input > 2 000 tokens before processing.
    """
    t0 = time.monotonic()
    cache_key = _cache_key("code:" + req.query)
    cached = _cache_get(cache_key)
    if cached:
        return CodeResponse(
            response=cached["response"],
            approx_input_tokens=approx_tokens(req.query),
            elapsed_ms=round((time.monotonic() - t0) * 1000, 1),
        )

    response = await agent_coder(req.query)
    _cache_set(cache_key, {"response": response})
    return CodeResponse(
        response=response,
        approx_input_tokens=approx_tokens(req.query),
        elapsed_ms=round((time.monotonic() - t0) * 1000, 1),
    )


@app.post("/rag/query", response_model=RAGResponse, summary="Direct RAG lookup")
async def post_rag_query(req: RAGRequest) -> RAGResponse:
    """
    Retrieve from ChromaDB and synthesise an answer with phi3.5.
    If ChromaDB is unavailable, falls back to parametric knowledge.
    """
    t0 = time.monotonic()
    cache_key = _cache_key("rag:" + req.query)
    cached = _cache_get(cache_key)
    if cached:
        return RAGResponse(
            response=cached["response"],
            excerpts_found=cached.get("excerpts_found", 0),
            elapsed_ms=round((time.monotonic() - t0) * 1000, 1),
        )

    excerpts = await _chroma_query(req.query, n_results=req.n_results)
    response = await agent_rag(req.query)
    _cache_set(cache_key, {"response": response, "excerpts_found": len(excerpts)})
    return RAGResponse(
        response=response,
        excerpts_found=len(excerpts),
        elapsed_ms=round((time.monotonic() - t0) * 1000, 1),
    )


@app.get("/status", summary="Backend health status")
async def get_status() -> dict[str, Any]:
    """
    Refreshes and returns the health of every model backend and ChromaDB.
    """
    status = await refresh_backend_status()
    cache_entries = len(_cache)
    return {
        "backends": {
            "ollama": {
                "online": status["ollama"],
                "url": OLLAMA_BASE_URL,
                "models": [MODEL_PHI35, MODEL_LLAVA, MODEL_QWEN_CODER],
            },
            "mistral": {
                "online": status["mistral"],
                "url": LLAMA_CPP_BASE_URL,
                "models": [MODEL_MISTRAL],
                "note": "Falls back to phi3.5 when offline",
            },
            "chromadb": {
                "online": status["chromadb"],
                "url": f"http://{CHROMA_HOST}:{CHROMA_PORT}",
                "collection": CHROMA_COLLECTION_NAME,
            },
        },
        "cache": {
            "entries": cache_entries,
        },
        "token_budget": {
            "phi35_ctx": CTX_PHI35,
            "mistral_ctx": CTX_MISTRAL,
            "summarise_threshold": SUMMARISE_THRESHOLD,
        },
    }


@app.get("/agents", summary="List all sub-agents and their models")
async def get_agents() -> dict[str, Any]:
    """
    Returns metadata for every registered sub-agent including which model
    it is currently backed by (accounting for fallback state).
    """
    mistral_online = _backend_status["mistral"]
    chat_model = MODEL_MISTRAL if mistral_online else f"{MODEL_PHI35} (fallback)"
    synth_model = MODEL_MISTRAL if mistral_online else f"{MODEL_PHI35} (fallback)"

    return {
        "agents": [
            {
                "name": "router",
                "description": "Classifies query into category; triggers summariser and splitter",
                "model": MODEL_PHI35,
                "backend": "ollama",
                "ctx_window": CTX_PHI35,
            },
            {
                "name": "chat",
                "description": "General-purpose reasoning and conversation",
                "model": chat_model,
                "backend": "llama.cpp" if mistral_online else "ollama (fallback)",
                "ctx_window": CTX_MISTRAL if mistral_online else CTX_PHI35,
            },
            {
                "name": "coder",
                "description": "Code generation, review, and debugging",
                "model": MODEL_QWEN_CODER,
                "backend": "ollama",
                "ctx_window": CTX_PHI35,
            },
            {
                "name": "rag",
                "description": "Retrieval-augmented generation from ChromaDB",
                "model": MODEL_PHI35,
                "backend": "ollama + chromadb",
                "ctx_window": CTX_PHI35,
                "chroma_online": _backend_status["chromadb"],
            },
            {
                "name": "summariser",
                "description": f"Compresses input > {SUMMARISE_THRESHOLD} tokens before specialist dispatch",
                "model": MODEL_PHI35,
                "backend": "ollama",
                "ctx_window": CTX_PHI35,
            },
            {
                "name": "synthesiser",
                "description": "Merges parallel sub-query answers into a coherent response",
                "model": synth_model,
                "backend": "llama.cpp" if mistral_online else "ollama (fallback)",
                "ctx_window": CTX_MISTRAL if mistral_online else CTX_PHI35,
            },
        ]
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=7860,
        reload=False,
        log_level="info",
    )

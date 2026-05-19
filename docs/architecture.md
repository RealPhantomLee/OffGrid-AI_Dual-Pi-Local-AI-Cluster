# OffGridAI — Architecture & Design Decisions

## Design Philosophy

Local-first AI is not a compromise — it is a deliberate architectural choice that prioritises privacy, latency, autonomy, and long-term cost control. When inference runs on hardware you physically own, your prompts never leave your network. There are no API logs, no training data pipelines harvesting your queries, no third-party retention policies to audit. For a home assistant that knows your schedule, your documents, your conversations, and your home automation state, this matters enormously. The alternative — routing every query through a cloud API — is functionally equivalent to speaking your private thoughts into a microphone owned by a corporation with misaligned incentives.

Latency is the second motivation. A local model on a wired LAN responds in 15–60 seconds for a full Mistral 24B inference pass. That sounds slow compared to GPT-4 Turbo, but for offline-capable home automation, document Q&A, and code assistance, it is entirely acceptable — and crucially, it works when your ISP is down, when an API is rate-limited, when a provider has an outage, or when you simply do not want to pay per token for a task that runs a hundred times a day. There are no cold starts, no queuing delays from shared infrastructure, no throttling based on your subscription tier.

The third motivation is sovereignty. Subscription lock-in is a real risk: the model you depend on today can be deprecated, price-hiked, or quietly lobotomised by RLHF in the next version. Running open weights means your system behaves identically in six months as it does today. You can pin a model version, run it indefinitely, and reproduce its outputs exactly. You own the weights; nobody can take them away or change their behaviour remotely.

Finally, distributed inference across two nodes is what makes this economically viable at the hobbyist tier. Two Raspberry Pi 5 8GB boards cost less than a single high-end SBC with comparable RAM, and their combined 16 GB of LPDDR4X gives enough headroom to run a 24B quantised model with room for KV cache. The Raspberry Pi 5 specifically was chosen because it is the first Pi to expose PCIe 2.0 via the HAT+ connector, enabling the Hailo-8 AI accelerator for vision workloads and NVMe storage. Its Cortex-A76 cores with NEON/DOTPROD extensions allow llama.cpp to run quantised matrix multiplications efficiently, and its 5–12W power envelope makes 24/7 operation practical on a normal household circuit.

---

## System Architecture

```
                          ┌─────────────────────────────────────────────────────┐
                          │                  HOME LAN  192.168.1.x               │
                          │                                                       │
  Browser / API Client    │  ┌────────────────────────────────────────────────┐  │
  ─────────────────────   │  │                 aipi  (192.168.1.10)           │  │
                          │  │                                                │  │
  :3000  Open WebUI ──────┼──┤  open-webui       :3000  (Docker)             │  │
  :8080  Agent API ───────┼──┤  agent-hub        :8080  (FastAPI)            │  │
  :11434 Ollama API ──────┼──┤  ollama           :11434                      │  │
  :8265  Ray Dashboard ───┼──┤  ray-worker       :8265  (dashboard)          │  │
  :8001  RAG HTTP ────────┼──┤  rag-ingest       :8001  (ChromaDB HTTP)      │  │
  :9000  Whisper API ─────┼──┤  whisper-server   :9000  (HTTP)               │  │
  :8090  llama-server ────┼──┤  llama-server     :8090  (llama.cpp HTTP)     │  │
                          │  │                                                │  │
                          │  │  ┌──────────┐   PCIe 2.0 x1 (HAT+ connector) │  │
                          │  │  │ Hailo-8  │◄──────────────────────────────  │  │
                          │  │  │ AI HAT   │  YOLOv8, face det., pose est.   │  │
                          │  │  │ 26 TOPS  │                                 │  │
                          │  │  └──────────┘                                 │  │
                          │  └───────────────────────┬────────────────────────┘  │
                          │                          │                           │
                          │              RPC  TCP    │  :50051 → :50052          │
                          │              (1GbE LAN)  │  layer distribution       │
                          │                          ▼                           │
                          │  ┌────────────────────────────────────────────────┐  │
                          │  │                 jolly  (192.168.1.11)          │  │
                          │  │                                                │  │
                          │  │  rpc-server       :50052  (llama.cpp RPC)     │  │
                          │  │  (no user-facing services)                    │  │
                          │  │  8 GB RAM donated to model layer storage      │  │
                          │  └────────────────────────────────────────────────┘  │
                          └─────────────────────────────────────────────────────┘
                                            │
                          ┌─────────────────┴──────────────────────────────────┐
                          │          TAILSCALE OVERLAY  100.x.x.x              │
                          │  Encrypted WireGuard mesh — management SSH only    │
                          │  aipi: 100.x.x.a    jolly: 100.x.x.b              │
                          └────────────────────────────────────────────────────┘
```

---

## Node Roles

**aipi** is the coordinator node. It runs all user-facing services, holds the inference client (llama-server), manages the Ollama sub-agent pool, serves the Open WebUI frontend, exposes the agent-hub FastAPI gateway, runs Whisper for speech-to-text, maintains the ChromaDB RAG vector store, and drives the Hailo-8 via the TAPPAS vision pipeline. When a user submits a query, every routing decision and every response assembly happens on aipi.

**jolly** is a pure compute expansion node. It runs exactly one process: the llama.cpp RPC server on port 50052. It has no web interface, no Ollama instance, no database, and no user-accessible endpoints. Its sole purpose is to donate its 8 GB of RAM to the distributed model context, accepting layer tensors from aipi's llama-server and returning computed activations over the LAN TCP connection. jolly can be rebooted independently; llama-server on aipi will log an error and retry until jolly returns.

---

## Why llama.cpp RPC (Not Alternatives)

| Runtime | Multi-node distributed inference | ARM64 support | Quantised model support | Overhead |
|---|---|---|---|---|
| **Ollama** | No — single-node only | Yes | Yes (via llama.cpp internally) | Not applicable |
| **vLLM** | Yes (PagedAttention) | No — x86/CUDA only | Limited | High (Python overhead) |
| **ExLlamaV2** | No — single-node | Limited | Yes (EXL2) | Low |
| **llama.cpp RPC** | **Yes — purpose-built** | **Yes — first-class** | **Yes — all GGUF quants** | **Minimal (binary TCP)** |
| **LMStudio** | No — desktop GUI only | Yes | Yes | High |

llama.cpp's RPC backend was introduced specifically to distribute model layers across multiple machines with independent memory pools. The build flag `-DGGML_RPC=ON` compiles both the server binary (run on the worker node) and the client-side RPC backend (linked into llama-server on the coordinator). At startup, llama-server connects to each registered RPC endpoint and negotiates available memory. Layer assignment is proportional to the memory reported by each node: aipi claims roughly 5.5 GB of layers; jolly claims the remainder up to the model's full 14.3 GB footprint.

The wire protocol is a compact binary format over TCP — not gRPC, not HTTP. This minimises per-layer-boundary overhead to approximately 1–2 ms on a Gigabit LAN. During a forward pass, activations are streamed from one node to the next as each layer boundary is crossed; there is no full-tensor serialisation round-trip. For a 24B model at Q4_K_M quantisation with 40 transformer layers, approximately 8–12 layer boundaries cross the wire per token, adding roughly 10–24 ms of network latency per token on 1GbE. This is acceptable at 2–6 t/s generation speed.

**Critical implementation note:** Add `--no-repack` to the coordinator's llama-server command. Without it, llama.cpp attempts to allocate a full-model CPU_REPACK buffer (~13.9 GB) on the local node even when most weights are held on remote nodes, causing an OOM crash on startup. With `--no-repack`, weights are used in their quantised layout directly without repacking to a different format in memory.

---

## Model Selection Rationale

**Primary model: Mistral Small 3.1 24B Q4_K_M**

Mistral Small 3.1 24B at Q4_K_M quantisation occupies 14.3 GB on disk and in RAM. With 16 GB combined across two nodes and a kernel-plus-services overhead of approximately 1.2 GB, this leaves roughly 450 MB of headroom for KV cache at a 4096-token context window — tight but functional. Key selection criteria:

| Property | Value | Significance |
|---|---|---|
| Parameters | 23.5 billion | Strong reasoning, coding, multilingual |
| File size (Q4_K_M) | 14.3 GB | Fits in 16 GB combined with ~1.7 GB headroom |
| Context window | 128K base (4096 deployed) | Expandable — limited here for RAM efficiency |
| Quantisation | Q4_K_M | Best quality/size tradeoff at 4-bit |
| Perplexity vs FP16 | < 0.5 degradation | Effectively indistinguishable in practice |

Comparison by tier:

- **7B models (e.g. Llama 3.2 7B):** Noticeably weaker on multi-step reasoning and code generation. Suitable as a fast router but not as a primary reasoning engine. Quality gap from 7B to 24B is substantial, not incremental.
- **13B models:** A meaningful step up, but Mistral Small 3.1 24B remains superior across benchmarks while fitting within the same hardware budget. Not worth the quality compromise.
- **34B models (e.g. Llama 3.1 34B Q4_K_M, ~20 GB):** Require a third node or 16 GB single-board computer. Out of scope for the current two-node setup, but a natural next upgrade.

**Sub-agent models (via Ollama on aipi):**

| Model | Purpose | RAM footprint | Why this model |
|---|---|---|---|
| `phi3.5` | Router, summariser, fast chat | ~2.2 GB | 2.2B params — fast classification, low latency |
| `llava-phi3` | Vision + language (non-Hailo path) | ~2.9 GB | Multimodal, runs on CPU, lightweight |
| `qwen2.5-coder:7b` | Code completion and review | ~4.7 GB | State-of-the-art coding at 7B tier |

---

## Token Maximisation Strategy

Every query passes through a 5-stage pipeline designed to minimise expensive Mistral 24B calls while maintaining quality on complex tasks.

```
Incoming Query
      │
      ▼
┌─────────────────────────────┐
│  Stage 1: SHA-256 Cache     │  HIT  → return cached response (~0 tokens, ~0 ms)
│  (Redis / file cache)       │  MISS → continue
└──────────────┬──────────────┘
               │ MISS
               ▼
┌─────────────────────────────┐
│  Stage 2: phi3.5 Router     │  ~200 tokens to classify intent
│  Classify: code / vision /  │  Confidence score attached to classification
│  rag / chat / escalate      │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│  Stage 3: Context Check     │  If context > 2000 tokens:
│  (phi3.5 Summariser)        │    summarise to <= 800 tokens before routing
│                             │  If context <= 2000 tokens: pass through
└──────────────┬──────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────────────┐
│  Stage 4: Specialist Routing                                 │
│                                                              │
│  code ──────► qwen2.5-coder:7b (Ollama)                     │
│  vision ─────► llava-phi3 (Ollama) or Hailo-8 + llava-phi3  │
│  rag ────────► ChromaDB retrieval → phi3.5 synthesis        │
│  chat ────────► phi3.5 (Ollama)                              │
│                                                              │
└──────────────┬───────────────────────────────────────────────┘
               │
               │ phi3.5 confidence < threshold  OR
               │ specialist returns "insufficient"
               ▼
┌─────────────────────────────┐
│  Stage 5: Mistral 24B       │  Full inference via llama-server + RPC
│  Escalation                 │  Most expensive path; invoked only when needed
└──────────────┬──────────────┘
               │
               ▼
          Response
    (written to cache for future hits)
```

This pipeline reduces Mistral 24B calls to approximately 20–30% of total queries in typical home-assistant workloads, cutting average inference cost per query by 70–80% compared to routing everything to the primary model. The SHA-256 cache is particularly effective for repeated questions about the same documents or recurring home-automation status queries.

---

## Service Dependency Graph

```
systemd startup order and dependencies:

docker.service
  └─► open-webui.service  (soft dep on ollama, llama-server — shows offline in UI if absent)

ollama.service              (standalone — no hard deps)
  └─► agent-hub.service   (requires ollama for sub-agent routing)

chromadb.service            (standalone — no hard deps)
  ├─► agent-hub.service   (soft dep — RAG path degrades gracefully without it)
  └─► rag-ingest.timer    (hard dep — timer pauses if chromadb unreachable)

jolly:rpc-server.service   (must be running on jolly BEFORE aipi llama-server starts)
  └─► llama-server.service (hard dep — exits on startup if RPC endpoint unreachable)

llama-server.service
  └─► agent-hub.service   (soft dep — escalation path unavailable without it)

whisper-server.service      (standalone — no deps, no dependents at startup)
```

Practical consequence: boot jolly first, confirm rpc-server is active (check with `systemctl status rpc-server` on jolly), then start aipi services in order. The `llama-server.service` unit on aipi has `Requires=network-online.target` and a 30-second `ExecStartPre` sleep to allow jolly's rpc-server to become reachable before the connection is attempted.

---

## Network Topology

Two physically separate network paths are used for different purposes and must not be conflated:

**Path 1 — LAN (192.168.1.x), Gigabit Ethernet, wired**

Used for: llama.cpp RPC inference traffic between aipi and jolly. This path is latency-critical. Each transformer layer boundary crossing adds approximately 1–2 ms. For a 24B model with 40 layers distributed across two nodes, this contributes 10–24 ms of network overhead per generated token. Any jitter — characteristic of WiFi, VPNs, and virtual network interfaces — causes inference stalls and RPC timeout errors. This path must be physically wired Gigabit Ethernet. The connection must be direct or via a dumb Gigabit switch; managed switches with STP convergence delays can cause intermittent RPC failures during boot.

**Path 2 — Tailscale overlay (100.x.x.x), WireGuard encrypted**

Used for: management SSH from external devices, remote administration, secure access from mobile clients. Tailscale creates a zero-config WireGuard mesh that works through NAT without port forwarding. All management traffic is encrypted end-to-end. This path adds 5–15 ms latency over the direct LAN path and is not used for inference. jolly's SSH port is restricted by UFW to Tailscale connections only; the raw LAN IP cannot SSH into jolly directly.

---

## Security Model

The current deployment is designed for a trusted home LAN with no direct WAN exposure. The security posture reflects this:

| Service | Binding | Auth | Firewall |
|---|---|---|---|
| llama-server | 0.0.0.0:8090 | None | Home LAN only (no WAN) |
| Ollama | 0.0.0.0:11434 | None | Home LAN only |
| Open WebUI | 0.0.0.0:3000 | Login required | WEBUI_SECRET_KEY set (64-char hex) |
| Agent Hub | 0.0.0.0:8080 | None | Home LAN only |
| Whisper STT | 0.0.0.0:9000 | None | Home LAN only |
| ChromaDB | 0.0.0.0:8001 | None | Home LAN only |
| jolly RPC | 0.0.0.0:50052 | UFW: aipi IP only | Port 50052 restricted to aipi's LAN IP |

Additional measures in place:
- `WEBUI_SECRET_KEY` is set to a cryptographically random 64-character hex string. This protects session cookies in Open WebUI against forgery.
- Open WebUI requires user account login. Anonymous access is disabled.
- No telemetry is transmitted by any service. Ollama's telemetry flag is explicitly disabled.
- No external API calls are made. No analytics, no crash reporting, no model download metrics after initial setup.

**Steps required before any internet exposure:**
1. Deploy an nginx reverse proxy in front of all HTTP services (ports 3000, 8080, 9000, 8001).
2. Add HTTP Basic Auth or OAuth2 proxy (Authelia or oauth2-proxy) on all endpoints.
3. Obtain a TLS certificate (Let's Encrypt via Certbot or DNS-01 challenge).
4. Move llama-server to bind on `127.0.0.1` only; expose only via agent-hub.
5. Rate-limit the agent-hub API endpoint (nginx `limit_req_zone`).
6. Restrict Tailscale ACLs to named devices only (remove the default allow-all policy).

---

## Design Tradeoffs

Honest assessment of the limitations of this architecture:

| Tradeoff | Reality | Mitigation |
|---|---|---|
| **CPU inference speed** | 2–6 tokens/sec on Mistral 24B (GPU: 50–200 t/s) | Acceptable for async home-assistant use; smaller models for latency-sensitive paths |
| **SD card wear** | 14 GB model loaded on every restart + ChromaDB continuous writes = heavy write amplification; typical SD card life 6–12 months of daily use | NVMe SSD via M.2 HAT strongly recommended; on jolly there is no PCIe conflict |
| **Single point of failure** | aipi failure kills all user-facing services | UPS on aipi; NVMe for filesystem durability; jolly is stateless and recovers quickly |
| **Context limit** | 4096 tokens default (model is 128K capable) | 4096 chosen conservatively for RAM headroom; expandable to 8192 by reducing Ollama concurrency; 16384+ requires a third node |
| **Cold start time** | 3 min from SD card; 30 sec from NVMe for model load | Schedule llama-server to start at boot and stay resident; use `--keep-in-memory` flag |
| **No GPU acceleration** | Pi 5 has no CUDA/ROCm-capable GPU | Hailo-8 handles vision workloads at 26 TOPS; CPU-only is the transformer inference path on this hardware |
| **RPC network overhead** | +50–200 ms added to time-to-first-token vs fully local | Imperceptible during reading; 2.5GbE switch approximately halves this |

---

## Extensibility

The architecture is designed to accept additional nodes and capabilities without structural changes:

**Third node (24 GB total RAM):** Adding a third Raspberry Pi 5 8GB increases combined RAM to 24 GB. This is sufficient for Llama 3.1 34B Q4_K_M (~20 GB) with comfortable KV cache headroom. Add the node's IP to llama-server's `--rpc` argument: `--rpc jolly:50052 --rpc node3:50052`. llama.cpp distributes layers proportionally across all three RPC endpoints automatically; no other changes are required.

**Hailo-8 vision pipeline:** The TAPPAS runtime on aipi drives the Hailo-8 for real-time object detection at 26 TOPS. Detection results are serialised to JSON and injected into the agent-hub context pipeline, giving Mistral 24B access to structured vision output (bounding boxes, class labels, confidence scores) without running a heavy multimodal LLM on every camera frame. The pipeline is: GStreamer source → Hailo TAPPAS element → JSON sink → agent-hub context store.

**Voice pipeline:** OpenWakeWord (wake word detection, runs on CPU) → Whisper STT (already running on port 9000) → agent-hub (inference) → Piper TTS (text-to-speech synthesis, CPU-only). This creates a fully local voice assistant with no cloud dependencies. Hardware required: USB microphone and speaker or 3.5mm audio output.

**Home automation:** An MQTT bridge service can subscribe to Home Assistant's MQTT broker and publish sensor readings, device states, and automation events into the agent-hub context store. This enables natural-language queries against live home state ("Is the back door locked?", "What is the living room temperature?") without polling REST APIs on every query.

**Quantisation upgrade:** When llama.cpp adds further optimised ARM DOTPROD kernels, Q5_K_M or Q6_K quantisation becomes feasible within the same 16 GB RAM budget, improving output quality without any hardware changes. Monitor llama.cpp release notes for ARM-specific performance improvements.

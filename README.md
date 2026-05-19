# OffGridAI — Sovereign Local AI Cluster

> A fully private, cloud-free AI infrastructure running on two Raspberry Pi 5s with a Hailo-8 AI accelerator.
> No subscriptions. No telemetry. No data leaving your network.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Hardware: RPi 5](https://img.shields.io/badge/Hardware-Raspberry%20Pi%205-red)](docs/hardware-guide.md)
[![Model: Mistral 24B](https://img.shields.io/badge/Model-Mistral%20Small%203.1%2024B-blue)](docs/architecture.md)
[![Accelerator: Hailo-8](https://img.shields.io/badge/Accelerator-Hailo--8%2026%20TOPS-green)](docs/hailo-setup.md)

---

## What is OffGridAI?

OffGridAI is a complete, production-ready local AI stack built on commodity ARM hardware. It pools the RAM of two Raspberry Pi 5s (16 GB combined) using llama.cpp's RPC distributed inference protocol to run a 24-billion-parameter language model that would otherwise require a dedicated GPU server.

The Hailo-8 AI accelerator (26 TOPS, connected via PCIe) handles vision inference — real-time object detection, scene understanding, and camera event processing — while the LLM handles reasoning, code generation, and conversation.

Everything runs locally. Nothing is sent to any cloud.

### Why this exists

Commercial AI services require trusting third parties with your queries, your context, your patterns of thought. OffGridAI is the answer to that dependency: a system that is owned, operated, and understood entirely by its user.

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                    aipi (Primary Node)               │
│          Raspberry Pi 5 · 8GB · Hailo-8 HAT         │
│                                                      │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────┐  │
│  │ llama-server │  │    Ollama    │  │  Hailo-8  │  │
│  │  port 8080   │  │  port 11434  │  │ /dev/hailo│  │
│  │  Mistral 24B │  │ phi3.5       │  │ 26 TOPS   │  │
│  │  (split RPC) │  │ llava-phi3   │  │ vision    │  │
│  └──────┬───────┘  │ qwen2.5-code │  └───────────┘  │
│         │ RPC      └──────────────┘                  │
│  ┌──────▼───────────────────────────────────────┐    │
│  │               Agent Hub (port 8000)          │    │
│  │  router → coder / chat / vision / rag / STT  │    │
│  └──────────────────────────────────────────────┘    │
│                                                      │
│  ┌────────────┐  ┌──────────┐  ┌─────────────────┐  │
│  │ Open WebUI │  │  Whisper │  │    ChromaDB     │  │
│  │  port 3000 │  │ port 9000│  │    port 8001    │  │
│  └────────────┘  └──────────┘  └─────────────────┘  │
└─────────────────────────┬───────────────────────────┘
                          │ llama.cpp RPC (port 50052)
                          │ 192.168.1.x (LAN direct)
┌─────────────────────────▼───────────────────────────┐
│                   jolly (Worker Node)                │
│          Raspberry Pi 5 · 8GB · Arch Linux          │
│                                                      │
│  ┌────────────────────────────────────────────┐      │
│  │         llama-rpc-server (port 50052)      │      │
│  │    Exposes 8GB RAM as RPC backend          │      │
│  │    Hosts ~half the Mistral 24B layers      │      │
│  └────────────────────────────────────────────┘      │
└─────────────────────────────────────────────────────┘
```

**Combined compute:** 16 GB RAM · 8 CPU cores per node · 26 TOPS neural accelerator

---

## Quickstart

> **Impatient path** — run this on your primary node (Raspberry Pi OS, Hailo-8 installed):

```bash
# Set your HuggingFace token (needed for the gated model download)
export HF_TOKEN=hf_your_token_here

# Run the full setup (takes ~30–60 min including model download)
curl -fsSL https://raw.githubusercontent.com/RealPhantomLee/offgrid-ai_dual-pi-local-ai-cluster/main/setup/setup-aipi.sh | sudo -E bash
```

Then open `http://<AIPI_IP>:3000` in your browser.

For the deep, step-by-step path see [docs/architecture.md](docs/architecture.md).

---

## What You Get

| Service | Port | Description |
|---|---|---|
| **Open WebUI** | 3000 | Browser chat interface — connects to all models |
| **llama-server** | 8080 | OpenAI-compatible API, Mistral Small 3.1 24B |
| **Ollama** | 11434 | Fast sub-agent models (phi3.5, llava-phi3, qwen2.5-coder) |
| **Agent Hub** | 8000 | Intelligent query router — minimizes token spend |
| **Whisper STT** | 9000 | Speech-to-text transcription endpoint |
| **ChromaDB** | 8001 | Local vector knowledge base (auto-updated daily) |
| **Hailo-8** | — | 26 TOPS vision accelerator, `/dev/hailo0` |

All services run as systemd units, restart on failure, and survive reboots.

---

## Hardware Requirements

| Component | Spec | Notes |
|---|---|---|
| Primary node | Raspberry Pi 5, 8GB | The 4GB model will not run Mistral 24B |
| Worker node | Raspberry Pi 5, 8GB | Any ARM64 board with 8GB works |
| AI accelerator | Hailo-8 AI HAT | PCIe M.2 form factor, RPi 5 only |
| Storage | 32GB+ SD card (primary) | NVMe strongly recommended — SD wears fast |
| Power | Official RPi 5 27W USB-C PSU × 2 | Underpowering causes random crashes |
| Network | Gigabit switch | RPC inference over 1GbE works; 2.5GbE is better |

**Total cost:** ~$250–350 USD (excluding storage and network switch)

---

## The Model

**Mistral Small 3.1 24B (Q4_K_M quantization)**

- Parameters: 23.5 billion
- File size: 14.3 GB
- Context: 4096 tokens (expandable to 8192+ with more RAM)
- Capabilities: reasoning, code generation, instruction-following, multilingual
- Memory split: ~7GB on aipi, ~7GB on jolly via RPC

Why this model? It's the largest model that fits comfortably in 16 GB combined RAM with headroom for the OS and services. It outperforms every 7B and 13B model substantially, and matches GPT-3.5-level quality on most tasks.

Sub-agent models (run on Ollama, single-node):
- `phi3.5` — fast routing and summarization
- `llava-phi3` — vision + multimodal tasks
- `qwen2.5-coder:7b` — code generation

---

## Sub-Agent Architecture

The Agent Hub (`agent-hub/main.py`) routes every query to the smallest capable model:

```
Query → Router (phi3.5, ~200 tokens)
          ├── simple/fast  → phi3.5 directly
          ├── code         → qwen2.5-coder:7b
          ├── vision       → llava-phi3
          ├── knowledge    → phi3.5 + ChromaDB RAG
          └── complex      → Mistral 24B (escalation gate)
```

**Token maximization strategy:**
1. Route first — classify before spending tokens on the big model
2. Summarize context if >2000 tokens before escalating
3. Cache identical/near-duplicate queries (SHA-256)
4. Only escalate to Mistral 24B when phi3.5 confidence is low

---

## RAG Knowledge Base

ChromaDB runs locally and is automatically populated with:
- Wikipedia featured articles
- arXiv abstracts (cs.AI, cs.LG, cs.CL)
- Local documents from `~/Documents/`

A systemd timer runs the ingestion pipeline daily at 2:00 AM. The LLM cites whether answers come from its training data or the local knowledge base.

---

## Security Posture

All services are local-network-only by design:
- No ports are exposed to the internet
- Tailscale provides encrypted overlay networking between nodes
- Open WebUI requires authentication (set `WEBUI_SECRET_KEY`)
- ChromaDB, llama-server, and Agent Hub are LAN-only; put a reverse proxy in front if you need external access
- No telemetry, no analytics, no callbacks to any external service

See [docs/architecture.md](docs/architecture.md) for the full security model.

---

## Roadmap

- [ ] **Voice pipeline** — Piper TTS + OpenWakeWord for hands-free interaction
- [ ] **Vision pipeline** — Hailo TAPPAS + GStreamer → real-time camera object detection → LLM context injection
- [ ] **Home automation** — MQTT bridge for Home Assistant / smart device control
- [ ] **Multi-node expansion** — Add a third node to reach 24GB combined (enables 34B models)
- [ ] **NVMe optimization** — Model load time from ~3 min (SD) to ~30 sec (NVMe)
- [ ] **OpenBLAS rebuild** — 20-40% inference speed improvement on RPi 5

---

## Deep Documentation

- [Architecture & Design Decisions](docs/architecture.md)
- [Hardware Guide & BOM](docs/hardware-guide.md)
- [Hailo-8 Driver Setup](docs/hailo-setup.md)
- [llama.cpp RPC Distributed Inference](docs/llama-rpc.md)
- [Agent Hub & Token Budget Strategy](docs/agent-hub.md)
- [RAG Pipeline](docs/rag-pipeline.md)

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Hardware compatibility reports, alternative model configurations, and documentation improvements are especially welcome.

---

## License

MIT — Copyright (c) 2026 [RealPhantomLee](https://github.com/RealPhantomLee)

# Benchmarks

Performance measurements for the verified hardware configuration:
- **aipi:** Raspberry Pi 5 8GB + Hailo-8 AI HAT + 32GB SD card
- **jolly:** Raspberry Pi 5 8GB + 32GB SD card
- **Network:** Both nodes on Gigabit Ethernet switch
- **Model:** Mistral Small 3.1 24B Q4_K_M (14.3 GB)
- **llama-server flags:** `--ctx-size 4096 --n-gpu-layers 0 --no-repack --rpc 192.168.1.16:50052`

## Inference Speed (Mistral Small 3.1 24B)

| Metric | Value | Notes |
|---|---|---|
| Token generation (t/s) | 2–5 t/s | Varies with context length; higher for shorter contexts |
| Time to first token (TTFT) | 15–60 sec | Scales with prompt token count |
| TTFT — short prompt (<50 tokens) | ~15–20 sec | Model layers loading + first token |
| TTFT — long prompt (>500 tokens) | ~45–60 sec | Prefill computation across both nodes |
| Model load time (cold start) | ~3 min | From 32GB SD card; ~30 sec with NVMe |
| Context window used | 4096 tokens | Expandable to 8192 with careful RAM management |

## Ollama Sub-Agent Speed

| Model | Size | Typical t/s | TTFT | Use Case |
|---|---|---|---|---|
| phi3.5 | 2.2 GB | 8–15 t/s | 3–5 sec | Router, summarizer, fast queries |
| llava-phi3 | 2.9 GB | 5–10 t/s | 5–8 sec | Vision QA, multimodal |
| qwen2.5-coder:7b | 4.7 GB | 4–8 t/s | 8–12 sec | Code generation |

Ollama models run on aipi's local CPU only (no RPC needed for sub-3B models).

## Memory Distribution

```
[llama-server startup log]
- CPU     : aipi   — 8062 MiB total, ~7000 MiB used by model layers + KV cache
- RPC0    : jolly  — 7953 MiB total, ~7000 MiB used by model layers
- Projected host memory: ~1277 MiB on aipi for non-tensor overhead
```

Combined active RAM for Mistral 24B: ~14.3 GB split ~50/50 across both nodes.

## Network RPC Overhead

| Metric | Value |
|---|---|
| LAN latency (1GbE) | ~0.3 ms round-trip |
| RPC crossings per token | ~20 (40 transformer layers, ~half on jolly) |
| Estimated RPC overhead per token | ~6–10 ms |
| As % of total inference time | ~3–5% at 3 t/s |

The RPC overhead is small relative to the compute cost on the Cortex-A76 CPUs. The bottleneck is CPU SIMD throughput, not network bandwidth.

## System Resources Under Load

| Resource | aipi | jolly |
|---|---|---|
| CPU usage | ~95–100% (all 4 cores) | ~95–100% (all 4 cores) |
| RAM usage | ~7.2 GB / 8 GB | ~7.1 GB / 8 GB |
| CPU temperature (active cooler) | 68–74°C | 65–70°C |
| Power draw | ~11–13W | ~9–11W |
| **Combined power draw** | **~20–24W** | |

## Comparative Context

| Setup | t/s (similar quality model) | Hardware cost |
|---|---|---|
| OffGridAI (this setup) | 2–5 t/s | ~$400 |
| RTX 4060 Ti (16GB VRAM) | 50–80 t/s | ~$500 (GPU alone) |
| Mac Mini M4 (16GB) | 30–50 t/s | $600 |
| Cloud API (GPT-4o equivalent) | ~50 t/s output | $0.01–0.03/1K tokens |
| Groq LPU (free tier) | ~300 t/s | Free (rate-limited) |

**OffGridAI's value is not inference speed** — it is privacy, sovereignty, zero operating cost, and zero network dependency. At 3 t/s, a 200-word response takes ~70 seconds. For interactive home use and async tasks, this is acceptable.

## Reproduction

To run your own benchmarks:

```bash
# Token generation speed test
time curl -s http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mistral-small-3.1-24b-instruct-Q4_K_M.gguf",
    "messages": [{"role": "user", "content": "Count from 1 to 50, one number per line."}],
    "max_tokens": 150
  }' | python3 -c "
import sys, json
r = json.load(sys.stdin)
u = r['usage']
print(f'Completion tokens: {u[\"completion_tokens\"]}')
print(f'Prompt tokens: {u[\"prompt_tokens\"]}')
print(f'Response: {r[\"choices\"][0][\"message\"][\"content\"][:100]}...')
"

# Check RAM on both nodes
free -h
ssh jolly@192.168.1.16 "free -h"

# Monitor temperature during inference
watch -n 2 'vcgencmd measure_temp'
```

## Upgrade Projections

| Upgrade | Investment | Expected improvement |
|---|---|---|
| NVMe SSD on aipi | $45 (SSD + M.2 HAT) | Model load: 3 min → 30 sec |
| OpenBLAS rebuild | $0 (time only) | ~20–40% faster t/s |
| Third RPi 5 8GB | ~$100 (Pi + PSU + cooler) | Run 32B model; +15% quality |
| 2.5GbE switch | $35 | Halve RPC network overhead |

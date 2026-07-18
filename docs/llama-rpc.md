# llama.cpp RPC: Distributed Inference Across Two Raspberry Pi 5s

This document covers everything you need to know to run llama.cpp in RPC (Remote Procedure Call) mode across the aipi + jolly dual-Pi cluster: how the protocol works internally, the critical build and runtime flags required on our hardware, startup log interpretation, firewall configuration, and realistic performance expectations.

---

## 1. How llama.cpp RPC Works

llama.cpp's RPC system lets a single llama-server instance distribute model layers across multiple physical machines. The `--rpc` flag registers one or more remote backends before model load begins.

**Layer distribution** is proportional to available memory. When you pass `--rpc jolly:50052`, llama.cpp queries the worker for its free memory, then assigns a share of the model's transformer layers to that backend. The allocation is computed at load time: layers are split so that each backend holds approximately `(backend_free_mem / total_free_mem_across_all_backends)` of the total weight. The host node (aipi) acts as the orchestrator and retains any layers that don't fit on remote backends.

**Tensor operations** for each layer are computed on whichever backend holds those weights. When the forward pass reaches a remotely-held layer, aipi serialises the activation tensor and sends it to jolly over TCP. jolly performs the matrix multiply, sends the result back, and the forward pass continues on aipi. There is no shared memory — every inter-layer activation crosses the wire.

**Communication protocol** is a custom binary framing over raw TCP, not gRPC or HTTP. Each message carries an opcode, a payload length, and the tensor data. The protocol is synchronous per-layer: the server blocks waiting for jolly's result before proceeding to the next layer. This means network latency adds directly to time-to-first-token (TTFT).

**Port**: the RPC worker listens on port 50052 by default. You can override this with `--port <n>` when starting `rpc-server`.

---

## 2. The --no-repack Fix (Critical)

This is the most important operational detail for running llama.cpp RPC on a Pi 5 with 8GB RAM.

**What CPU_REPACK does**: In newer llama.cpp builds (roughly late 2024 and beyond), CPU weight repacking (`GGML_CPU_REPACK`) attempts to allocate a single contiguous buffer containing all model weights rearranged into an optimised layout for faster CPU matrix multiplication (specifically, the `aarch64` NEON GEMM kernels). The buffer is allocated for the full model, not just the locally-held layers.

**The problem with RPC**: With `--rpc`, this repacked buffer is still allocated on the *local* orchestrating node — in our case, aipi — even for layers whose weights will actually live and be computed on jolly. For Mistral 24B Q4_K_M, this buffer is approximately 13.3 GB. aipi has 8 GB total. The allocation fails with:

```
ggml_backend_cpu_buffer_type_alloc_buffer: failed to allocate buffer of size 13946880000
```

This error was confirmed on our aipi node and will appear immediately at model load time, before inference begins.

**The fix**: Pass `--no-repack` in the `ExecStart` line of your `llama-server.service` unit file:

```ini
ExecStart=/home/merry/llama.cpp/build/bin/llama-server \
    --model /home/merry/models/mistral-24b-q4_k_m.gguf \
    --rpc <JOLLY_LOCAL_IP>:50052 \
    --host 0.0.0.0 \
    --port 8080 \
    --ctx-size 4096 \
    --no-repack
```

`--no-repack` disables the repacking optimisation entirely. CPU matrix multiplication will use the standard unoptimised layout. The speed penalty is real but modest on Cortex-A76 — DOTPROD and NEON still apply to the standard kernels — and it is the only way to load a model larger than your local RAM with RPC.

---

## 3. The rpc-server Binary Name Change

In llama.cpp builds from approximately mid-2024 onwards, the RPC worker binary was renamed from `llama-rpc-server` to `rpc-server`. If you built from a recent commit and are looking for `llama-rpc-server`, you won't find it.

Check your build:

```bash
ls ~/llama.cpp/build/bin/ | grep -i rpc
```

On our jolly node this outputs `rpc-server`. Your systemd unit for jolly should reference this name:

```ini
ExecStart=/home/merry/llama.cpp/build/bin/rpc-server \
    --host 0.0.0.0 \
    --port 50052
```

If you built from source before mid-2024 and have `llama-rpc-server`, that binary is functionally identical — the rename was cosmetic. Just make sure your service files and documentation are consistent.

---

## 4. Build Instructions

Both nodes need llama.cpp built with RPC support enabled. RPC is not compiled in by default.

### Prerequisites (both nodes)

**On Raspberry Pi OS (Debian-based) — aipi:**

```bash
sudo apt update
sudo apt install -y build-essential cmake curl git
```

**On Arch Linux — jolly:**

```bash
sudo pacman -S --noconfirm base-devel cmake curl git
```

### Clone and Build

```bash
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
mkdir build && cd build
cmake .. -DGGML_RPC=ON -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release -j$(nproc)
```

The `-DGGML_RPC=ON` flag is mandatory. Without it, neither `rpc-server` nor the `--rpc` flag in `llama-server` will be present.

### CPU Feature Auto-Detection on RPi 5

The Raspberry Pi 5's Cortex-A76 cores expose three instruction set extensions that llama.cpp uses for accelerated matrix multiplication:

- **NEON**: 128-bit SIMD, the baseline ARM vector extension
- **ARM_FMA**: Fused multiply-add, reduces rounding error and latency
- **DOTPROD**: 8-bit dot product instruction, the most significant for quantised inference

llama.cpp detects these automatically at build time (no flags required). You can confirm they were detected in the startup log:

```
CPU : NEON = 1 | ARM_FMA = 1 | DOTPROD = 1
```

All three values should be `1` on a Pi 5. If any are `0`, your build environment is missing the relevant compiler flags — verify you are compiling natively on the Pi (not cross-compiling) and that your compiler is GCC 8+ or Clang 7+.

---

## 5. Startup Log Interpretation

When llama-server starts with `--rpc`, the startup log tells you the state of every backend before the first token is generated. Here is an annotated excerpt from a successful Mistral 24B load on our cluster:

```
llama_model_load: loading model from '/home/merry/models/mistral-24b-q4_k_m.gguf'
...
CPU : NEON = 1 | ARM_FMA = 1 | DOTPROD = 1 | ...
...
RPC0: <JOLLY_LOCAL_IP>:50052 (7953 MiB, 7953 MiB free)
...
llama_model_load: offloading N layers to RPC0
...
llm_load_tensors: CPU_Mapped model buffer size = XXXX MiB
llm_load_tensors: projected to use 1277 MiB of host memory
```

Line-by-line interpretation:

| Log line | Meaning |
|---|---|
| `RPC0: <JOLLY_LOCAL_IP>:50052 (7953 MiB, 7953 MiB free)` | jolly is reachable at that IP and port. First number is total memory reported by the worker; second is free memory available for layer allocation. 7953 MiB ≈ 7.8 GB, confirming jolly's 8 GB is mostly free at startup. |
| `offloading N layers to RPC0` | The number of transformer layers assigned to jolly. For Mistral 24B, the model has 40 layers; layer distribution depends on relative free memory between aipi and jolly. |
| `projected to use 1277 MiB of host memory` | aipi's expected RAM usage for context buffers, KV cache, scratch space, and the locally-held layers. This is the number to watch — it must fit within aipi's available RAM after the OS, Ollama, and other services are loaded. |

If `RPC0` does not appear in the log, the `rpc-server` on jolly is not running or not reachable. See the troubleshooting table in section 9.

---

## 6. UFW Firewall Configuration

Port 50052 must be open on jolly (the worker node) so that aipi can connect. On Arch Linux, UFW is not installed by default, but if it is enabled on jolly you must explicitly allow the connection.

**Best practice**: restrict to aipi's IP only, do not open the port to all hosts:

```bash
# On jolly — replace with aipi's actual IP
sudo ufw allow from <PRIMARY_NODE_IP> to any port 50052
sudo ufw reload
sudo ufw status
```

This rule was required on our Arch Linux jolly node. Without it, `rpc-server` starts successfully but aipi's connection attempt is silently dropped and llama-server reports the backend as unreachable.

If you need to diagnose a firewall issue:

```bash
# On aipi — test raw TCP connectivity to jolly
nc -zv <JOLLY_LOCAL_IP> 50052
```

A successful response is `Connection to <JOLLY_LOCAL_IP> 50052 port [tcp/*] succeeded!`. A timeout or `Connection refused` indicates either the service is not running or UFW is blocking it.

---

## 7. Performance on RPi 5

**Test configuration**: Mistral 24B Q4_K_M split across aipi + jolly, both Raspberry Pi 5 (8 GB), CPU-only inference, Gigabit Ethernet between nodes.

| Metric | Typical range |
|---|---|
| Time to first token (TTFT) — short prompt (<200 tokens) | 15–25 seconds |
| Time to first token — long prompt (>1000 tokens) | 40–60 seconds |
| Generation speed (tokens/sec) | 2–6 tok/s |
| Model load time (cold) | 45–90 seconds |

**Why DOTPROD matters**: The Cortex-A76's DOTPROD instruction processes 4x INT8 multiplications per cycle in the dot-product path. For Q4_K_M quantisation, which dequantises to INT8 for matrix multiply, this is the dominant operation. Without DOTPROD, throughput drops approximately 40%. Verify it is enabled before assuming your cluster is performing as expected.

**Network overhead**: Each inter-layer activation crossing the wire adds ~0.5–2 ms of latency depending on activation tensor size and network congestion. For a 40-layer model with 20 layers on jolly, this adds roughly 10–40 ms to every forward pass — significant relative to TTFT but invisible in generation speed (which is dominated by memory bandwidth, not network).

**Thermal throttling**: Pi 5 CPU cores clock down from 2.4 GHz to 1.5 GHz at ~80°C. Sustained inference runs at 100% CPU utilisation and will heat the board. Without active cooling, generation speed degrades by 20–40% after the first few minutes of heavy load. Fit heatsinks and a fan on both nodes.

---

## 8. Scaling to 3+ Nodes

Adding additional RPC workers is purely additive — append one `--rpc` flag per worker:

```bash
llama-server \
    --model /path/to/model.gguf \
    --rpc node1:50052 \
    --rpc node2:50052 \
    --rpc node3:50052 \
    --no-repack \
    --host 0.0.0.0 \
    --port 8080
```

llama.cpp queries each worker's free memory at startup and distributes layers proportionally across all of them, including the local node.

**Memory capacity by node count** (assuming 8 GB RPi 5 nodes, ~7.5 GB usable per node):

| Nodes | Usable memory | Models that fit |
|---|---|---|
| 2 | ~15 GB | Mistral 24B Q4_K_M (13.8 GB), Llama 3 8B Q8 (8.5 GB) |
| 3 | ~22.5 GB | Mistral 24B Q8_0 (22 GB), CodeLlama 34B Q4_K_M (19 GB) |
| 4 | ~30 GB | Llama 3 34B Q4_K_M (19 GB), Llama 70B Q2_K (26 GB) |

Note: quantisation format significantly affects size. Q2_K degrades quality noticeably for reasoning tasks; Q4_K_M is the recommended minimum for Mistral-class models.

Each additional node requires:
1. llama.cpp built with `-DGGML_RPC=ON`
2. `rpc-server` running and listening on port 50052
3. UFW (or iptables) allowing TCP 50052 from aipi's IP
4. Gigabit Ethernet — slower links become the bottleneck

---

## 9. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `Connection refused` on port 50052 | `rpc-server` is not running on the worker node | `sudo systemctl start llama-rpc.service` on jolly; verify with `systemctl status llama-rpc.service` |
| TCP connection times out | UFW or iptables blocking port 50052 | `sudo ufw allow from <AIPI_IP> to any port 50052` on jolly |
| `failed to allocate buffer of size 13946880000` at model load | Missing `--no-repack` flag | Add `--no-repack` to ExecStart in `llama-server.service` |
| Model load fails immediately after tensor read | Incomplete model download | Check file size: `ls -lh /path/to/model.gguf`. Re-download if size doesn't match expected. For Mistral 24B Q4_K_M, expect ~13.8 GB |
| Generation starts fast then slows after ~5 minutes | CPU thermal throttling | Check temperature: `vcgencmd measure_temp` on both nodes. Fit active cooling; target <70°C under sustained load |
| Mid-generation connection drop / truncated output | RPC TCP timeout | Network instability between aipi and jolly. Check cable quality, switch port, run `ping -i 0.01 jolly` for sustained packet loss |
| `RPC0` not shown in startup log | Worker not reachable at startup | Confirm jolly's `rpc-server` is listening: `ss -tlnp | grep 50052` on jolly. Confirm IP address with `ip addr` |
| Very slow TTFT (>90 seconds) on short prompts | Context too large, or DOTPROD not active | Confirm `DOTPROD = 1` in startup log; reduce `--ctx-size` if needed |
| `NEON = 0` or `DOTPROD = 0` in startup log | Cross-compiled binary or old compiler | Rebuild natively on the Pi: `ssh pi@aipi` then compile locally |

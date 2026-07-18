# Multi-Node Cluster Expansion

OffGridAI scales horizontally with no architectural changes — just add `--rpc` arguments. Each additional Raspberry Pi 5 8GB adds 7.8GB of usable model RAM.

## How RPC Scaling Works

llama.cpp distributes model layers proportional to each backend's available memory:

```
llama-server \
  --rpc <node1>:50052 \   # jolly:  7953 MiB
  --rpc <node2>:50052 \   # delta:  7953 MiB
  --rpc <node3>:50052 \   # echo:   7953 MiB
  ...
```

The coordinator (aipi) handles user connections, KV-cache, and final token sampling. Worker nodes compute tensor operations for their assigned layers. Communication is a binary TCP protocol at each layer boundary.

## Memory Tiers and Model Recommendations

| Nodes | Combined RAM | Recommended Model | File Size | Quality Jump |
|---|---|---|---|---|
| 2 (current) | ~16 GB | Mistral Small 3.1 24B Q4_K_M | 14.3 GB | Baseline |
| 3 | ~24 GB | Qwen2.5 32B Q4_K_M | ~19.4 GB | +15–20% on reasoning |
| 4 | ~32 GB | Llama 3.1 34B Q4_K_M | ~20 GB | Marginal over 32B |
| 5 | ~40 GB | Llama 3.3 70B Q3_K_M | ~37 GB | Major quality jump |
| 6 | ~48 GB | Llama 3.3 70B Q4_K_M | ~42 GB | Best quality at this tier |

> **Recommendation for first expansion:** A 3-node cluster running Qwen2.5 32B Q4_K_M gives a meaningful quality improvement over Mistral 24B for ~$100 in additional hardware. The 5→6 node jump to Llama 70B Q4_K_M is the next major milestone.

## Adding a Node

### Prerequisites
- Raspberry Pi 5 (8GB recommended, 4GB limits model tier significantly)
- ARM64-compatible OS (Arch Linux or Raspberry Pi OS both work — `setup-jolly.sh` handles both)
- Reachable from aipi on the LAN or via Tailscale
- `setup-jolly.sh` from the `/setup/` directory

### Step 1 — Run setup-jolly.sh on the new node

```bash
# From aipi, copy and run:
scp setup/setup-jolly.sh user@<NEW_NODE_IP>:~/
ssh user@<NEW_NODE_IP> "echo 'PASSWORD' | sudo -S bash ~/setup-jolly.sh"
```

This builds llama.cpp with `-DGGML_RPC=ON` and starts `rpc-server` on port 50052 as a systemd service.

### Step 2 — Open port 50052 on the new node

```bash
# On the new node (UFW):
sudo ufw allow from <AIPI_IP> to any port 50052
sudo ufw reload

# Verify from aipi:
nc -w2 <NEW_NODE_IP> 50052 && echo "OPEN" || echo "BLOCKED"
```

### Step 3 — Update llama-server.service on aipi

```bash
sudo nano /etc/systemd/system/llama-server.service
```

Append `--rpc <NEW_NODE_IP>:50052` to the ExecStart line:

```ini
ExecStart=/home/merry/llama.cpp/build/bin/llama-server \
    --model /home/merry/models/mistral-small-3.1-24b-instruct-Q4_K_M.gguf \
    --host 0.0.0.0 \
    --port 8080 \
    --ctx-size 4096 \
    --n-gpu-layers 0 \
    --no-repack \
    --rpc <JOLLY_LOCAL_IP>:50052 \
    --rpc <NEW_NODE_IP>:50052
```

```bash
sudo systemctl daemon-reload
sudo systemctl restart llama-server
```

### Step 4 — Verify the new node appears

```bash
journalctl -u llama-server -n 30 --no-pager | grep RPC
```

Expected output with 3 nodes:
```
I   - RPC0    : <JOLLY_LOCAL_IP>:50052 (7953 MiB, 7953 MiB free)
I   - RPC1    : <WORKER_2_IP>:50052 (7953 MiB, 7953 MiB free)
```

## Upgrading the Model After Adding a Node

Once you have 3 nodes (~24 GB combined):

```bash
# Download Qwen2.5 32B Q4_K_M
wget -c \
  --header "Authorization: Bearer $HF_TOKEN" \
  -O ~/models/qwen2.5-32b-instruct-Q4_K_M.gguf \
  "https://huggingface.co/bartowski/Qwen2.5-32B-Instruct-GGUF/resolve/main/Qwen2.5-32B-Instruct-Q4_K_M.gguf"

# Update llama-server.service model path
sudo nano /etc/systemd/system/llama-server.service
# Change: --model /home/merry/models/qwen2.5-32b-instruct-Q4_K_M.gguf

sudo systemctl daemon-reload && sudo systemctl restart llama-server

# Update the Ollama Modelfile if desired
nano ~/models/Modelfile.aipi-assistant
# Change: FROM /home/merry/models/qwen2.5-32b-instruct-Q4_K_M.gguf
ollama create aipi-assistant -f ~/models/Modelfile.aipi-assistant
```

## Network Considerations

Each additional RPC node adds ~20 network round-trips per generated token (one per transformer layer boundary crossing):

| Configuration | Network overhead per token | Notes |
|---|---|---|
| 2 nodes (1GbE) | ~20–40ms | ~20 layer crossings at ~1–2ms each |
| 3 nodes (1GbE) | ~30–60ms | Layers distributed across 3 nodes |
| 5+ nodes (1GbE) | ~50–100ms | Diminishing returns; 2.5GbE recommended |
| Any nodes (2.5GbE) | ~½ of above | Approximately halves network overhead |

**WiFi note:** Do not use WiFi for RPC worker nodes. Jitter spikes cause inference stalls and occasional disconnects mid-generation.

## Node Health Monitoring

```bash
# Quick health check for all nodes
for ip in <JOLLY_LOCAL_IP> <WORKER_2_IP> <WORKER_3_IP>; do
    nc -w2 "$ip" 50052 2>/dev/null && echo "✓ $ip:50052" || echo "✗ $ip:50052 UNREACHABLE"
done

# Show active RPC connections from aipi
ss -tnp | grep 50052

# Check rpc-server status on a remote node
ssh <username>@<JOLLY_LOCAL_IP> "systemctl is-active rpc-server"
```

## Heterogeneous Nodes

Nodes do not need to be identical. A mix of 8GB and 4GB boards works — llama.cpp distributes layers proportionally. However, **the slowest node limits throughput**: if one node is heavily loaded with other processes or has slower RAM, it creates a bottleneck. For best results, all worker nodes should be dedicated to RPC serving.

## Submitted Hardware Compatibility

See [.github/ISSUE_TEMPLATE/hardware_compatibility.md](../.github/ISSUE_TEMPLATE/hardware_compatibility.md) to report that OffGridAI works on non-RPi hardware. Community reports of compatibility with Rock 5B, Orange Pi 5 Plus, and other ARM64 boards with 8GB are welcome.

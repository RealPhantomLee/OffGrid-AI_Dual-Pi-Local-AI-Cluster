# Setup Scripts

## Which script runs where

| Script | Run on | Purpose |
|---|---|---|
| `setup-aipi.sh` | Primary node (Hailo-8 Pi) | Full stack: drivers, llama.cpp, Ollama, Docker, all services |
| `setup-jolly.sh` | Worker node (any Pi / Arch Linux compatible) | llama.cpp RPC worker only |

## Prerequisites

### Primary node (aipi)
- Raspberry Pi 5 (8 GB recommended)
- Hailo-8 AI HAT physically seated in PCIe slot
- Raspberry Pi OS (Debian 12 Bookworm or newer)
- Internet connection for initial package downloads
- HuggingFace account with access to `bartowski/mistralai_Mistral-Small-3.1-24B-Instruct-2503-GGUF`

### Worker node (jolly)
- Raspberry Pi 5 (8 GB recommended) or compatible ARM64 board
- Arch Linux ARM or Raspberry Pi OS
- Must be reachable from aipi on the local network (or via Tailscale)
- SSH access from aipi

## Running setup-aipi.sh

```bash
# 1. Set your HuggingFace token
export HF_TOKEN=hf_your_token_here

# 2. Run the setup script
sudo bash setup-aipi.sh
```

The script will:
1. Install system packages (hailo-all, cmake, ffmpeg, etc.)
2. Build llama.cpp with RPC support (~20 min on RPi 5)
3. Install and start Ollama, pull phi3.5 / llava-phi3 / qwen2.5-coder:7b
4. Create Python venv and install all Python dependencies
5. Install Docker and pull Open WebUI ARM64 image
6. Create all systemd service files
7. Enable and start all services

**After the script completes**, update `llama-server.service` with your worker node's IP:
```bash
sudo nano /etc/systemd/system/llama-server.service
# Replace <JOLLY_IP> with your worker node's actual IP
sudo systemctl daemon-reload && sudo systemctl restart llama-server
```

## Running setup-jolly.sh

Copy the script to your worker node and run it:

```bash
# From aipi:
scp setup-jolly.sh user@<JOLLY_IP>:~/
ssh user@<JOLLY_IP> "sudo bash ~/setup-jolly.sh"
```

The script will:
1. Install build dependencies via pacman (Arch) or apt (Debian)
2. Clone and build llama.cpp with RPC support
3. Create and start `llama-rpc-server.service` on port 50052

**Important:** Open port 50052 on your worker node's firewall for the primary node's IP:
```bash
# UFW (Ubuntu/Arch):
sudo ufw allow from <AIPI_IP> to any port 50052

# nftables:
sudo nft add rule ip filter input ip saddr <AIPI_IP> tcp dport 50052 accept
```

## Generating a WEBUI_SECRET_KEY

Before starting Open WebUI in production, generate a real secret key:

```bash
openssl rand -hex 32
```

Then update the service file:
```bash
sudo nano /etc/systemd/system/open-webui.service
# Replace <CHANGE_ME: openssl rand -hex 32> with your generated key
sudo systemctl daemon-reload && sudo systemctl restart open-webui
```

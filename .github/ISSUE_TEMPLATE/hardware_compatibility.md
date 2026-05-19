---
name: Hardware Compatibility Report
about: Report that OffGridAI works (or doesn't work) on your hardware
title: '[HARDWARE] '
labels: hardware-compatibility
assignees: ''
---

## Hardware

**Board name and revision:**

| Specification | Details |
|---|---|
| CPU | e.g. Cortex-A76 × 4 @ 2.4GHz |
| RAM | e.g. 8GB LPDDR4X |
| Storage | e.g. 32GB microSD / 256GB NVMe |
| AI accelerator | e.g. Hailo-8 AI HAT / none |
| Network | e.g. Gigabit Ethernet built-in |
| PCIe | e.g. PCIe 2.0 x1 via HAT+ |

## Operating System

| Field | Details |
|---|---|
| OS Name | e.g. Raspberry Pi OS / Arch Linux ARM / Ubuntu 24.04 |
| OS Version | e.g. Debian bookworm 12 |
| Kernel version | e.g. 6.12.x-rpi |
| Architecture | aarch64 / armv7l |

## Role Tested

- [ ] Primary node (running all services: llama-server, Ollama, Open WebUI, Agent Hub, etc.)
- [ ] Worker node (running rpc-server only)
- [ ] Both roles tested

## Changes Required from Default Setup

List any modifications to `setup-aipi.sh` or `setup-jolly.sh` needed for this hardware:

```bash
# Example: different package names, kernel flags, service configs
```

## Test Results

| Service | Status | Notes |
|---|---|---|
| setup script completes | ✅ / ❌ / ⚠️ | |
| llama-server + RPC | ✅ / ❌ / ⚠️ | |
| rpc-server (worker) | ✅ / ❌ / ⚠️ | |
| Ollama | ✅ / ❌ / ⚠️ | |
| Open WebUI | ✅ / ❌ / ⚠️ | |
| Agent Hub | ✅ / ❌ / ⚠️ | |
| Whisper STT | ✅ / ❌ / ⚠️ | |
| ChromaDB | ✅ / ❌ / ⚠️ | |
| Hailo-8 (if applicable) | ✅ / ❌ / ⚠️ / N/A | |

## Performance Notes

| Metric | Value |
|---|---|
| Model tested | e.g. Mistral Small 3.1 24B Q4_K_M |
| Tokens/sec (generation) | e.g. 3.2 t/s |
| Time to first token (TTFT) | e.g. 22 seconds |
| Model load time | e.g. 2.5 minutes |
| Peak CPU temperature under load | e.g. 74°C |
| Combined power draw (both nodes) | e.g. 23W |

## Additional Notes

Any other findings, quirks, or recommendations for this hardware combination.

---
name: Bug Report
about: Something isn't working as expected
title: '[BUG] '
labels: bug
assignees: ''
---

## Description
A clear and concise description of the bug.

## Steps to Reproduce
1. 
2. 
3. 

## Expected Behavior
What you expected to happen.

## Actual Behavior
What actually happened. Include the exact error message if any.

## Hardware

| Component | Details |
|---|---|
| Primary node | Raspberry Pi 5 8GB / 4GB / other: |
| Worker node | Raspberry Pi 5 8GB / 4GB / other: |
| AI accelerator | Hailo-8 AI HAT / none |
| Storage (primary) | SD card / NVMe / USB: |
| Storage (worker) | SD card / NVMe / USB: |

## Operating System

| Node | OS | Version / Kernel |
|---|---|---|
| Primary (aipi) | Raspberry Pi OS / Ubuntu / other: | |
| Worker (jolly) | Arch Linux ARM / Raspberry Pi OS / other: | |

## Node Role Affected
- [ ] Primary node (aipi) only
- [ ] Worker node (jolly) only
- [ ] Both nodes

## Service Affected
- [ ] llama-server (port 8080)
- [ ] Ollama (port 11434)
- [ ] Open WebUI (port 3000)
- [ ] Agent Hub (port 8000)
- [ ] Whisper STT (port 9000)
- [ ] ChromaDB (port 8001)
- [ ] Hailo-8 / TAPPAS
- [ ] setup-aipi.sh
- [ ] setup-jolly.sh
- [ ] rag-ingest (timer / service)

## Relevant Logs

```
# Paste output from:
# journalctl -u <service-name> -n 50 --no-pager
```

## Additional Context
Any other context, screenshots, or configuration details that might help diagnose the issue.

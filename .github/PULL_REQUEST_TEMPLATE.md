## Description

A clear description of what this PR changes and why.

## Type of Change

- [ ] Bug fix (non-breaking, fixes an issue)
- [ ] New feature (non-breaking, adds functionality)
- [ ] Breaking change (changes existing behavior)
- [ ] Documentation improvement
- [ ] Hardware compatibility report
- [ ] Security fix

## Testing Done

Describe how you tested this change:

- [ ] Tested on real hardware (details below)
- [ ] Setup scripts complete without errors
- [ ] All core services start and pass health checks:
  ```bash
  systemctl is-active llama-server ollama open-webui agent-hub whisper-server chromadb
  curl -s http://localhost:8000/status | python3 -m json.tool
  curl -s http://localhost:8080/v1/models
  ```
- [ ] RAG ingest runs cleanly: `sudo systemctl start rag-ingest.service`

**Hardware tested on:**
- Primary node (aipi): 
- Worker node (jolly): 
- OS versions: 

## Security Checklist

- [ ] No hardcoded IP addresses — use `<AIPI_IP>` / `<JOLLY_IP>` placeholders
- [ ] No credentials, tokens, passwords, or secrets in committed files
- [ ] No HuggingFace tokens — use `$HF_TOKEN` environment variable pattern
- [ ] `.env.example` updated if new environment variables were added
- [ ] No new external/cloud service dependencies introduced
- [ ] `WEBUI_SECRET_KEY` placeholder preserved (not a real key)

## Documentation

- [ ] Relevant docs in `/docs/` updated
- [ ] New services/endpoints documented
- [ ] `README.md` updated if user-facing changes were made

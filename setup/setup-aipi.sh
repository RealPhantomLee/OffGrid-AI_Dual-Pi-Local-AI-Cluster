#!/usr/bin/env bash
# setup-aipi.sh — One-time setup for aipi (Raspberry Pi 5 with Hailo-8)
# Run as: sudo bash setup-aipi.sh  (or as a user with sudo access)
set -e

# ---------------------------------------------------------------------------
# Colors
# ---------------------------------------------------------------------------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

section() {
    echo ""
    echo -e "${CYAN}${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${CYAN}${BOLD}  $1${NC}"
    echo -e "${CYAN}${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
}

ok()   { echo -e "${GREEN}  [OK]${NC} $1"; }
info() { echo -e "${YELLOW}  [..] $1${NC}"; }
err()  { echo -e "${RED}  [!!] $1${NC}"; }

MERRY_HOME="/home/merry"
LLAMA_DIR="${MERRY_HOME}/llama.cpp"
MODELS_DIR="${MERRY_HOME}/models"
AGENT_DIR="${MERRY_HOME}/agent-hub"

# ---------------------------------------------------------------------------
# Phase 1: System packages
# ---------------------------------------------------------------------------
section "Phase 1: System Packages"
info "Updating apt and installing dependencies..."
apt-get update -qq
apt-get install -y \
    hailo-all \
    cmake \
    libcurl4-openssl-dev \
    python3-pip \
    python3-venv \
    git \
    curl \
    build-essential \
    ffmpeg
ok "System packages installed."

# ---------------------------------------------------------------------------
# Phase 2: Build llama.cpp
# ---------------------------------------------------------------------------
section "Phase 2: Building llama.cpp (with RPC support)"

if [ ! -d "${LLAMA_DIR}" ]; then
    info "Cloning llama.cpp..."
    git clone https://github.com/ggerganov/llama.cpp.git "${LLAMA_DIR}"
fi

info "Running cmake configure..."
cmake -B "${LLAMA_DIR}/build" \
    -S "${LLAMA_DIR}" \
    -DGGML_RPC=ON \
    -DCMAKE_BUILD_TYPE=Release

info "Building llama.cpp (this will take a while on Pi 5)..."
cmake --build "${LLAMA_DIR}/build" -j4

ok "llama.cpp built successfully."
ls "${LLAMA_DIR}/build/bin/"

# ---------------------------------------------------------------------------
# Phase 3: Ollama
# ---------------------------------------------------------------------------
section "Phase 3: Installing Ollama and Pulling Models"
info "Installing Ollama..."
curl -fsSL https://ollama.ai/install.sh | sh
ok "Ollama installed."

info "Enabling ollama service..."
systemctl enable ollama
systemctl start ollama

# Give Ollama a moment to start
sleep 5

info "Pulling phi3.5..."
ollama pull phi3.5

info "Pulling llava-phi3..."
ollama pull llava-phi3

info "Pulling qwen2.5-coder:7b..."
ollama pull qwen2.5-coder:7b

ok "All Ollama models pulled."

# ---------------------------------------------------------------------------
# Phase 4: Python packages
# ---------------------------------------------------------------------------
section "Phase 4: Installing Python Packages"
info "Creating Python virtual environment at /home/merry/venv..."
python3 -m venv "${MERRY_HOME}/venv"
chown -R merry:merry "${MERRY_HOME}/venv"

info "Installing pip packages into venv..."
"${MERRY_HOME}/venv/bin/pip" install \
    fastapi \
    "uvicorn[standard]" \
    openai \
    chromadb \
    sentence-transformers \
    httpx \
    wikipedia \
    arxiv \
    chromadb \
    "sentence-transformers" \
    "openai-whisper" \
    huggingface_hub
# Note: open-webui has no ARM64 pip wheel — installed via Docker below (Phase 4b)

ok "Python packages installed into ${MERRY_HOME}/venv."

# ---------------------------------------------------------------------------
# Phase 4b: Docker + Open WebUI (ARM64 image — no pip wheel available)
# ---------------------------------------------------------------------------
section "Phase 4b: Installing Docker and Open WebUI"
if ! command -v docker &>/dev/null; then
    info "Installing Docker..."
    curl -fsSL https://get.docker.com | sh
    usermod -aG docker merry
    systemctl enable --now docker
    ok "Docker installed."
else
    ok "Docker already installed: $(docker --version)"
fi

info "Pulling Open WebUI ARM64 image..."
mkdir -p "${MERRY_HOME}/open-webui-data"
chown merry:merry "${MERRY_HOME}/open-webui-data"
docker pull ghcr.io/open-webui/open-webui:main
ok "Open WebUI image pulled."

# ---------------------------------------------------------------------------
# Phase 5: Systemd service files
# ---------------------------------------------------------------------------
section "Phase 5: Creating Systemd Service Files"

# --- llama-server -----------------------------------------------------------
# NOTE: Replace <JOLLY_IP> below with jolly's actual Tailscale IPv4
#       once jolly has been added to the Tailscale network. Run:
#         tailscale ip -4   (on jolly)
#       then update the --rpc argument here and reload:
#         sudo systemctl daemon-reload && sudo systemctl restart llama-server
info "Creating llama-server.service..."
cat > /etc/systemd/system/llama-server.service << 'EOF'
[Unit]
Description=llama.cpp HTTP server (mistral-small-3.1-22b with RPC offload to jolly)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=merry
WorkingDirectory=/home/merry
ExecStart=/home/merry/llama.cpp/build/bin/llama-server \
    --model /home/merry/models/mistral-small-3.1-24b-instruct-Q4_K_M.gguf \
    --host 0.0.0.0 \
    --port 8080 \
    --ctx-size 4096 \
    --n-gpu-layers 0 \
    --rpc <JOLLY_IP>:50052
Restart=on-failure
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF
ok "llama-server.service created (edit <JOLLY_IP> once jolly is on Tailscale)."

# --- open-webui -------------------------------------------------------------
info "Creating open-webui.service..."
cat > /etc/systemd/system/open-webui.service << 'EOF'
[Unit]
Description=Open WebUI (Docker, ARM64)
After=docker.service network-online.target
Requires=docker.service

[Service]
Type=simple
User=merry
ExecStartPre=-/usr/bin/docker rm -f open-webui
ExecStart=/usr/bin/docker run --rm --name open-webui \
    -p 3000:8080 \
    -e OLLAMA_BASE_URL=http://host-gateway:11434 \
    -e OPENAI_API_BASE_URL=http://host-gateway:8080/v1 \
    -e OPENAI_API_KEY=none \
    -e WEBUI_SECRET_KEY=changeme-replace-with-random-string \
    --add-host=host-gateway:host-gateway \
    -v /home/merry/open-webui-data:/app/backend/data \
    ghcr.io/open-webui/open-webui:main
ExecStop=/usr/bin/docker stop open-webui
Restart=on-failure
RestartSec=15
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF
ok "open-webui.service created (runs via Docker)."

# --- agent-hub --------------------------------------------------------------
info "Creating agent-hub.service..."
mkdir -p "${AGENT_DIR}"
# Create a minimal placeholder main.py if it doesn't exist
if [ ! -f "${AGENT_DIR}/main.py" ]; then
    cat > "${AGENT_DIR}/main.py" << 'PYEOF'
from fastapi import FastAPI

app = FastAPI(title="Agent Hub")

@app.get("/health")
def health():
    return {"status": "ok"}
PYEOF
fi

cat > /etc/systemd/system/agent-hub.service << 'EOF'
[Unit]
Description=Agent Hub (FastAPI orchestrator)
After=network-online.target ollama.service
Wants=network-online.target

[Service]
Type=simple
User=merry
WorkingDirectory=/home/merry/agent-hub
Environment="VIRTUAL_ENV=/home/merry/venv"
Environment="PATH=/home/merry/venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
ExecStart=/home/merry/venv/bin/uvicorn main:app --host 0.0.0.0 --port 8000 --workers 2
Restart=on-failure
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF
ok "agent-hub.service created."

# --- whisper-server ---------------------------------------------------------
info "Creating whisper-server.service..."
# openai-whisper already installed in venv (Phase 4)

cat > /etc/systemd/system/whisper-server.service << 'EOF'
[Unit]
Description=Whisper transcription server (faster-whisper)
After=network-online.target

[Service]
Type=simple
User=merry
WorkingDirectory=/home/merry
ExecStart=/home/merry/venv/bin/python3 -c "
import uvicorn, tempfile, os, whisper
from fastapi import FastAPI, UploadFile, File

app = FastAPI(title='Whisper Server')
model = whisper.load_model('base')

@app.post('/transcribe')
async def transcribe(audio: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as f:
        f.write(await audio.read())
        tmp_path = f.name
    try:
        result = model.transcribe(tmp_path)
        return {'text': result['text'], 'language': result.get('language', 'en')}
    finally:
        os.unlink(tmp_path)

uvicorn.run(app, host='0.0.0.0', port=9000)
"
Restart=on-failure
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF
ok "whisper-server.service created."

# --- rag-ingest timer -------------------------------------------------------
info "Creating rag-ingest.service and rag-ingest.timer..."
chmod +x "${MERRY_HOME}/scripts/rag-ingest.sh"
cp "${MERRY_HOME}/scripts/rag-ingest.service" /etc/systemd/system/rag-ingest.service
cp "${MERRY_HOME}/scripts/rag-ingest.timer"   /etc/systemd/system/rag-ingest.timer
ok "rag-ingest timer created (runs daily at 2:00 AM)."

# ---------------------------------------------------------------------------
# Phase 6: Enable and start all services
# ---------------------------------------------------------------------------
section "Phase 6: Enabling and Starting Services"

systemctl daemon-reload

SERVICES=(ollama llama-server open-webui agent-hub whisper-server)

for svc in "${SERVICES[@]}"; do
    info "Enabling ${svc}..."
    systemctl enable "${svc}" || err "Failed to enable ${svc}"
done

for svc in "${SERVICES[@]}"; do
    info "Starting ${svc}..."
    systemctl start "${svc}" || err "Failed to start ${svc} (check: journalctl -u ${svc})"
done

# Enable the timer (don't start — it fires on schedule)
info "Enabling rag-ingest.timer..."
systemctl enable rag-ingest.timer
systemctl start rag-ingest.timer
ok "RAG ingestion timer active — next run at 2:00 AM."

# Short wait for services to initialise
sleep 3

ok "All services enabled and started."

# ---------------------------------------------------------------------------
# Phase 7: Status and next steps
# ---------------------------------------------------------------------------
section "Phase 7: Final Status"

echo ""
for svc in "${SERVICES[@]}"; do
    status=$(systemctl is-active "${svc}" 2>/dev/null || echo "unknown")
    if [ "${status}" = "active" ]; then
        echo -e "  ${GREEN}${BOLD}[RUNNING]${NC}  ${svc}"
    else
        echo -e "  ${RED}${BOLD}[${status^^}]${NC}   ${svc}"
    fi
done

echo ""
echo -e "${BOLD}Next steps:${NC}"
echo "  1. Register a Modelfile:   ollama create aipi-assistant -f ${MODELS_DIR}/Modelfile.aipi-assistant"
echo "  2. Set up jolly:           copy setup-jolly.sh to jolly and run it"
echo "  3. Update llama-server:    edit /etc/systemd/system/llama-server.service"
echo "                             replace <JOLLY_IP> with jolly's Tailscale IP"
echo "                             then: sudo systemctl daemon-reload && sudo systemctl restart llama-server"
echo "  4. Open WebUI:             http://$(hostname -I | awk '{print $1}'):3000"
echo "  5. llama-server API:       http://$(hostname -I | awk '{print $1}'):8080"
echo "  6. Agent Hub API:          http://$(hostname -I | awk '{print $1}'):8000"
echo "  7. Whisper Server:         http://$(hostname -I | awk '{print $1}'):9000"
echo "  8. RAG timer status:       systemctl list-timers rag-ingest.timer"
echo "     Run ingest now:         sudo systemctl start rag-ingest.service"
echo ""
echo -e "${GREEN}${BOLD}Setup complete!${NC}"

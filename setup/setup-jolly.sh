#!/usr/bin/env bash
# setup-jolly.sh — Run on jolly to set it up as an llama.cpp RPC worker node.
# Copy this file to jolly and run: sudo bash setup-jolly.sh
set -e

# ---------------------------------------------------------------------------
# Colors
# ---------------------------------------------------------------------------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

section() {
    echo ""
    echo -e "${CYAN}${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${CYAN}${BOLD}  $1${NC}"
    echo -e "${CYAN}${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
}

ok()   { echo -e "${GREEN}  [OK]${NC} $1"; }
info() { echo -e "${YELLOW}  [..] $1${NC}"; }
err()  { echo -e "${RED}  [!!] $1${NC}"; }

JOLLY_USER="${SUDO_USER:-$(whoami)}"
JOLLY_HOME="/home/${JOLLY_USER}"
LLAMA_DIR="${JOLLY_HOME}/llama.cpp"

# ---------------------------------------------------------------------------
# Phase 1: System packages
# ---------------------------------------------------------------------------
section "Phase 1: System Packages"
info "Installing dependencies via pacman (Arch Linux)..."
pacman -Sy --noconfirm base-devel cmake curl git
ok "System packages installed."

# ---------------------------------------------------------------------------
# Phase 2: Build llama.cpp with RPC support
# ---------------------------------------------------------------------------
section "Phase 2: Building llama.cpp (RPC server mode)"

if [ ! -d "${LLAMA_DIR}" ]; then
    info "Cloning llama.cpp into ${LLAMA_DIR}..."
    git clone https://github.com/ggerganov/llama.cpp.git "${LLAMA_DIR}"
    chown -R "${JOLLY_USER}:${JOLLY_USER}" "${LLAMA_DIR}"
fi

info "Configuring cmake with RPC support..."
cmake -B "${LLAMA_DIR}/build" \
    -S "${LLAMA_DIR}" \
    -DGGML_RPC=ON \
    -DCMAKE_BUILD_TYPE=Release

info "Building (using all available cores)..."
cmake --build "${LLAMA_DIR}/build" -j"$(nproc)"

ok "llama.cpp built. Available binaries:"
ls "${LLAMA_DIR}/build/bin/"

# ---------------------------------------------------------------------------
# Phase 3: Systemd service for llama-rpc-server
# ---------------------------------------------------------------------------
section "Phase 3: Creating llama-rpc-server Systemd Service"

info "Writing /etc/systemd/system/llama-rpc-server.service..."
cat > /etc/systemd/system/llama-rpc-server.service << EOF
[Unit]
Description=llama.cpp RPC worker node (jolly)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=${JOLLY_USER}
WorkingDirectory=${JOLLY_HOME}
ExecStart=${LLAMA_DIR}/build/bin/llama-rpc-server \
    --host 0.0.0.0 \
    --port 50052
Restart=on-failure
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF
ok "llama-rpc-server.service created."

# ---------------------------------------------------------------------------
# Phase 4: Enable and start the service
# ---------------------------------------------------------------------------
section "Phase 4: Enabling and Starting Service"

systemctl daemon-reload
systemctl enable llama-rpc-server
systemctl start llama-rpc-server

sleep 2

status=$(systemctl is-active llama-rpc-server 2>/dev/null || echo "unknown")
if [ "${status}" = "active" ]; then
    ok "llama-rpc-server is running on port 50052."
else
    err "llama-rpc-server status: ${status}. Check logs with: journalctl -u llama-rpc-server -n 50"
fi

# ---------------------------------------------------------------------------
# Final: Print Tailscale IP
# ---------------------------------------------------------------------------
section "Done — Tailscale IP for aipi configuration"

echo ""
if command -v tailscale &>/dev/null; then
    TAILSCALE_IP="$(tailscale ip -4 2>/dev/null || echo 'unavailable')"
    if [ "${TAILSCALE_IP}" != "unavailable" ]; then
        echo -e "${GREEN}${BOLD}  Jolly's Tailscale IPv4: ${TAILSCALE_IP}${NC}"
    else
        err "tailscale returned no IP — is Tailscale connected? Run: sudo tailscale up"
        TAILSCALE_IP="<not yet available>"
    fi
else
    err "Tailscale is not installed on this machine."
    info "Install it with: curl -fsSL https://tailscale.com/install.sh | sh"
    TAILSCALE_IP="<install tailscale first>"
fi

echo ""
echo -e "${BOLD}Next step — on aipi, update the llama-server service:${NC}"
echo "  sudo nano /etc/systemd/system/llama-server.service"
echo "  Change:  --rpc JOLLY_TAILSCALE_IP:50052"
echo "  To:      --rpc ${TAILSCALE_IP}:50052"
echo ""
echo "  Then reload:"
echo "    sudo systemctl daemon-reload && sudo systemctl restart llama-server"
echo ""
echo -e "${GREEN}${BOLD}jolly setup complete!${NC}"

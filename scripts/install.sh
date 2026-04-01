#!/usr/bin/env bash
# Bonsai Fund — One-time installer
set -e

SKILL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BONDIR="$SKILL_DIR/bonsai_fund"

echo "=== Bonsai Fund Installer ==="
echo "Skill directory: $SKILL_DIR"

# Create data directory
DATA_DIR="$HOME/.hermes/bonsai_fund_data"
mkdir -p "$DATA_DIR"/{logs,data}
echo "Data directory: $DATA_DIR"

# Create symlink so bonsai_fund package resolves correctly
# (already in ~/.hermes/skills/bonsai-fund/bonsai_fund/)

# Check Python version
PYTHON=$(command -v python3)
echo "Python: $PYTHON"
$PYTHON --version

# Install Python dependencies
echo ""
echo "=== Installing Python dependencies ==="
$PYTHON -m pip install pyyaml cryptography 2>/dev/null || true

# Download model (optional — requires ~1.2 GB)
if [ -t 0 ]; then
    echo ""
    read -p "Download Bonsai-8B model now? (~1.2 GB) [y/N]: " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        $PYTHON "$BONDIR/launch.py" download-model
    fi
fi

# Check for llama-server
if ! command -v llama-server &>/dev/null && [ ! -f "$HOME/bin/llama-server" ]; then
    echo ""
    echo "llama-server not found in PATH."
    echo "Options:"
    echo "  1. Install PrismML llama.cpp: bash bonsai_fund/scripts/install_llama.sh"
    echo "  2. Download pre-built: python3 bonsai_fund/launch.py install-llama"
    echo "  3. Use existing llama-server in PATH"
fi

# Create config.yaml from template
if [ ! -f "$SKILL_DIR/config.yaml" ]; then
    cat > "$SKILL_DIR/config.yaml" << 'EOF'
bonsai_fund:
  data_dir: ~/.hermes/bonsai_fund_data
  bankroll: 50.00
  paper_mode: true

  model_path: ~/.local/share/llama.cpp/bonsai/bonsai-8b-v1-q1_0.gguf
  model_name: Bonsai-8B.gguf
  port_base: 8090
  instances: 7

  # Risk
  max_position_pct: 0.05
  min_edge: 0.03
  min_confidence: 0.55
  min_vote_margin: 2
  max_drawdown_pct: 0.25

  # Scheduling
  scan_interval_min: 30
  board_report_utc_hour: 8
  board_report_utc_day: 6    # 0=Mon, 6=Sun

  # Notifications
  telegram_bot_token: ""
  telegram_chat_id: "YOUR_CHAT_ID"
  alerts_enabled: false
EOF
    echo "Created config.yaml — edit $SKILL_DIR/config.yaml to configure"
fi

echo ""
echo "=== Installation complete ==="
echo ""
echo "Next steps:"
echo "  1. python3 bonsai_fund/launch.py install-llama   # install llama-server"
echo "  2. python3 bonsai_fund/launch.py download-model   # download Bonsai-8B (~1.2 GB)"
echo "  3. python3 bonsai_fund/launch.py start --count 7  # launch 7 instances"
echo "  4. python3 bonsai_fund/hedge_fund.py status       # verify everything works"
echo ""
echo "Hermes integration:"
echo "  hermes skills install bonsai-fund"
echo "  /bonsai scan"
echo "  /bonsai status"

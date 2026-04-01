"""
Bonsai Fund — Standalone Configuration
All defaults + YAML overrides. No bonsai_fund imports.
"""

from __future__ import annotations
import os, yaml
from pathlib import Path

HERMES_HOME = Path(os.environ.get("HERMES_HOME", Path.home() / ".hermes"))
SKILL_DIR   = HERMES_HOME / "skills" / "bonsai-fund"
DATA_DIR    = HERMES_HOME / "bonsai_fund_data"

DATA_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
_DEFAULTS = {
    # Paths
    "model_path":       str(Path.home() / ".local" / "share" / "llama.cpp" / "bonsai" / "bonsai-8b-v1-q1_0.gguf"),
    "llama_server_bin": str(Path.home() / "bin" / "llama-server"),
    # Trading
    "bankroll":          50.0,
    "paper_mode":        True,
    "max_position_pct":  0.05,
    "min_edge":          0.03,
    "min_confidence":    0.55,
    "min_vote_margin":   2,
    "max_drawdown_pct":  0.25,
    # Bonsai
    "model_name":        "Bonsai-8B.gguf",
    "port_base":         8090,
    "instances":         7,
    # Scheduling
    "scan_interval_min":  30,
    "board_report_day":  6,     # Sunday
    "board_report_hour": 8,     # UTC
    # Notifications
    "telegram_chat_id":  "6953738253",
    "telegram_bot_token_env": "MVE_TELEGRAM_BOT_TOKEN",
    "alerts_enabled":    False,
}

# Load YAML overrides
_CFG = {}
_cfg_file = SKILL_DIR / "config.yaml"
if _cfg_file.exists():
    try:
        raw = yaml.safe_load(open(_cfg_file)) or {}
        _CFG = raw.get("bonsai_fund", {})
    except Exception:
        pass

def get(key: str, default=None):
    return _CFG.get(key, _DEFAULTS.get(key, default))

# ---------------------------------------------------------------------------
# Expose as module-level attributes
# ---------------------------------------------------------------------------
BANKROLL          = get("bankroll")
PAPER_MODE        = get("paper_mode", True)
MAX_POSITION_PCT  = get("max_position_pct", 0.05)
MIN_EDGE          = get("min_edge", 0.03)
MIN_CONFIDENCE    = get("min_confidence", 0.55)
MIN_VOTE_MARGIN   = get("min_vote_margin", 2)
MAX_DRAWDOWN_PCT  = get("max_drawdown_pct", 0.25)
MODEL_NAME        = get("model_name", "Bonsai-8B.gguf")
MODEL_PATH        = Path(get("model_path"))
LLAMA_SERVER_BIN  = get("llama_server_bin")
PORT_BASE         = get("port_base", 8090)
INSTANCES         = get("instances", 7)
SCAN_INTERVAL_MIN = get("scan_interval_min", 30)
BOARD_REPORT_DAY  = get("board_report_day", 6)
BOARD_REPORT_HOUR = get("board_report_hour", 8)
TELEGRAM_CHAT_ID  = get("telegram_chat_id", "6953738253")
TELEGRAM_TOKEN_ENV= get("telegram_bot_token_env", "MVE_TELEGRAM_BOT_TOKEN")
ALERTS_ENABLED    = get("alerts_enabled", False)

# Derived
DB_PATH     = DATA_DIR / "bonsai_portfolio.db"
VOTES_DB_PATH = DATA_DIR / "bonsai_votes.db"
STATE_FILE  = DATA_DIR / "bonsai_scheduler_state.json"
REPORT_PATH = DATA_DIR / "bonsai_board_report_latest.json"
LOG_DIR     = DATA_DIR / "logs"
LOG_FILE    = DATA_DIR / "bonsai.log"

"""
Bonsai Fund — Hermes Cron Integration Examples

Use these as templates for /cron add commands in Hermes.
Run via: hermes cron add "Bonsai Scan" ...
"""

SCAN_CRON = """
hermes cron add "Bonsai Market Scan" \
  --schedule "every 30m" \
  --deliver local \
  --skill bonsai-fund \
  --prompt "Run: cd ~/.hermes/skills/bonsai-fund && python3 bonsai_fund/scheduler.py once"
"""

BOARD_REPORT_CRON = """
hermes cron add "Bonsai Board Report" \
  --schedule "0 8 * * 0" \
  --deliver telegram \
  --skill bonsai-fund \
  --prompt "Run: cd ~/.hermes/skills/bonsai-fund && python3 bonsai_fund/scheduler.py board-report"
"""

ALERT_CRON = """
# Only if you want instant Telegram alerts on every signal (not just Board Reports)
hermes cron add "Bonsai Signal Alert" \
  --schedule "every 60m" \
  --deliver telegram \
  --skill bonsai-fund \
  --prompt "Edit bonsai_fund/config.yaml to set alerts_enabled: true, then run: cd ~/.hermes/skills/bonsai-fund && python3 bonsai_fund/scheduler.py once"
"""

# Direct Hermes slash commands (no cron needed)
HERMES_COMMANDS = """
After installing via: hermes skills install bonsai-fund

/bonsai scan          # Run a full market scan now
/bonsai status        # Show portfolio + risk status
/bonsai board-report  # Send Board Report to Telegram now
/bonsai circuit-break # Reset circuit breaker after review
/bonsai reset        # Liquidate all positions (start fresh)
/bonsai vote KXNBA-26 # Vote on a specific market
"""


if __name__ == "__main__":
    print("=== Bonsai Fund — Hermes Cron Integration ===\n")
    print("SCAN CRON:")
    print(SCAN_CRON.strip())
    print("\nBOARD REPORT CRON:")
    print(BOARD_REPORT_CRON.strip())
    print("\nDIRECT HERMES COMMANDS (after install):")
    print(HERMES_COMMANDS.strip())

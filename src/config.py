import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Environment / runtime config
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")

HEADLESS = os.getenv("HEADLESS", "true").lower() in {"1", "true", "yes"}
BROWSER_TIMEOUT_MS = int(os.getenv("BROWSER_TIMEOUT_MS", "30000"))

OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "outputs"))
LOG_DIR = OUTPUT_DIR / "logs"
SCREENSHOT_DIR = OUTPUT_DIR / "screenshots"
REPORT_DIR = OUTPUT_DIR / "reports"


def ensure_dirs():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)


def require_api_key():
    if not ANTHROPIC_API_KEY:
        raise RuntimeError(
            "Missing ANTHROPIC_API_KEY. Put it in .env (NOT committed) or environment variables."
        )

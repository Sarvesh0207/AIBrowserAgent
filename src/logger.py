import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from .config import LOG_DIR, ensure_dirs


def utc_ts() -> str:
    return datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")


def jsonl_path(prefix: str = "run") -> Path:
    ensure_dirs()
    return LOG_DIR / f"{prefix}_{utc_ts()}.jsonl"


def append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

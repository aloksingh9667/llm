"""Local .env loader (safe secrets handling).

Reads KEY=VALUE lines from a local `.env` (gitignored) into os.environ
WITHOUT overwriting already-set variables. Committed code must NEVER
contain secret values — see `.env.example` for the template.
"""
import os
from pathlib import Path


def load_dotenv(path: str = ".env") -> dict[str, str]:
    """Load .env if present. Returns {key: 'set'|'skipped-existing'|'missing-file'}."""
    p = Path(path)
    if not p.exists():
        return {"_status": "missing-file"}
    status: dict[str, str] = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip("'\"")
        if not key or not value:
            continue
        if key in os.environ:
            status[key] = "skipped-existing"
        else:
            os.environ[key] = value
            status[key] = "set"
    return status

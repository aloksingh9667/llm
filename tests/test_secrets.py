"""Safe-secrets gates: .env loader behavior (values never committed)."""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from myai.utils.env import load_dotenv


def test_dotenv_sets_missing_and_keeps_existing(tmp_path, monkeypatch):
    f = tmp_path / ".env"
    f.write_text("AAA_TEST_KEY=hello\nBBB_TEST_KEY=\n# comment\n", encoding="utf-8")
    monkeypatch.setenv("BBB_TEST_KEY", "keepme")
    st = load_dotenv(str(f))
    assert os.environ["AAA_TEST_KEY"] == "hello"
    assert os.environ["BBB_TEST_KEY"] == "keepme"  # empty .env value never overrides
    assert st["AAA_TEST_KEY"] == "set" and "BBB_TEST_KEY" not in st
    del os.environ["AAA_TEST_KEY"]


def test_dotenv_missing_file():
    assert load_dotenv("/nonexistent-xyz/.env") == {"_status": "missing-file"}


def test_no_secret_values_in_repo():
    """Guardrail: .env.example must not contain real secret values."""
    text = Path(".env.example").read_text(encoding="utf-8")
    for line in text.splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            assert line.endswith("="), f"value committed in .env.example: {line[:20]}"

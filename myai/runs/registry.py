"""Neon Postgres registry for MyAI runs/checkpoints (audit section 23).

Metadata ONLY in Postgres (run_id, step, tokens, hashes, URIs, status);
large tensors stay in file/object storage. Connection string is read
from the local `.env` (NEON_DATABASE_URL) — never committed, never logged.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SCHEMA = """
CREATE TABLE IF NOT EXISTS myai_runs (
    run_id        TEXT PRIMARY KEY,
    model_name    TEXT NOT NULL,
    git_commit    TEXT,
    dataset_hash  TEXT,
    tokenizer_hash TEXT,
    config        JSONB,
    status        TEXT NOT NULL DEFAULT 'created',
    created_utc   TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_utc   TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS myai_checkpoints (
    id            BIGSERIAL PRIMARY KEY,
    run_id        TEXT NOT NULL REFERENCES myai_runs(run_id),
    step          INTEGER NOT NULL,
    tokens_seen   BIGINT NOT NULL,
    train_loss    DOUBLE PRECISION,
    val_loss      DOUBLE PRECISION,
    checkpoint_uri TEXT NOT NULL,
    created_utc   TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (run_id, step)
);
"""


def _load_env(path: str = ".env") -> None:
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line and "DATABASE_URL" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


def connect():
    import psycopg2

    _load_env()
    url = os.environ.get("NEON_DATABASE_URL")
    if not url:
        raise SystemExit("NEON_DATABASE_URL missing — check local .env (never commit it)")
    return psycopg2.connect(url)


def setup() -> None:
    conn = connect()
    try:
        with conn, conn.cursor() as cur:
            cur.execute(SCHEMA)
        print("neon: myai_runs + myai_checkpoints ready")
    finally:
        conn.close()


def log_run(run_id: str, model_name: str, git_commit=None, dataset_hash=None,
            tokenizer_hash=None, config=None, status="created") -> None:
    import json

    conn = connect()
    try:
        with conn, conn.cursor() as cur:
            cur.execute(
                """INSERT INTO myai_runs (run_id, model_name, git_commit, dataset_hash,
                                          tokenizer_hash, config, status, updated_utc)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,now())
                   ON CONFLICT (run_id) DO UPDATE SET status=EXCLUDED.status,
                       updated_utc=now()""",
                (run_id, model_name, git_commit, dataset_hash, tokenizer_hash,
                 json.dumps(config or {}), status),
            )
    finally:
        conn.close()


def log_checkpoint(run_id: str, step: int, tokens_seen: int, checkpoint_uri: str,
                   train_loss=None, val_loss=None) -> None:
    conn = connect()
    try:
        with conn, conn.cursor() as cur:
            cur.execute(
                """INSERT INTO myai_checkpoints (run_id, step, tokens_seen, train_loss,
                                                 val_loss, checkpoint_uri)
                   VALUES (%s,%s,%s,%s,%s,%s)
                   ON CONFLICT (run_id, step) DO UPDATE SET tokens_seen=EXCLUDED.tokens_seen,
                       train_loss=EXCLUDED.train_loss, val_loss=EXCLUDED.val_loss,
                       checkpoint_uri=EXCLUDED.checkpoint_uri""",
                (run_id, step, tokens_seen, train_loss, val_loss, checkpoint_uri),
            )
            cur.execute("UPDATE myai_runs SET status='training', updated_utc=now() WHERE run_id=%s",
                        (run_id,))
    finally:
        conn.close()


if __name__ == "__main__":
    setup()

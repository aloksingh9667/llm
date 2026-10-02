# MyAI — From-Scratch Foundation LLM (base scaffolding, Step 1-2)

CPU-first base. No pretrained weights. Random init only.

## Quickstart (Docker, recommended)
```powershell
docker compose build
docker compose run --rm myai-cpu python scripts/check_env.py
docker compose run --rm myai-cpu python scripts/smoke_test.py
```

## Quickstart (local venv)
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements/base.txt
python scripts/check_env.py
python scripts/smoke_test.py
```

## Layout
- `myai/` model + pipeline packages (placeholders until Steps 3-9)
- `configs/` 20M/100M + mixture + manifest example
- `docker/Dockerfile.cpu` + `docker-compose.yml`
- `scripts/check_env.py`, `scripts/smoke_test.py`

Next: Steps 3-9 tokenizer + RMSNorm + RoPE + attention + SwiGLU + block + model.

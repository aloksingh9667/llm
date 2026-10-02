# MyAI on Kaggle GPU — runbook (30h/week budget)

## Setup (2 min)
1. Kaggle → New Notebook → Settings → **Accelerator: GPU** (T4/P100 free).
2. Run cells top→bottom in `MyAI_Kaggle_Training.ipynb`.
3. Everything durable goes to `/kaggle/working` (checkpoints, reports).
   Session time limits (~9-12h) apply — checkpoints let you resume.

## Resume after interrupt
```bash
python scripts/train.py --model-config configs/myai-100m.yaml \
  --data-dir data/processed/fw10m-32k --max-steps 5000 \
  --batch-size 4 --accum 8 --amp \
  --resume /kaggle/working/ckpts-100m/step-0000500.pt \
  --out-dir /kaggle/working/ckpts-100m --report /kaggle/working/train-100m.json
```

## Budget math (measure, don't guess)
- Cell 6 prints `tok/s`. Extrapolate: `hours = total_tokens / tok/s / 3600`.
- Rough guide (T4, MyAI-100M, seq 512, eff.batch 16k tok/step):
  ~80M tokens ≈ 4–8 GPU-hours. 10M-token corpus × 8 epochs ≈ 80M.
- Week plan: wk1 smoke+20M → wk2 100M run → wk3 eval+SFT (`myai/training/sft.py`).

## Notes
- Data: FineWeb ODC-By-1.0, streamed (no full download), manifest per run.
- Tokenizer 32k trains on GPU box (cell 4) — replaces the CPU seed vocab.
- SFT after base: build pairs with `myai.training.sft.build_sft_pair`
  (prompt masked `-100`), then reuse `scripts/train.py` on packed SFT data.

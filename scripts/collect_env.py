"""Snapshot run reproducibility bundle (audit section 39).

Usage:
    python scripts/collect_env.py --label myai100m-001 --config configs/myai-100m.yaml \\
        --dataset-manifest data/processed/fw10m-32k/manifest.json --out runs/

Writes runs/<utc>_<label>/{environment.json,config.yaml,dataset.json}
so any run can be traced to code + data + machine.
"""
import argparse
import datetime
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default="run")
    parser.add_argument("--config", default=None)
    parser.add_argument("--dataset-manifest", default=None)
    parser.add_argument("--out", default="runs")
    args = parser.parse_args()

    import torch

    env = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "collected_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    try:
        import subprocess

        env["git_commit"] = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10
        ).stdout.strip() or None
    except Exception:
        env["git_commit"] = None

    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d-%H%M%S")
    d = Path(args.out) / f"{stamp}_{args.label}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "environment.json").write_text(json.dumps(env, indent=2) + "\n", encoding="utf-8")
    if args.config and Path(args.config).exists():
        shutil.copy(args.config, d / "config.yaml")
    if args.dataset_manifest and Path(args.dataset_manifest).exists():
        shutil.copy(args.dataset_manifest, d / "dataset.json")
    print(f"run bundle -> {d}")


if __name__ == "__main__":
    main()

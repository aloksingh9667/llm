"""Kaggle-native leg continuation — runs INSIDE a leg kernel.

After a leg finishes training, this renders the next leg's kernel dir via
the same template the local relay uses (scripts/run_legs.build_leg_dir)
and `kaggle kernels push`es it, keeping the GPU chain alive with no local
terminal open. Credentials come from a kaggle.json mounted out of a
private Kaggle dataset (or KAGGLE_KEY env as a fallback).

Called by the leg notebook's final cell: `python scripts/push_next_leg.py --leg N`
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.run_legs import build_leg_dir, stop_for, TARGET_STEPS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--leg", type=int, required=True, help="this kernel's leg number")
    parser.add_argument("--owner", default="aloksingh9182")
    parser.add_argument("--kernel-prefix", default="myai-100m-train-leg")
    parser.add_argument("--creds-dataset", default=None,
                        help="slug of the private dataset carrying kaggle.json")
    args = parser.parse_args()

    nxt = args.leg + 1
    if stop_for(nxt) > TARGET_STEPS:
        print(f"chain complete: target {TARGET_STEPS:,} steps reached after leg {args.leg}")
        return
    if not os.environ.get("KAGGLE_KEY") and not (Path.home() / ".kaggle" / "kaggle.json").exists():
        raise SystemExit("no kaggle credentials in env or ~/.kaggle — chain halts here; "
                         "attach the creds dataset or the KAGGLE_KEY secret")

    here = f"{args.owner}/{args.kernel_prefix}{args.leg}"
    d, info = build_leg_dir(nxt, here, "shared 92.2M-token pack")
    meta_path = d / "kernel-metadata.json"
    if args.creds_dataset:
        import json

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        src = meta.get("dataset_sources", [])
        if args.creds_dataset not in src:
            src.append(args.creds_dataset)
        meta["dataset_sources"] = src
        meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        print(f"creds dataset attached: {args.creds_dataset}")

    print(f"pushing {here} -> next leg {nxt} (stop {stop_for(nxt):,})", flush=True)
    p = subprocess.run([sys.executable, "-m", "kaggle", "kernels", "push", "-p", str(d)],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    out = (p.stdout + p.stderr).strip()
    print(out[-600:])
    if p.returncode != 0:
        raise SystemExit(f"next-leg push failed (returncode {p.returncode}) — chain halts; "
                         "re-run with scripts/run_legs.py --start-leg N after fixing")


if __name__ == "__main__":
    main()

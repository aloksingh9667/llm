"""Kaggle-native leg continuation — runs INSIDE a leg kernel.

After a leg finishes training, this renders the next leg's kernel dir via
the same template the local relay uses (scripts/run_legs.build_leg_dir)
and `kaggle kernels push`es it, keeping the GPU chain alive with no local
terminal open. Credentials come from a kaggle.json mounted out of a
private Kaggle dataset (or KAGGLE_KEY env as a fallback).

Called by the leg notebook's final cell: `python scripts/push_next_leg.py --leg N`
"""
import argparse
import glob
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.run_legs import build_leg_dir, stop_for, TARGET_STEPS


def find_refresh_token() -> str | None:
    """Locate refresh_token.txt possibly mounted from the private creds dataset."""
    if not os.path.isdir("/kaggle/input"):
        return None
    hits = glob.glob("/kaggle/input/**/refresh_token.txt", recursive=True)
    return hits[0] if hits else None


def mint_access_token(refresh_token: str) -> str:
    """Mint a fresh 12h access token via kagglesdk (kaggle>=2.2). Falls back
    to partial errors as SystemExit so the relay/local chain can alert."""
    from datetime import timedelta

    from kaggle.api.kaggle_api_extended import KaggleApi
    from kagglesdk import KaggleCredentials

    api = KaggleApi()
    client = KaggleApi.build_kaggle_client_with_params(
        args=[], response_processor=api.get_response_processor())
    creds = KaggleCredentials(client=client, refresh_token=refresh_token.strip())
    resp = creds.generate_access_token(timedelta(hours=12))
    token = resp.token if resp else None
    if not token:
        raise SystemExit("refresh-token access mint failed — chain halts here")
    return token


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--leg", type=int, required=True, help="this kernel's leg number")
    parser.add_argument("--owner", default="aloksingh9182")
    parser.add_argument("--kernel-prefix", default="myai-100m-train-leg")
    parser.add_argument("--creds-dataset", default=None,
                        help="slug of the private dataset carrying kaggle.json / refresh token")
    args = parser.parse_args()

    nxt = args.leg + 1
    if stop_for(nxt) > TARGET_STEPS:
        print(f"chain complete: target {TARGET_STEPS:,} steps reached after leg {args.leg}")
        return

    # Credentials resolution, most-preferred first:
    #   1. refresh_token.txt (mounted from private creds dataset) -> mint fresh 12h token
    #   2. kaggle.json / KAGGLE_KEY classic auth (already wired by the notebook cell)
    rt_file = find_refresh_token()
    if rt_file and not os.environ.get("KAGGLE_API_TOKEN"):
        os.environ["KAGGLE_API_TOKEN"] = mint_access_token(
            Path(rt_file).read_text(encoding="utf-8"))
        print(f"minted fresh 12h access token from {rt_file}")

    if not (os.environ.get("KAGGLE_API_TOKEN") or os.environ.get("KAGGLE_KEY")
            or (Path.home() / ".kaggle" / "kaggle.json").exists()):
        raise SystemExit("no kaggle credentials in env or ~/.kaggle — chain halts here; "
                         "attach the creds dataset (refresh_token.txt) "
                         "or the KAGGLE_KEY secret")

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

"""Rescue path when the previous leg's checkpoint is NOT mounted.

Background: Kaggle-side (kernel) pushes once produced leg kernels with
`kernel_sources: []` (the mount was silently lost), so the train cell died
with NO_CKPT. This fetches the previous leg's `final.pt` directly through the
kernels-output API (token minted from the private creds dataset), so legs
never hard-depend on the mount again.

Runs INSIDE a leg kernel (invoked by the train cell when no mounted final.pt
is found). Prints the final.pt path on stdout; exits 1 with the CLI error on
failure.
"""
import argparse
import glob
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.push_next_leg import find_refresh_token, mint_access_token
from scripts.run_legs import KERNEL_PREFIX, OWNER


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--leg", type=int, required=True, help="this kernel's leg number")
    parser.add_argument("--owner", default=OWNER)
    parser.add_argument("--kernel-prefix", default=KERNEL_PREFIX)
    parser.add_argument("--dest", default="/kaggle/tmp/prev-ckpt")
    args = parser.parse_args()

    prev_ref = f"{args.owner}/{args.kernel_prefix}{args.leg - 1}"

    if not os.environ.get("KAGGLE_API_TOKEN"):
        rt = find_refresh_token()
        if not rt:
            raise SystemExit("fetch_prev_ckpt: refresh_token.txt not mounted and no "
                             "KAGGLE_API_TOKEN — cannot fetch via API")
        os.environ["KAGGLE_API_TOKEN"] = mint_access_token(Path(rt).read_text(encoding="utf-8"))

    Path(args.dest).mkdir(parents=True, exist_ok=True)
    print(f"fetch_prev_ckpt: downloading latest output of {prev_ref} -> {args.dest}",
          flush=True)
    p = subprocess.run([sys.executable, "-m", "kaggle", "kernels", "output", prev_ref,
                        "-p", args.dest],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    if p.returncode != 0:
        print(((p.stdout or "") + (p.stderr or ""))[-600:], file=sys.stderr, flush=True)
        raise SystemExit(f"fetch_prev_ckpt: kernels output failed for {prev_ref}")

    hits = glob.glob(str(Path(args.dest) / "**" / "final.pt"), recursive=True)
    if not hits:
        # fall back to the newest checkpoint file (mid-leg crash resume)
        cands = sorted(glob.glob(str(Path(args.dest) / "**" / "*.pt"), recursive=True),
                       key=os.path.getmtime)
        if cands:
            hits = [cands[-1]]
    if not hits:
        raise SystemExit(f"fetch_prev_ckpt: no checkpoint files in {args.dest}")
    print(hits[0])


if __name__ == "__main__":
    main()

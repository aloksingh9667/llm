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
import time
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

    dest = Path(args.dest)
    # clean slate (stale files from a previous attempt must not mask failure)
    if dest.exists():
        for f in dest.iterdir():
            if f.is_file():
                f.unlink()
    dest.mkdir(parents=True, exist_ok=True)

    if not os.environ.get("KAGGLE_API_TOKEN"):
        rt = find_refresh_token()
        if not rt:
            raise SystemExit("fetch_prev_ckpt: refresh_token.txt not mounted and no "
                             "KAGGLE_API_TOKEN — cannot fetch via API")
        os.environ["KAGGLE_API_TOKEN"] = mint_access_token(Path(rt).read_text(encoding="utf-8"))

    # NOTE: ALL informational output goes to STDERR — stdout must carry ONLY
    # the final path (the caller captures stdout as the CKPT value).
    print(f"fetch_prev_ckpt: downloading latest output of {prev_ref} -> {dest}",
          file=sys.stderr, flush=True)

    last_err = ""
    for attempt, pattern in enumerate(("*final.pt", None)):
        cmd = [sys.executable, "-m", "kaggle", "kernels", "output", prev_ref,
               "-p", str(dest), "-o"]
        if pattern:
            cmd += ["--file-pattern", pattern]
        p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                           errors="replace", cwd=dest)
        if p.returncode != 0:
            last_err = ((p.stdout or "") + (p.stderr or ""))[-600:]
        else:
            # search dest hard, then the CLI's cwd (older CLIs dump there), then working
            for root in (dest, dest.parent, Path.cwd(), Path("/kaggle/working")):
                hits = glob.glob(str(root / "**" / "final.pt"), recursive=True)
                if hits:
                    print(hits[0])
                    return
        print(f"fetch_prev_ckpt: attempt {attempt + 1} found no files yet — retrying "
              f"{'(full download)' if pattern is None else '(filtered)'}",
              file=sys.stderr, flush=True)
        time.sleep(20)

    if last_err:
        print(last_err, file=sys.stderr, flush=True)
    raise SystemExit(f"fetch_prev_ckpt: no checkpoint files found for {prev_ref}")


if __name__ == "__main__":
    main()

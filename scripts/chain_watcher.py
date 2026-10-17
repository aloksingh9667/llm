"""Kaggle-native chain watcher — runs INSIDE a scheduled CPU kernel.

The chain problem it solves: when a train leg sits QUEUED with no GPU worker,
nothing in the chain is executing, so no chain kernel can self-heal. This
watcher is *scheduled by Kaggle itself* (e.g. hourly), so Kaggle's own
scheduler wakes it up every interval to repair the chain.

What it does per run (no local terminal involved):
  1. Mint a fresh 12h access token from the private creds dataset
  2. Scan legs 1..20 via kernels status API; find the first non-COMPLETE leg
  3. Repair:
       - QUEUED            -> version-bump re-push now; sleep ~55 min; if still
                                QUEUED, bump once more (max 2 bumps per run)
       - ERROR / CANCELLED -> re-push the same leg (deterministic resume)
       - MISSING next leg while previous leg COMPLETE -> render + push it
                                (self-heal of a failed kernel-side self-push)
       - all COMPLETE      -> chain done, exit quietly
  4. RUNNING legs are left alone (a bump would only waste GPU)

Neon logging is intentionally absent (no DB creds inside kernels); the local
relay backfills Neon when a session is alive.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.run_legs import (DATASET, KERNEL_PREFIX, OWNER, TARGET_STEPS,
                              build_leg_dir, stop_for)
from scripts.push_next_leg import find_refresh_token, mint_access_token

LAST_LEG = 20  # stop_for(20) == 122_000 == TARGET_STEPS
SLEEP_BEFORE_SECOND_BUMP = 55 * 60
MAX_BUMPS_PER_RUN = 2
DATA_NOTE = "shared 92.2M-token pack"


def sh(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(args, capture_output=True, text=True,
                          encoding="utf-8", errors="replace")


def leg_status(leg: int) -> str:
    """'QUEUED' | 'RUNNING' | 'COMPLETE' | 'ERROR' | 'CANCELLED' | 'MISSING'."""
    p = sh([sys.executable, "-m", "kaggle", "kernels", "status",
            f"{OWNER}/{KERNEL_PREFIX}{leg}"])
    out = (p.stdout or "") + (p.stderr or "")
    for s in ("COMPLETE", "RUNNING", "QUEUED", "CANCEL_ACKNOWLEDGED",
              "CANCEL_INITIATED", "ERROR"):
        if f"KernelWorkerStatus.{s}" in out:
            return "CANCELLED" if s.startswith("CANCEL") else s
    return "MISSING" if p.returncode != 0 else "UNKNOWN"


def bump(leg: int) -> bool:
    """Re-push leg N (version bump). Returns True on success."""
    resume_ref = (f"{OWNER}/{KERNEL_PREFIX}{leg - 1}"
                  if leg > 1 else f"{OWNER}/myai-100m-base")
    d, info = build_leg_dir(leg, resume_ref, DATA_NOTE)
    p = sh([sys.executable, "-m", "kaggle", "kernels", "push", "-p", str(d)])
    ok = p.returncode == 0
    out = (p.stdout or "") + (p.stderr or "")
    print(out[-300:], flush=True)
    return ok


def scan() -> dict:
    states = {log: leg_status(log) for log in range(1, LAST_LEG + 1)}
    if states[1] == "MISSING":  # network/auth failure, not a real gap
        raise SystemExit("watcher: cannot reach kernels API — aborting this run")
    return states


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bumps-per-run", type=int, default=MAX_BUMPS_PER_RUN)
    args = parser.parse_args()

    rt = find_refresh_token()
    if not rt:
        raise SystemExit("watcher: refresh_token.txt not mounted — attach "
                         "aloksingh9182/myai-kaggle-creds to this kernel first")
    os_token = mint_access_token(Path(rt).read_text(encoding="utf-8"))
    import os

    os.environ["KAGGLE_API_TOKEN"] = os_token
    print("watcher: minted fresh access token", flush=True)

    states = scan()
    print("watcher: " + " ".join(f"{log}={s}" for log, s in states.items()), flush=True)

    done_legs = [log for log, s in states.items() if s == "COMPLETE"]
    active = None
    for log in sorted(states):
        if states[log] != "COMPLETE":
            active = log
            break

    if active is None:
        print(f"watcher: all {LAST_LEG} legs COMPLETE — 2B-token target reached "
              f"({TARGET_STEPS:,} steps). Chain done; no further action.", flush=True)
        return

    s = states[active]
    print(f"watcher: active leg {active} (stop {stop_for(active):,}), status {s}", flush=True)

    bumps = 0
    if s in ("ERROR", "CANCELLED"):
        print(f"watcher: leg {active} failed ({s}) — re-pushing same leg", flush=True)
        bumps += 1
        bump(active)
    elif s == "QUEUED":
        print(f"watcher: leg {active} pinned QUEUED — version-bump re-push", flush=True)
        bumps += 1
        bump(active)
        print(f"watcher: sleeping {SLEEP_BEFORE_SECOND_BUMP // 60} min before re-check "
              "(queued may just need a worker to free up)", flush=True)
        time.sleep(SLEEP_BEFORE_SECOND_BUMP)
        s2 = leg_status(active)
        print(f"watcher: leg {active} now {s2}", flush=True)
        if s2 == "QUEUED" and bumps < args.bumps_per_run:
            print("watcher: still QUEUED — second version-bump re-push", flush=True)
            bumps += 1
            bump(active)
    elif s == "RUNNING":
        print(f"watcher: leg {active} is RUNNING normally — nothing to do", flush=True)
        return
    elif s == "MISSING" and active > 1 and states.get(active - 1) == "COMPLETE":
        # previous leg finished but its self-push failed: create the missing leg
        print(f"watcher: leg {active} MISSING while leg {active - 1} COMPLETE — "
              "creating it now (self-heal of failed chain push)", flush=True)
        bumps += 1
        bump(active)
    else:
        print(f"watcher: leg {active} status {s} — no repair pattern applies", flush=True)
        return

    after = leg_status(active)
    print(f"watcher: leg {active} after repair -> {after} "
          f"({bumps} bump(s) this run)", flush=True)


if __name__ == "__main__":
    main()

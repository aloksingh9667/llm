"""Leg relay: run myai-100m pretraining in parts until --target-tokens (default 2B).

One logical cosine run split across Kaggle GPU kernels. Each leg:
  resume previous leg output -> train until stop-step -> complete.
The relay polls status, pushes the next leg on completion, retries on
quota exhaustion (weekly refresh), and logs every leg to Neon.

State: reports/leg-relay.json. Resume legitimately with --start-leg N.

Usage (local, needs .env for Neon + kaggle CLI creds):
    python scripts/run_legs.py --start-leg 2
"""
import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STATE_PATH = ROOT / "reports" / "leg-relay.json"
DATASET = "aloksingh9182/myai-fwab-32k"
CREDS_DATASET = "aloksingh9182/myai-kaggle-creds"  # private: kaggle.json only
OWNER = "aloksingh9182"
KERNEL_PREFIX = "myai-100m-train-leg"
MOUNT = "/kaggle/input/myai-fwab-32k"
TARGET_STEPS = 122_000   # 122_000 steps x 16_384 tok/step = 2.0B tokens
START_STEP = 5_000       # K3 base ended here
PER_LEG = 6_000          # ~2.9h GPU at K3's measured speed
POLL_S = 300
RETRY_SLEEP_S = 3 * 3600


def stop_for(leg: int) -> int:
    return min(START_STEP + PER_LEG * leg, TARGET_STEPS)


def leg_kernel_ref(leg: int) -> str:
    return f"{OWNER}/{KERNEL_PREFIX}{leg}"


def run_kaggle(*args: str, cwd: Path | None = None) -> str:
    import os

    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    p = subprocess.run([sys.executable, "-m", "kaggle", *args],
                       capture_output=True, text=True, encoding="utf-8",
                       errors="replace", env=env, cwd=str(cwd or ROOT), timeout=600)
    out = p.stdout + p.stderr
    if p.returncode != 0:
        raise RuntimeError(f"kaggle {' '.join(args)} failed:\n{out[-1000:]}")
    return out


def status(ref: str, retries: int = 5, delay: int = 60) -> str:
    """Status with retry-on-transient-network (DNS/timeouts must not kill the relay)."""
    for attempt in range(retries):
        try:
            out = run_kaggle("kernels", "status", ref)
            m = re.search(r'"KernelWorkerStatus\.(\w+)"', out)
            return m.group(1) if m else "UNKNOWN"
        except RuntimeError as e:
            msg = str(e)[-300:]
            transient = any(k in msg for k in ("NameResolutionError", "ConnectionError",
                                               "Max retries exceeded", "timed out"))
            if attempt < retries - 1 and transient:
                print(f"[net] status retry {attempt + 1}/{retries} for {ref}: {msg[:120]}",
                      flush=True)
                time.sleep(delay)
            else:
                raise


def log_state(update: dict) -> None:
    st = json.loads(STATE_PATH.read_text(encoding="utf-8")) if STATE_PATH.exists() else {}
    st.update(update)
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(st, indent=2) + "\n", encoding="utf-8")


def neon_log(run_id: str, status_s: str, config: dict | None = None) -> None:
    try:
        sys.path.insert(0, str(ROOT))
        from myai.runs.registry import log_run

        log_run(run_id, "myai-100m", git_commit=_git_commit() or "relay", tokenizer_hash="f9728d0c89c5",
                config=config or {}, status=status_s)
        print(f"[neon] {run_id} -> {status_s}", flush=True)
    except Exception as e:  # registry must never kill the relay
        print(f"[neon:{type(e).__name__}] {run_id} {status_s} (non-fatal)", flush=True)


def _git_commit():
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
            cwd=str(ROOT)).stdout.strip()
    except Exception:
        return None


def build_leg_dir(leg: int, resume_ref: str, data_note: str) -> tuple[Path, dict]:
    """Materialize kaggle/kernel-k6-legN (metadata + notebook) for this leg."""
    stop = stop_for(leg)
    out_d = ROOT / "kaggle" / f"kernel-k6-leg{leg}"
    out_d.mkdir(parents=True, exist_ok=True)

    nb = {
        "cells": [
            {"cell_type": "markdown", "metadata": {}, "source": [
                f"# MyAI 100M Train — Leg {leg} (GPU)\n",
                f"Resume from {resume_ref} (current checkpoint). One logical cosine run "
                f"to {TARGET_STEPS:,} steps (~2B tokens on the {data_note}); this leg "
                f"stops at step {stop:,}. Exact resume: RNG/optimizer/scaler/data_pos. "
                "On completion this kernel pushes the next leg itself (no terminal needed)."]},
            {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [
                "!git clone https://github.com/aloksingh9667/llm.git myai 2>&1 | tail -1\n",
                "%cd /kaggle/working/myai\n",
                "!pip install -q -r requirements/train.txt 2>&1 | tail -1\n"]},
            {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [
                "!ls /kaggle/input\n",
                "!find /kaggle/input -name 'final.pt' 2>/dev/null\n"]},
            {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [
                "%%bash\n",
                "cd /kaggle/working/myai\n",
                "CKPT=$(find /kaggle/input -name 'final.pt' 2>/dev/null | head -1)\n",
                'test -n "${CKPT}" || { echo NO_CKPT; exit 1; }\n',
                "DATA=$(dirname \"$(find /kaggle/input -name 'train.pt' 2>/dev/null | head -1)\")\n",
                'test -f "${DATA}/manifest.json" || { echo NO_DATA; exit 1; }\n',
                'echo "resume-from: ${CKPT}  data-dir: ${DATA}"\n',
                "python scripts/train.py --model-config configs/myai-100m.yaml \\\n",
                '    --data-dir "${DATA}" \\\n',
                f"    --max-steps {TARGET_STEPS} --stop-step {stop} \\\n",
                "    --batch-size 4 --accum 8 --amp --no-shuffle \\\n",
                "    --eval-every 500 --ckpt-every 2500 \\\n",
                '    --resume "${CKPT}" \\\n',
                f"    --out-dir /kaggle/working/ckpts-leg{leg} \\\n",
                f"    --report /kaggle/working/train-leg{leg}.json\n"]},
            {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [
                f"!test -f /kaggle/working/ckpts-leg{leg}/final.pt && echo LEG{leg}_OK\n",
                f"!python -c \"import json; r=json.load(open('/kaggle/working/train-leg{leg}.json')); "
                "[print(k, r.get(k)) for k in ('steps','first_loss','final_train_loss',"
                "'val_loss','tokens_seen')]\"\n",
                "import os, glob, shutil, pathlib\n",
                "cands = glob.glob('/kaggle/input/**/kaggle.json', recursive=True)\n",
                "if cands:\n",
                "    kx = pathlib.Path.home() / '.kaggle'\n",
                "    kx.mkdir(exist_ok=True)\n",
                "    shutil.copy(cands[0], kx / 'kaggle.json')\n",
                "    os.chmod(kx / 'kaggle.json', 0o600)\n",
                "    print('kaggle.json mounted:', cands[0])\n",
                "try:  # secret fallback for kaggle creds\n",
                "    from kaggle_secrets import UserSecretsClient\n",
                "    os.environ['KAGGLE_KEY'] = UserSecretsClient().get_secret('KAGGLE_KEY')\n",
                "    os.environ.setdefault('KAGGLE_USERNAME', 'aloksingh9182')\n",
                "    print('KAGGLE_KEY secret attached')\n",
                "except Exception as e:\n",
                "    print(f'no KAGGLE_KEY secret ({type(e).__name__}) — relying on kaggle.json mount')\n",
                f"!python scripts/push_next_leg.py --leg {leg} --creds-dataset {CREDS_DATASET}\n"]},
        ],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                     "name": "python3"},
                      "language_info": {"name": "python", "version": "3.10.0"}},
        "nbformat": 4, "nbformat_minor": 4,
    }
    (out_d / f"MyAI_100M_Leg{leg}.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8")

    meta = {
        "id": f"{OWNER}/{KERNEL_PREFIX}{leg}",
        "title": f"MyAI 100M Train Leg{leg}",
        "code_file": f"MyAI_100M_Leg{leg}.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "dataset_sources": [DATASET, CREDS_DATASET],
        "kernel_sources": [resume_ref] if resume_ref else [],
        "competition_sources": [],
        "model_sources": [],
    }
    (out_d / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return out_d, {"leg": leg, "stop_step": stop, "meta": meta}


QUEUED_BUMP_S = 45 * 60      # leg pinned QUEUED longer than this -> version-bump re-push
STUCK_RUNNING_S = 12 * 3600  # leg RUNNING way past a normal leg (~3.2h) -> bump re-push
MAX_BUMPS_PER_LEG = 8        # cap watchdog interventions per leg


def push_with_retry(leg_dir: Path, tag: str, retry_sleep: int, suffix: str = "") -> None:
    """kernels push with the relay's standard quota/network retry policy."""
    attempt = 0
    while True:
        attempt += 1
        try:
            run_kaggle("kernels", "push", "-p", str(leg_dir))
            print(f"[{tag}] pushed{suffix}", flush=True)
            return
        except Exception as e:
            msg = str(e)[-400:]
            print(f"[{tag}] push failed: {msg}", flush=True)
            if "quota" in msg.lower() or "403" in msg or "429" in msg:
                print(f"[{tag}] quota exhausted — sleeping {retry_sleep // 3600}h, "
                      "retrying (relay does not stop)", flush=True)
            time.sleep(retry_sleep if attempt > 1 else 60)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-leg", type=int, required=True)
    parser.add_argument("--poll", type=int, default=POLL_S)
    parser.add_argument("--retry-sleep", type=int, default=RETRY_SLEEP_S)
    args = parser.parse_args()

    data_note = "shared 92.2M-token pack"
    log_state({"config": {"target_steps": TARGET_STEPS, "per_leg": PER_LEG,
                          "start_step": START_STEP, "total_tokens": TARGET_STEPS * 16384}})

    prev_ref = None
    leg = args.start_leg
    bump_count: dict = {}  # cumulative watchdog interventions per leg (survive re-push cycles)
    if leg > 1:
        prev_ref = leg_kernel_ref(leg - 1)
        print(f"relay: waiting for {prev_ref} (leg {leg - 1}) to complete first", flush=True)
        while True:
            s = status(prev_ref)
            print(f"  leg{leg - 1}: {s}", flush=True)
            if s == "COMPLETE":
                break
            if s in ("ERROR", "CANCEL_ACKNOWLEDGED", "CANCEL_INITIATED"):
                raise SystemExit(f"leg {leg - 1} ended in {s} — inspect logs before resuming")
            time.sleep(args.poll)

    while stop_for(leg) <= TARGET_STEPS and leg <= 100:
        stop = stop_for(leg)
        resume_ref = prev_ref if prev_ref else (leg_kernel_ref(leg - 1) if leg > 1 else (
            f"{OWNER}/myai-100m-base"))
        # leg1 of the 2B plan resumes K3 base; later legs resume the prior leg
        if leg == 1:
            resume_ref = f"{OWNER}/myai-100m-base"

        eff_resume = prev_ref or (f"{OWNER}/myai-100m-base" if leg == 1 else None)
        leg_dir, info = build_leg_dir(leg, eff_resume, data_note)
        run_id = f"k6-train-leg{leg}"
        push_with_retry(leg_dir, f"leg{leg}", args.retry_sleep, suffix=f" (stop {stop:,})")
        log_state({f"leg{leg}": {"status": "RUNNING", "stop_step": stop}})
        neon_log(run_id, "training", config={"resume": eff_resume, "stop_step": stop,
                                             "max_steps": TARGET_STEPS, "dataset": DATASET})

        # ---- watch the leg: complete / fail / queue-starvation watchdog ----
        queued_since = None
        running_since = time.time()
        while True:
            held = bump_count.get(leg, 0)
            s = status(leg_kernel_ref(leg))
            now = time.time()
            if s == "COMPLETE":
                print(f"[leg{leg}] COMPLETE", flush=True)
                break
            if s in ("ERROR", "CANCEL_ACKNOWLEDGED", "CANCEL_INITIATED"):
                if held >= MAX_BUMPS_PER_LEG:
                    raise SystemExit(f"leg {leg} ended in {s} after {held} total bumps "
                                     "— inspect logs before resuming")
                bump_count[leg] = held + 1
                print(f"[leg{leg}] {s} — re-pushing same leg "
                      f"(deterministic resume, bump {held + 1}/{MAX_BUMPS_PER_LEG})", flush=True)
                neon_log(run_id, f"retry-after-{s}", config={"bump": held + 1})
                break  # outer loop re-pushes this leg
            if s == "QUEUED":
                if queued_since is None:
                    queued_since = now
                elif now - queued_since > QUEUED_BUMP_S and held < MAX_BUMPS_PER_LEG:
                    bump_count[leg] = held + 1
                    mins = int((now - queued_since) / 60)
                    print(f"[leg{leg}] QUEUED {mins}min — version-bump re-push "
                          f"{held + 1}/{MAX_BUMPS_PER_LEG}", flush=True)
                    neon_log(run_id, f"queued-bump-{held + 1}",
                             config={"queued_minutes": mins, "stop_step": stop})
                    push_with_retry(leg_dir, f"leg{leg}", args.retry_sleep,
                                    suffix=f" (queue-bump {held + 1})")
                    log_state({f"leg{leg}": {"status": "QUEUED-BUMPED",
                                             "stop_step": stop, "bumps": held + 1}})
                    queued_since = time.time()
                    running_since = time.time()  # bumped version restarts both watches
            else:  # RUNNING or other transient — reset queue timer, check stuck-run
                queued_since = None
                if s == "RUNNING" and held < MAX_BUMPS_PER_LEG and now - running_since > STUCK_RUNNING_S:
                    bump_count[leg] = held + 1
                    hrs = int((now - running_since) / 3600)
                    print(f"[leg{leg}] RUNNING {hrs}h (far past a normal leg) — "
                          f"version-bump re-push {held + 1}/{MAX_BUMPS_PER_LEG} "
                          "(check UI: an old stuck version may need manual cancel)", flush=True)
                    neon_log(run_id, f"stuck-running-bump-{held + 1}", config={"hours": hrs})
                    push_with_retry(leg_dir, f"leg{leg}", args.retry_sleep,
                                    suffix=f" (stuck-bump {held + 1})")
                    running_since = time.time()
            time.sleep(args.poll)

        if status(leg_kernel_ref(leg)) != "COMPLETE":
            continue  # re-push loop for the same leg
        log_state({f"leg{leg}": {"status": "COMPLETE", "stop_step": stop}})
        neon_log(run_id, "complete", config={"stop_step": stop})
        prev_ref = leg_kernel_ref(leg)
        leg += 1

    print(f"relay: step {stop_for(leg - 1):,} / {TARGET_STEPS:,} reached "
          f"({stop_for(leg - 1) * 16384 / 1e9:.2f}B tokens) — training objective met", flush=True)
    neon_log("k6-scale-2b", "complete-or-interrupted",
             config={"last_completed_leg": leg - 1, "final_step": stop_for(leg - 1)})


if __name__ == "__main__":
    main()

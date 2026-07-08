#!/usr/bin/env python3
"""Pre-run checks for an eval; exits non-zero if the run would be silently degraded.

Guards the three ways a run has quietly produced garbage for us, and records the git commit
so a result maps to an exact config:
  1. a missing task image     -> that task scores an automatic 0
  2. a corrupted deps prefix  -> every rollout import-crashes in ~5s with no error
  3. an unreachable endpoint  -> mass empty responses

Usage: preflight.py --dataset data/x.jsonl --formatter '/path/sifs/{instance_id}.sif'
                    [--endpoint URL --model NAME --skip-canary]
"""
import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--formatter", required=True, help="sif path template, e.g. /sifs/{instance_id}.sif")
    p.add_argument("--endpoint", default="https://inference-api.nvidia.com/v1")
    p.add_argument("--model", default="nvidia/nvidia/nemotron-3-ultra")
    p.add_argument("--skip-canary", action="store_true")
    return p.parse_args()


def check_images(dataset, formatter):
    """Every task's .sif must exist and be non-trivially sized (a stub/partial file scores 0)."""
    ids = [json.loads(l)["responses_create_params"]["metadata"]["instance_id"] for l in open(dataset)]
    missing = [i for i in ids
               if not (os.path.isfile(formatter.format(instance_id=i))
                       and os.path.getsize(formatter.format(instance_id=i)) > 1_000_000)]
    print(f"[images] {len(ids) - len(missing)}/{len(ids)} present")
    for i in missing:
        print(f"  MISSING: {i}")
    return not missing


def check_deps_prefix(dataset):
    """Import the hot path with the prefix's own python; a half-replaced package would otherwise
    crash every in-container rollout silently."""
    prefix = os.path.abspath(os.path.join(os.path.dirname(dataset), "..", "anyswe_openclaw_agent_deps"))
    python = os.path.join(prefix, "bin", "python")
    if not os.path.exists(python):
        print("[deps-prefix] not built yet (will build at env start) -- OK")
        return True
    probe = subprocess.run(
        [python, "-c", "import anyio.to_thread, openai, httpx; print('deps imports OK')"],
        capture_output=True, text=True)
    print(f"[deps-prefix] {probe.stdout.strip() or probe.stderr.strip().splitlines()[-1]}")
    return probe.returncode == 0


def check_endpoint(endpoint, model):
    """A one-shot canary request; failing (or no key) means the endpoint isn't usable."""
    key = os.environ.get("NVIDIA_INTERNAL_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not key:
        print("[canary] SKIP: no key in env (source .env first)")
        return False
    body = {"model": model, "messages": [{"role": "user", "content": "Say OK."}], "max_tokens": 500}
    request = urllib.request.Request(
        endpoint.rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"})
    try:
        started = time.time()
        with urllib.request.urlopen(request, timeout=90) as resp:
            json.load(resp)
        print(f"[canary] endpoint OK ({time.time() - started:.1f}s)")
        return True
    except Exception as e:
        print(f"[canary] ENDPOINT FAILED: {e}")
        return False


def print_provenance():
    """Record the commit so a result is traceable; warn if the tree is dirty."""
    def git(*args):
        return subprocess.run(["git", *args], capture_output=True, text=True).stdout.strip()
    sha = git("rev-parse", "--short", "HEAD")
    dirty = git("status", "--porcelain", "-uno")
    print(f"[provenance] git {sha}{' DIRTY(tracked)' if dirty else ''}")
    if dirty:
        print("  WARNING: tracked files modified -- result won't map to a commit hash")


def main():
    args = parse_args()
    ok = check_images(args.dataset, args.formatter)
    ok = check_deps_prefix(args.dataset) and ok        # runs regardless of the image result
    if not args.skip_canary:
        ok = check_endpoint(args.endpoint, args.model) and ok
    print_provenance()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

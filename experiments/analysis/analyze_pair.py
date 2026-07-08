#!/usr/bin/env python3
"""Paired comparison of two eval runs over the SAME tasks -- the only sound way to compare two
configs given the ~5-10pp run-to-run noise. Prints the overall resolve rate, a per-repo split,
the McNemar discordant-pair counts, each run's failure decomposition, and mean run time.

Usage: analyze_pair.py <basenameA> <basenameB> <labelA> <labelB>
  where <basename>.jsonl lives in $RESULTS_DIR (default: the anyswe results dir).
"""
import json
import os
import re
import sys
from collections import Counter

RESULTS_DIR = os.environ.get(
    "RESULTS_DIR",
    "/lustre/fsw/portfolios/coreai/users/jnolan/Gym/responses_api_agents/anyswe_agent/results")

# OpenClaw bootstrap files: editing them isn't a real code change (see deathmask.py).
BOOTSTRAP_FILES = {
    "AGENTS.md", "SOUL.md", "IDENTITY.md", "HEARTBEAT.md", "TOOLS.md", "USER.md",
    "BOOTSTRAP.md", "NOTES.md", "PLAN.md", "SCRATCHPAD.md", "openclaw-workspace-state.json",
}
DIFF_HEADER = re.compile(r"^diff --git a/(\S+) b/(\S+)", re.M)


def real_source_edits(patch):
    count = 0
    for match in DIFF_HEADER.finditer(patch or ""):
        path = match.group(2)
        name = path.split("/")[-1]
        if name in BOOTSTRAP_FILES:
            continue
        if path.endswith(".md") and "/" not in path:
            continue
        if path.lower().endswith((".png", ".jpg", ".jpeg", ".pdf", ".svg")):
            continue
        count += 1
    return count


def classify(rollout):
    """RESOLVED, or the reason it failed."""
    if rollout.get("resolved"):
        return "RESOLVED"
    if rollout.get("agent_timed_out") or rollout.get("error_kind"):
        return "FAIL_INFRA"
    output = rollout["response"].get("output") or []
    n_tool_calls = sum(1 for it in output if isinstance(it, dict) and it.get("type") == "function_call")
    if n_tool_calls < 2:
        return "FAIL_EARLY_DEATH"     # endpoint returned ~nothing
    if real_source_edits(rollout.get("model_patch")) == 0:
        return "FAIL_NO_EDIT"         # ran but never edited source
    return "FAIL_WRONG_FIX"           # edited the right file, fix didn't pass


def load_run(basename):
    """instance_id -> {resolved, repo, bucket, run_time}; empty dict if the file is absent."""
    path = os.path.join(RESULTS_DIR, basename + ".jsonl")
    tasks = {}
    if not os.path.exists(path):
        return tasks
    for line in open(path):
        rollout = json.loads(line)
        iid = rollout["responses_create_params"]["metadata"]["instance_id"]
        tasks[iid] = dict(
            resolved=bool(rollout.get("resolved")),
            repo=iid.split("__")[0],
            bucket=classify(rollout),
            run_time=rollout.get("openhands_run_time"))
    return tasks


def main():
    base_a, base_b, label_a, label_b = sys.argv[1:5]
    a, b = load_run(base_a), load_run(base_b)
    ids = sorted(set(a) & set(b))
    if not ids:
        sys.exit(f"no paired rollouts (A={len(a)} B={len(b)})")
    n = len(ids)

    resolved_a = sum(a[i]["resolved"] for i in ids)
    resolved_b = sum(b[i]["resolved"] for i in ids)
    print(f"{label_a}: {len(a)} rollouts | {label_b}: {len(b)} rollouts | paired: {n}\n")
    print(f"OVERALL:  {label_a} {resolved_a}/{n} = {100*resolved_a/n:.1f}%   |   "
          f"{label_b} {resolved_b}/{n} = {100*resolved_b/n:.1f}%   "
          f"delta {100*(resolved_b-resolved_a)/n:+.1f}pp\n")

    print(f"PER-REPO ({label_a} -> {label_b}):")
    for repo in sorted({a[i]["repo"] for i in ids}):
        in_repo = [i for i in ids if a[i]["repo"] == repo]
        sa = sum(a[i]["resolved"] for i in in_repo)
        sb = sum(b[i]["resolved"] for i in in_repo)
        print(f"  {repo:<14} {sa:2d}/{len(in_repo):<2d} -> {sb:2d}/{len(in_repo)}")
    print()

    both = sum(1 for i in ids if a[i]["resolved"] and b[i]["resolved"])
    a_only = [i for i in ids if a[i]["resolved"] and not b[i]["resolved"]]
    b_only = [i for i in ids if b[i]["resolved"] and not a[i]["resolved"]]
    neither = sum(1 for i in ids if not a[i]["resolved"] and not b[i]["resolved"])
    print(f"PAIRED: both={both}  {label_a}-only={len(a_only)}  {label_b}-only={len(b_only)}  neither={neither}")
    print(f"  discordant (McNemar): {len(a_only)} vs {len(b_only)} -> {label_b} net {len(b_only)-len(a_only):+d}\n")

    print("FAILURE DECOMPOSITION:")
    for label, run in ((label_a, a), (label_b, b)):
        print(f"  {label:<10} {dict(Counter(run[i]['bucket'] for i in ids))}")
    print()

    times_a = [a[i]["run_time"] for i in ids if a[i]["run_time"]]
    times_b = [b[i]["run_time"] for i in ids if b[i]["run_time"]]
    print(f"mean run_time: {label_a} {sum(times_a)/len(times_a):.0f}s | {label_b} {sum(times_b)/len(times_b):.0f}s")
    print(f"{label_b}-only solves: {b_only}")
    print(f"{label_a}-only solves: {a_only}")


if __name__ == "__main__":
    main()

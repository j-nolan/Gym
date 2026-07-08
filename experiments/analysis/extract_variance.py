#!/usr/bin/env python3
"""Variance decomposition across N same-config reps of the same task set.

Question it answers: of the run-to-run score variance, how much is real agent/sampling variance
vs endpoint flakiness? For each task it counts how often it was solved across the reps
(deterministic vs "coin-flip"), then decomposes the coin-flip failures into endpoint deaths /
gave-up-without-editing / wrong-fix.

Usage: extract_variance.py <rep1_rollouts.jsonl> <rep2_rollouts.jsonl> ...   (one file per rep)
Also dumps per-rollout features to /tmp/variance_feats.json for follow-up trace-diving.
"""
import json
import os
import re
import statistics
import sys
from collections import Counter

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


def features(rollout):
    output = rollout["response"].get("output") or []
    return dict(
        iid=rollout["responses_create_params"]["metadata"]["instance_id"],
        resolved=bool(rollout.get("resolved")),
        error_kind=rollout.get("error_kind"),
        timed_out=bool(rollout.get("agent_timed_out")),
        run_time=rollout.get("openhands_run_time"),
        n_calls=sum(1 for it in output if isinstance(it, dict) and it.get("type") == "function_call"),
        n_real_edits=real_source_edits(rollout.get("model_patch")),
    )


def classify(f):
    if f["resolved"]:
        return "RESOLVED"
    if f["timed_out"] or f["error_kind"]:
        return "FAIL_INFRA"
    if f["n_calls"] < 2:
        return "FAIL_EARLY_DEATH"     # endpoint returned ~nothing
    if f["n_real_edits"] == 0:
        return "FAIL_NO_EDIT"         # explored but never edited source
    return "FAIL_WRONG_FIX"           # edited the right file, fix didn't pass


# NeMo Gym writes these sidecar files next to a rollouts jsonl; they aren't rollouts, so skip
# them if a glob sweeps them in.
SIDECAR_MARKERS = ("materialized", "reward_profiling", "_failures", "metrics")


def load_reps(paths):
    """tag -> {instance_id -> features}, one entry per rep file (tag = filename without .jsonl)."""
    reps = {}
    for path in paths:
        name = os.path.basename(path)
        if any(marker in name for marker in SIDECAR_MARKERS):
            continue
        reps[name.replace(".jsonl", "")] = {
            f["iid"]: f for f in (features(json.loads(l)) for l in open(path))
        }
    return reps


def main():
    rep_files = sys.argv[1:]
    if not rep_files:
        sys.exit(__doc__)
    reps = load_reps(rep_files)
    tags = list(reps)
    task_ids = sorted(next(iter(reps.values())))
    n_reps = len(tags)
    print(f"REPS ({n_reps}): {tags}\n")

    # overall bucket mix across every rollout
    buckets = Counter(classify(reps[t][i]) for t in tags for i in task_ids)
    total = sum(buckets.values())
    print(f"=== ALL {total} rollouts ({n_reps} reps x {len(task_ids)} tasks) bucketed ===")
    for bucket in ["RESOLVED", "FAIL_WRONG_FIX", "FAIL_NO_EDIT", "FAIL_EARLY_DEATH", "FAIL_INFRA"]:
        print(f"  {bucket:<18} {buckets[bucket]:3d}  ({100*buckets[bucket]/total:.1f}%)")
    print()

    # per task: how many reps solved it + the fail mix (ED=early-death, NE=no-edit, WF=wrong-fix,
    # INF=infra); tasks solved in 2..N-2 reps are the "coin-flips" that carry the variance
    print(f"=== per-task across {n_reps} reps: solves | fails ED/NE/WF/INF | medRT | call-spread ===")
    coin_flips = []
    for iid in task_ids:
        outcomes = [classify(reps[t][iid]) for t in tags]
        solved = outcomes.count("RESOLVED")
        times = [reps[t][iid]["run_time"] for t in tags if reps[t][iid]["run_time"]]
        calls = [reps[t][iid]["n_calls"] for t in tags]
        if 2 <= solved <= n_reps - 2:
            coin_flips.append(iid)
            tag = "COINFLIP"
        else:
            tag = "always" if solved == n_reps else "never" if solved == 0 else ""
        print(f"{iid:<30} {solved}/{n_reps}  "
              f"ED{outcomes.count('FAIL_EARLY_DEATH')} NE{outcomes.count('FAIL_NO_EDIT')} "
              f"WF{outcomes.count('FAIL_WRONG_FIX')} INF{outcomes.count('FAIL_INFRA')}  "
              f"{statistics.median(times) if times else 0:4.0f}s  {min(calls)}-{max(calls)}  {tag}")

    print(f"\n=== VARIANCE SOURCE on the {len(coin_flips)} coin-flip tasks (they carry all the variance) ===")
    cf = Counter(classify(reps[t][iid]) for iid in coin_flips for t in tags)
    solves = cf["RESOLVED"]
    fails = sum(v for k, v in cf.items() if k != "RESOLVED")
    print(f"coin-flip rollouts: {len(coin_flips)*n_reps}  (solves {solves}, fails {fails})")
    if fails:
        print(f"  of the fails: EARLY_DEATH(endpoint) {cf['FAIL_EARLY_DEATH']} ({100*cf['FAIL_EARLY_DEATH']/fails:.0f}%) | "
              f"NO_EDIT(gave up) {cf['FAIL_NO_EDIT']} ({100*cf['FAIL_NO_EDIT']/fails:.0f}%) | "
              f"WRONG_FIX(sampling) {cf['FAIL_WRONG_FIX']} ({100*cf['FAIL_WRONG_FIX']/fails:.0f}%)")

    # NO_EDIT fails: long + few-calls = latency-starved; short = quick give-up
    no_edit = [(iid, t, reps[t][iid]["n_calls"], reps[t][iid]["run_time"])
               for iid in coin_flips for t in tags if classify(reps[t][iid]) == "FAIL_NO_EDIT"]
    if no_edit:
        print("\n  NO_EDIT fails (explored but never edited source) -- longest first:")
        for iid, tag, calls, rt in sorted(no_edit, key=lambda x: -(x[3] or 0)):
            print(f"    {iid:<30} {tag:<12} calls={calls:<3d} rt={int(rt) if rt else '?'}s")

    json.dump({t: reps[t] for t in tags}, open("/tmp/variance_feats.json", "w"))
    print("\n(per-rollout features -> /tmp/variance_feats.json)")


if __name__ == "__main__":
    main()

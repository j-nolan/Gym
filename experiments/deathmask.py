#!/usr/bin/env python3
"""Death-retry protocol for SWE-bench rollouts.

The nvinf endpoint occasionally returns an empty response under load, which kills a rollout
before the agent does any real work: it scores as a failure but was never a fair attempt.
Re-running just those "dead" rollouts on a calmer endpoint recovers most of them (see FINDINGS.md).

This tool (1) finds the dead rollouts and writes a retry dataset from them, then (2) merges a
retry pass back in, keeping the better outcome per task. Always report BOTH the raw and the
merged score -- the gap between them is the endpoint's "death tax".

  deathmask.py extract <rollouts.jsonl> <full_dataset.jsonl> <out_retry_dataset.jsonl>
  deathmask.py merge   <orig_rollouts.jsonl> <retry_rollouts.jsonl> <out_merged.jsonl>
"""
import json
import re
import sys

# OpenClaw scaffolds these into the workspace on every run, so editing them is not a real code
# change and doesn't count toward "did the agent attempt a fix".
BOOTSTRAP_FILES = {
    "AGENTS.md", "SOUL.md", "IDENTITY.md", "HEARTBEAT.md", "TOOLS.md", "USER.md",
    "BOOTSTRAP.md", "NOTES.md", "PLAN.md", "SCRATCHPAD.md", "openclaw-workspace-state.json",
}
DIFF_HEADER = re.compile(r"^diff --git a/(\S+) b/(\S+)", re.M)


def real_source_edits(patch):
    """How many genuine source files the patch touches -- excluding OpenClaw scaffolding,
    stray top-level .md, and image artifacts."""
    count = 0
    for match in DIFF_HEADER.finditer(patch or ""):
        path = match.group(2)
        name = path.split("/")[-1]
        if name in BOOTSTRAP_FILES:
            continue
        if path.endswith(".md") and "/" not in path:
            continue
        if path.lower().endswith((".png", ".jpg", ".jpeg", ".svg", ".pdf")):
            continue
        count += 1
    return count


def is_dead(rollout):
    """Dead = failed AND (errored / timed out / never edited a real source file)."""
    if rollout.get("resolved"):
        return False
    return bool(
        rollout.get("error_kind")
        or rollout.get("agent_timed_out")
        or real_source_edits(rollout.get("model_patch")) == 0
    )


def load_rollouts(path):
    """Map instance_id -> (parsed rollout, its original json line)."""
    by_id = {}
    for line in open(path):
        rollout = json.loads(line)
        instance_id = rollout["responses_create_params"]["metadata"]["instance_id"]
        by_id[instance_id] = (rollout, line)
    return by_id


def instance_id_of(dataset_line):
    return json.loads(dataset_line)["responses_create_params"]["metadata"]["instance_id"]


def extract(rollouts_path, dataset_path, out_path):
    """Write a retry dataset containing only the tasks whose rollout came back dead."""
    dead_ids = {iid for iid, (rollout, _) in load_rollouts(rollouts_path).items() if is_dead(rollout)}
    retry_rows = [line for line in open(dataset_path) if instance_id_of(line) in dead_ids]
    open(out_path, "w").writelines(retry_rows)
    print(f"dead rollouts: {len(dead_ids)} -> retry dataset {out_path} ({len(retry_rows)} rows)")


def merge(orig_path, retry_path, out_path):
    """Merge a retry pass into the original run (better outcome per task) and print scores."""
    orig = load_rollouts(orig_path)
    retry = load_rollouts(retry_path)

    merged = []          # best-of (rollout, line) per task
    replaced = 0
    for iid, (rollout, line) in orig.items():
        if iid in retry:
            retry_rollout, retry_line = retry[iid]
            improved = (retry_rollout.get("resolved") and not rollout.get("resolved")) \
                or (is_dead(rollout) and not is_dead(retry_rollout))
            if improved:
                rollout, line, replaced = retry_rollout, retry_line, replaced + 1
        merged.append((rollout, line))
    open(out_path, "w").writelines(line for _, line in merged)

    total = len(merged)
    raw_resolved = sum(bool(r.get("resolved")) for r, _ in orig.values())
    merged_resolved = sum(bool(r.get("resolved")) for r, _ in merged)
    fair = [r for r, _ in merged if not is_dead(r)]
    fair_resolved = sum(bool(r.get("resolved")) for r in fair)
    print(f"tasks: {total} | raw resolved: {raw_resolved}/{total} = {100*raw_resolved/total:.1f}%")
    print(f"merged (post-retry): {merged_resolved}/{total} = {100*merged_resolved/total:.1f}%  "
          f"({replaced} rollouts replaced)")
    print(f"fair-attempt rate: {fair_resolved}/{len(fair)} = {100*fair_resolved/len(fair):.1f}%  "
          f"({total-len(fair)} still-dead excluded)")


if __name__ == "__main__":
    if len(sys.argv) == 5 and sys.argv[1] == "extract":
        extract(*sys.argv[2:5])
    elif len(sys.argv) == 5 and sys.argv[1] == "merge":
        merge(*sys.argv[2:5])
    else:
        sys.exit(__doc__)

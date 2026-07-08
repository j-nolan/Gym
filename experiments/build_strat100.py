#!/usr/bin/env python3
"""Build a reproducible, difficulty-x-repo-stratified 100-task subset of SWE-bench Verified.

Design (per Gaia's tips): span all repos AND all difficulty bands, over-sample the scarce
hard tasks so they carry real signal, cap any single repo so django (46% of the 500) can't
dominate. Deterministic (sorted + no randomness) so the list is a stable repro/smoke set.
"""
import json, collections

SRC = "/lustre/fsw/portfolios/coreai/users/jnolan/Gym/responses_api_agents/anyswe_agent/data/verified500.jsonl"
OUT = "/lustre/fsw/portfolios/coreai/users/jnolan/Gym/responses_api_agents/anyswe_agent/data/strat100.jsonl"
IDS = "/lustre/fsw/portfolios/coreai/users/jnolan/strat100_ids.txt"

# difficulty band targets (sum=100). Population is 194/261/42/3; we flatten toward the hard
# end so the ~9% hard tasks get real representation in a smoke/iteration set.
DIFF_TARGET = {"<15 min fix": 30, "15 min - 1 hour": 45, "1-4 hours": 22, ">4 hours": 3}
REPO_CAP = 15  # no single repo may exceed this in the subset (django would otherwise be ~46)

rows = [json.loads(l) for l in open(SRC)]
def meta(r): return r["responses_create_params"]["metadata"]
def repo(r): return meta(r)["instance_id"].split("__")[0]
def diff(r): return json.loads(meta(r)["instance_dict"]).get("difficulty", "?")

# cells: (repo, difficulty) -> [rows], each sorted by instance_id for determinism
cells = collections.defaultdict(list)
for r in sorted(rows, key=lambda x: meta(x)["instance_id"]):
    cells[(repo(r), diff(r))].append(r)

repos = sorted({repo(r) for r in rows})
picked, per_repo = [], collections.Counter()

# hardest band first so scarce hard tasks are locked in before repo caps fill
for band in [">4 hours", "1-4 hours", "15 min - 1 hour", "<15 min fix"]:
    target = DIFF_TARGET[band]
    got = 0
    # round-robin over repos to spread the band across repos
    progress = True
    while got < target and progress:
        progress = False
        for rp in repos:
            if got >= target:
                break
            if per_repo[rp] >= REPO_CAP:
                continue
            bucket = cells.get((rp, band), [])
            if bucket:
                picked.append(bucket.pop(0))
                per_repo[rp] += 1
                got += 1
                progress = True
    if got < target:
        print(f"  NOTE: band {band!r} short: got {got}/{target} (supply/cap limited)")

with open(OUT, "w") as f:
    for r in picked:
        f.write(json.dumps(r) + "\n")
with open(IDS, "w") as f:
    f.write("\n".join(meta(r)["instance_id"] for r in picked) + "\n")

# report the resulting matrix
print(f"built {len(picked)} tasks -> {OUT}")
rc = collections.Counter(repo(r) for r in picked)
dc = collections.Counter(diff(r) for r in picked)
print("difficulty:", dict((k, dc[k]) for k in DIFF_TARGET))
print("repos:", dict(rc.most_common()))
print("repo x difficulty matrix:")
mat = collections.Counter((repo(r), diff(r)) for r in picked)
bands = ["<15 min fix", "15 min - 1 hour", "1-4 hours", ">4 hours"]
print("  %-14s %s" % ("repo", " ".join("%12s" % b for b in bands)))
for rp in sorted(rc, key=lambda x: -rc[x]):
    print("  %-14s %s" % (rp, " ".join("%12d" % mat[(rp, b)] for b in bands)))

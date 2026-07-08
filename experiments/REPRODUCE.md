# OpenClaw × Nemotron-3-ultra on SWE-bench Verified

## Goal
Measure how well Nemotron-3-ultra, driven by the OpenClaw agent harness, resolves real bugs from
SWE-bench Verified — run on the `nsc-svg-slurm-1` cluster through NeMo Gym's `anyswe` harness.

## Approach
We run OpenClaw under **harbor's** configuration rather than a hand-tuned one, so the parameters
are an outside reference point (harbor is a parity-validated agent-benchmark framework) instead of
choices we'd have to defend. The one departure is `timeoutSeconds: 390`: harbor keeps OpenClaw's
120s idle default, which silently aborts Nemotron's slow prefill against our endpoint.

## Building blocks
- **Base** — NeMo Gym on the `cmunley1/anyswe` branch: the `anyswe` SWE harness + its apptainer
  sandbox provider.
- **Config** — `configs/anyswe_openclaw_harborparity.yaml`: harbor's defaults (thinking high, no
  injected system prompt, minimal tools) plus the timeout deviation and the `--no-mount home,cwd`
  sandbox fix. Individual knobs are commented in the file.
- **Launcher + meta-container** — the Slurm side lives in the team's slurm-evaluations repo
  (`interactive-agents/slurm-evaluations` @ `a090ab7`): its `scripts/run_eval.sh` drives the run,
  and its `container/Dockerfile.apptainer` builds the `nemogym-eval:apptainer` meta-container
  (ubuntu + python + apptainer) that the eval re-execs into, since the cluster has no native
  apptainer. `run_eval.sh` pulls that image from the registry. Run it from that repo — don't vendor
  a copy.
- **Task images** — one apptainer `.sif` per task, built from the public `swebench` Docker images.
- **Datasets** — `verified500` (the full split, for the headline number) and `strat100` (a
  difficulty/repo-stratified subset for fast iteration).

## Pipeline
`sbatch` launches slurm-evaluations `run_eval.sh`, which re-execs into the meta-container so
apptainer is on PATH, starts the NeMo Gym servers, then for each task hands OpenClaw its own
apptainer sandbox on `/testbed`. OpenClaw edits the repo; the SWE-bench harness then grades the
resulting `git diff`.

## Running it
1. `uv venv --python 3.12 && uv sync`  (uv 0.11.7; uv cache on lustre)
2. Build task images: `prepare.py --sif-dir $LUSTRE/sifs --jobs N`
3. Datasets: `prepare.py` → `verified500`; `python experiments/build_strat100.py` → `strat100`
4. Secrets: `.env` (`NVIDIA_INTERNAL_API_KEY`, `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `LUSTRE`) and
   `env.yaml` (`policy_*` via `${oc.env:...}`)
5. `python experiments/preflight.py --dataset <data>.jsonl --formatter $LUSTRE/sifs/{instance_id}.sif`
6. From a slurm-evaluations checkout (it pulls the meta-container itself — no `CONTAINER_IMAGE`
   override needed):
   `sbatch --time=24:00:00 scripts/run_eval.sh -b anyswe -d strat100 -a openclaw_harborparity -c 4`
   For the full 500, shard the dataset into slices that each fit the wall and concatenate the
   rollouts (see caveats).
7. `experiments/deathmask.py`: retry endpoint-killed rollouts off-peak, then merge — report the raw
   and the fair-attempt score

## Caveats
- The endpoint doesn't report token counts, so derive them by tokenizing the saved transcripts.
- Endpoint load varies through the day: under weekday peak it drops requests as empty responses,
  which look like failures. Run off-peak and/or use the death-retry pass to separate that out.
- A run longer than the 24h wall must be **sharded** — disjoint dataset slices run as separate
  jobs, rollouts concatenated after — not chained with `--resume`. Gym's resume re-keys on the
  regenerated materialized-input indices, so a follow-on leg matches nothing, restarts from
  scratch, and overwrites the prior leg's output.
- `--no-container-mount-home` guards a second apptainer footgun (the outer container mounting
  `$HOME`); it's in slurm-evaluations `run_eval.sh` as of MR #6.

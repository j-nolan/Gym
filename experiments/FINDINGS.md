# Findings & design decisions

Why the setup is what it is — the durable conclusions behind the config and pipeline. The
detailed run-by-run history lives in git history and the result files, not here.

## Reference configuration: harbor's
We evaluate under harbor's OpenClaw config rather than a hand-tuned one, so the parameters are an
external reference (harbor is a parity-validated agent-benchmark framework) rather than choices
we'd have to defend. Sole deviation: `timeoutSeconds: 390` — harbor keeps OpenClaw's 120s idle
default, which silently aborts Nemotron's slow prefill against our endpoint.

## Measurement discipline
- Single-run scores are noise-dominated (~5-10pp), so compare configs **paired on the same tasks**
  (McNemar on the discordant pairs), never two separate runs.
- The endpoint drops requests under load as empty responses that read as failures. Report **two
  numbers** — raw, and fair-attempt after the death-retry pass; the gap is endpoint tax, not model
  quality.
- Iterate on the stratified subset; reserve the full 500 for the headline number.

## What doesn't move the score (don't re-try these)
Independent, mechanically-verified negatives — each changed the agent's behavior but not the
resolve rate:
- **Temperature** (1.0 → 0.2): no significant effect. The failing fixes are wrong, not random;
  lower temperature just commits harder to the wrong answer.
- **Env-fix prompt** (point the agent at the pre-installed testbed env): removed the "wander into a
  build rabbit hole" failures, but those tasks then failed as wrong-fixes — net flat.
- **Sandbox runtime** (enroot vs apptainer): identical within noise.
- **Aggressive context-bounding**: actively *hurt* (starved the agent); the latency stalls it was
  meant to fix are better handled by a generous idle timeout.

Common thread: OpenClaw × Nemotron sits at a **~50% capability plateau**; the dominant failure is
"edited the right file, fix doesn't pass" — a model-capability limit, not a harness knob.

## Sandbox escape (why the config carries `--no-mount home,cwd`)
Apptainer auto-mounts the working directory (and `$HOME`) writable into every task container. The
working dir is the Gym root, so agents running `pip install` wrote through it into the *shared*
dependency prefix and corrupted it for all concurrent tasks — silently, scored as failures.
`--no-mount home,cwd` closes it; `--no-container-mount-home` closes the same footgun at the outer
enroot layer.

## Run protocol
`preflight.py` (images / deps prefix / endpoint) before a run; pinned sampling; `deathmask.py`
death-retry after; and a pace-guard that kills a run when rollouts complete impossibly fast — the
signature of prefix corruption.

# The campaign boundary

A *campaign* is 10⁴–10⁶ registered, restartable runs over a rented fleet
(DESIGN.md preamble). This note draws the seam between the **physics** and the
**orchestration** so the orchestration can eventually become a standalone,
physics-agnostic package — without committing to a second repo before 0.1.

## Why a boundary at all (the research, 2026-06-12)

Two verified literature sweeps drive this:

1. **The niche is open.** No JAX soliton/topological-field engine and no
   farm-scale soliton campaign system exist in the verified literature
   (10 candidate URLs, 0 qualified). Campaign-scale orchestration *is* the
   moat — see README positioning.
2. **Build thin, don't rebuild.** Of 8 orchestration tools assessed against a
   five-part contract, **none covers it**, but the split is uneven:

   | Contract | What it is | Best off-the-shelf | Verdict |
   |---|---|---|---|
   | **A** | config-hashed run registry | DVC (partial, stage-level) | **build** (already have it) |
   | **B** | bit-identical full-integrator-state restart | — (all "re-run from artifact") | **build** (already have it) |
   | **C** | event-records-not-fields + triggered full-state capture | Balsam/Parsl (event logs, no trigger) | **build** |
   | **D** | spot-fleet fan-out + preemption recovery | **SkyPilot / dstack** (clean) | **adopt** |
   | **E** | probe-or-bail admission on flaky hosts | — (served by no one) | **build** (novel) |

   The executor layer (D) is a crowded, solved market — adopt SkyPilot, never
   rebuild it. The provenance/restart/event/admission contract (A/B/C/E) is
   unserved, and **E exists nowhere** — every surveyed tool assumes reliable
   hosts. That is the part worth owning.

## The contract → DESIGN.md principles

| Letter | Protocol | Responsibility | Principle |
|---|---|---|---|
| A, B | `RunRegistry` | config-hashed dirs + manifest; full-state checkpoints; idempotent skip | P4 |
| C | `EventSink` | stream small per-run records; capture full fields only when triggered | P6, P7 |
| D | `Executor` | fan out tasks over a fleet; recover from preemption (re-submit incomplete) | adopted |
| E | `Admission` | probe a host's real compute+network capacity; bail loudly on bad hosts | P9 |

A/B already live in `runs.py` (`RunConfig.config_hash`, `save_checkpoint`,
`load_checkpoint`) — they are *already physics-agnostic*. The campaign module
wraps them today; at extraction time they move wholesale into the new package.

## The one seam the physics crosses

The **only** soliton-specific thing that crosses this boundary is a single
injected callable:

```python
RunFn = Callable[[RunConfig, RunContext], dict]
```

`RunContext` hands the physics four orchestration capabilities and nothing
else: `ctx.resume` (prior full state or `None`), `ctx.checkpoint(state, step)`,
`ctx.emit(record)`, `ctx.trigger(state, reason)`. The physics returns a small
result record. **No module under `campaign/` imports a model, a stepper, or
jax** (beyond the array-I/O already in `runs.py`). That discipline is what
keeps the layer extractable.

## Extraction plan (rule of three)

Keep `campaign/` an internal module of `jax-solitons` until **either** a second
real consumer appears **or** the A/B/C/E API stabilizes through one real
collider campaign. Then lift `campaign/` + the `RunConfig`/checkpoint helpers
out of `runs.py` into a standalone package (working title: a "checkpointed
campaign registry over a pluggable spot executor"). The research proves the
niche stays open, so there is no first-mover cost to waiting — only API risk
avoided. Splitting a second repo now, with one consumer and C/E unbuilt, buys
version-lock friction and a guessed-at abstraction.

## Status

Sketch only. `protocols.py` is the contract; `reference.py` has thin
local-machine implementations (functional) plus a `SkyPilotExecutor` stub
(documents the intended mapping, raises `NotImplementedError`). Nothing here
is wired into CI yet — it is the boundary, drawn, so the collider-campaign
work (TODO.md) builds against a fixed contract.

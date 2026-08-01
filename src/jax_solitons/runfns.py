"""First real `RunFn`: the Faddeev relax-then-ID pipeline behind the campaign
contract (design: CAMPAIGN.md).

Proof-of-use for `run_farm` (the extracted campaign layer). A physics `RunFn` that deep-relaxes a
rational-map hopfion in the CP^1 spinor frame (the frame that reaches the
virial point; see steppers/adam.py), streams a charge/energy ledger as it goes
(P6), checkpoints FULL optimizer state so a preempted run resumes bit-
identically (P4/contract B), and — only AFTER the quench — traces the soliton
core curve and captures it as a triggered full-state event (P7, relax-then-ID:
descent cannot create topology, so basin-ID is faithful where in-bath is not).

This module is the ONE place physics meets the boundary: it imports models,
seeds, steppers, and run-farm's `RunContext`, and exposes a single callable
matching `run_farm.RunFn`. Nothing in run-farm imports anything here.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from run_farm.protocols import RunContext
from jax_solitons.grid import BoxGrid
from jax_solitons.measure import curve_length, trace_curves
from jax_solitons.models.faddeev import faddeev_cp1_model, n_from_state
from jax_solitons.runs import RunConfig
from jax_solitons.seeds import rational_map_hopfion_cp1
from jax_solitons.steppers.adam import adam_flow


def _ledger_row(model, z, grid, step) -> dict:
    """One small event record: the energy split, virial ratio, and charge."""
    E2 = float(model.terms[0](z, grid))
    E4 = float(model.terms[1](z, grid))
    QH = float(model.charges[0](z, grid))
    return {
        "step": int(step),
        "E": E2 + E4,
        "E2": E2,
        "E4": E4,
        "E2_over_E4": (E2 / E4) if E4 else float("inf"),
        "Q_H": QH,
    }


def faddeev_relax_then_id(config: RunConfig, ctx: RunContext) -> dict:
    """RunFn: relax a (n, m) hopfion in the spinor frame, then ID its core.

    config.params keys (all optional): c4, lr, R, n, m, segments. The run is
    split into `segments` Adam chunks; after each, a ledger row is emitted and
    full state (spinor + Adam moments + step counter) is checkpointed, so a
    spot preemption resumes mid-relaxation with no lost work and no drift.
    Returns the final ledger row plus the core-curve census.
    """
    p = config.params
    dtype = jnp.float64 if config.dtype == "float64" else jnp.float32
    grid = BoxGrid(N=config.N, L=config.L, dtype=dtype)
    model = faddeev_cp1_model(c4=p.get("c4", 4.0))
    lr = p.get("lr", 2e-3)
    n_seg = max(1, min(int(p.get("segments", 8)), config.steps))
    # Distribute the remainder so the executed steps sum to EXACTLY config.steps
    # (a fixed steps//n_seg silently drops it -- a provenance smell for a
    # provenance layer). Divisible configs are unchanged (e.g. 120/4 -> 30x4).
    base, rem = divmod(config.steps, n_seg)
    seg_lengths = [base + (1 if i < rem else 0) for i in range(n_seg)]

    # Resume from a checkpoint (contract B) or seed a fresh rational-map hopfion.
    if ctx.resume is not None:
        r = ctx.resume
        z = jnp.asarray(r["z"])
        opt = (jnp.asarray(r["m"]), jnp.asarray(r["v"]), int(np.asarray(r["t"])))
        seg0 = int(np.asarray(r["seg"]))
    else:
        z = rational_map_hopfion_cp1(
            grid, R=p.get("R", 3.5), n=p.get("n", 1), m=p.get("m", 1))
        opt, seg0 = None, 0
        ctx.emit(_ledger_row(model, z, grid, 0))

    # Segmented descent: emit + checkpoint full optimizer state between chunks.
    # The cumulative step comes through the contract (ctx.resume_step), not
    # smuggled in State; seg0/optimizer moments are genuine loop state.
    step = ctx.resume_step if ctx.resume_step is not None else 0
    for seg in range(seg0, n_seg):
        z, _obs, opt = adam_flow(
            model, z, grid, lr=lr, steps=seg_lengths[seg],
            opt_state=opt, return_opt_state=True)
        step += seg_lengths[seg]
        ctx.emit(_ledger_row(model, z, grid, step))
        m, v, t = opt
        ctx.checkpoint({
            "z": np.asarray(z), "m": np.asarray(m), "v": np.asarray(v),
            "t": np.array(t), "seg": np.array(seg + 1),
        }, step)

    # Quench complete -> relax-then-ID (P7): trace the core {n1=0, n2=0, n3>0}.
    result = _ledger_row(model, z, grid, step)   # step == config.steps exactly
    nf = np.asarray(n_from_state(z))
    try:
        curves = trace_curves(nf[0], nf[1], grid, mask=nf[2] > 0)
    except Exception as e:  # ID is best-effort; a failed trace is a recorded fact
        result["id_error"] = str(e)
        curves = []
    result["n_curves"] = len(curves)
    result["core_length"] = float(curve_length(curves[0])) if curves else 0.0

    # The rare kept full-state capture: only the relaxed, identified event (P6).
    if curves:
        ctx.trigger(
            {"z": np.asarray(z), "core_curve": np.asarray(curves[0])},
            reason="relaxed_core")
    return result

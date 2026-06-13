"""Projected Adam descent.

The arrested (backtracking) flow is guaranteed monotone but can stall ~2%
above the Derrick minimum: near the bottom the constraint retraction blocks
any strictly-descending step, and the line search arrests before the
soliton finishes growing to its virial size (E2/E4 freezes below 1). Adam
needs no strict descent and glides through that stall -- but only in the
right COORDINATE FRAME. Its per-coordinate steps see the soft Derrick
scaling mode through the state parametrization: for Faddeev-Skyrme, Adam
on the n-field plateaus at E2/E4 ~ 0.68 with E creeping UP at any constant
lr (measured at N=96: lr=2e-3 and 1e-2 both crawl +0.002/1k steps), while
Adam on the CP^1 spinor (models.faddeev_cp1_model) reaches the virial
point E2/E4 ~ 0.91 within 2k steps at lr=2e-3. Deep relaxation should use
the spinor frame. The state is retracted onto the constraint manifold
after every step (projected Adam).

`lr` may be a traceable schedule (step -> lr) for annealed descent.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from jax_solitons.grid import BoxGrid
from jax_solitons.model import Model


def adam_flow(model: Model, state, grid: BoxGrid, *, lr=2e-3, steps=5000,
              b1=0.9, b2=0.999, eps=1e-8, observe_every=0, observer=None,
              opt_state=None, return_opt_state=False):
    """Projected Adam on model.energy. Returns (state, observations).

    `lr` is a float or a traceable schedule step -> learning rate (step is a
    traced 1-based jnp scalar; use jnp ops, not python branches).

    For checkpointable / restartable descent (campaign contract B, P4), pass
    `opt_state=(m, v, t)` from a prior call's `return_opt_state=True` result:
    the moment estimates and the bias-correction step counter carry over, so a
    segmented run is bit-identical to the uninterrupted one. Both the schedule
    AND the observer see the GLOBAL step `t`, not a per-segment reset, so
    observation labels stay monotone across a resume.
    """
    grad = jax.grad(lambda s: model.energy(s, grid))
    constraint = model.constraint
    schedule = lr if callable(lr) else (lambda t: lr)

    @jax.jit
    def step(state, m, v, t):
        g = grad(state)
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g**2
        mh = m / (1 - b1**t)
        vh = v / (1 - b2**t)
        new = state - schedule(t) * mh / (jnp.sqrt(vh) + eps)
        if constraint is not None:
            new = constraint.retract(new)
        return new, m, v

    if opt_state is None:
        m = jnp.zeros_like(state)
        v = jnp.zeros_like(state)
        t0 = 0
    else:
        m, v, t0 = opt_state
        m, v, t0 = jnp.asarray(m), jnp.asarray(v), int(t0)
    obs = []
    for i in range(steps):
        if observer and observe_every and (i % observe_every == 0):
            obs.append(observer(t0 + i, state))   # global step, not segment-local
        state, m, v = step(state, m, v, t0 + i + 1)
    if observer:
        obs.append(observer(t0 + steps, state))
    if return_opt_state:
        return state, obs, (m, v, t0 + steps)
    return state, obs

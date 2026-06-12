"""Projected Adam descent.

The arrested (backtracking) flow is guaranteed monotone but can stall ~2%
above the Derrick minimum: near the bottom the constraint retraction blocks
any strictly-descending step, and the line search arrests before the
soliton finishes growing to its virial size (E2/E4 freezes below 1). Adam
needs no strict descent -- its per-coordinate adaptive step glides through
the stall and drives E2 -> E4. The state is retracted onto the constraint
manifold after every step (projected Adam).

Use arrested_flow for guaranteed-monotone scouting and basin entry; finish
with projected Adam when the converged minimum (virial ratio ~ 1) matters.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from jax_solitons.grid import BoxGrid
from jax_solitons.model import Model


def adam_flow(model: Model, state, grid: BoxGrid, *, lr=2e-3, steps=5000,
              b1=0.9, b2=0.999, eps=1e-8, observe_every=0, observer=None):
    """Projected Adam on model.energy. Returns (state, observations)."""
    grad = jax.grad(lambda s: model.energy(s, grid))
    constraint = model.constraint

    @jax.jit
    def step(state, m, v, t):
        g = grad(state)
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g**2
        mh = m / (1 - b1**t)
        vh = v / (1 - b2**t)
        new = state - lr * mh / (jnp.sqrt(vh) + eps)
        if constraint is not None:
            new = constraint.retract(new)
        return new, m, v

    m = jnp.zeros_like(state)
    v = jnp.zeros_like(state)
    obs = []
    for i in range(steps):
        if observer and observe_every and (i % observe_every == 0):
            obs.append(observer(i, state))
        state, m, v = step(state, m, v, i + 1)
    if observer:
        obs.append(observer(steps, state))
    return state, obs

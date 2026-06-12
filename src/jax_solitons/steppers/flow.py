"""Arrested (monotone backtracking) energy descent.

Fixed-step gradient flow on a stiff quartic is not monotone: once a soliton
tightens, the step overshoots and the energy rises past the minimum -- an
integration artifact that masquerades as an instability. Here every step is
accepted only if E decreases; otherwise dt halves and retries, and dt grows
slowly on success (arrested-Newton behaviour: the flow settles AT the
minimum). A Fourier preconditioner 1/(1 + dt(alpha k^2 + beta k^4)) treats
the E2 (k^2) and quartic (k^4) stiffness implicitly so usable steps are not
crushed to zero.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from jax_solitons.grid import BoxGrid
from jax_solitons.model import Model


def arrested_flow(model: Model, state, grid: BoxGrid, *, dt=2e-4, steps=2000,
                  alpha=2.0, beta=1.0, dt_min=1e-10, dt_max=5e-3,
                  log_every=0):
    """Monotone backtracking preconditioned descent on model.energy.

    `state` is the field array (component axis leading, e.g. (3, N, N, N));
    the model's constraint retraction is applied after every trial step.
    Returns (state, history) with history rows (step, E, dt).
    """
    _, _, _, K2 = grid.k_vectors()
    energy = jax.jit(lambda s: model.energy(s, grid))
    grad = jax.jit(jax.grad(lambda s: model.energy(s, grid)))

    @jax.jit
    def trial(state, dt):
        g = grad(state)
        if model.constraint is not None:
            g = model.constraint.project_tangent(state, g)
        precond = 1.0 / (1.0 + dt * (alpha * K2 + beta * K2**2))
        stepped = state - dt * g
        sh = jnp.fft.fftn(stepped, axes=(-3, -2, -1))
        new = jnp.real(jnp.fft.ifftn(precond * sh, axes=(-3, -2, -1)))
        new = new.astype(state.dtype)
        if model.constraint is not None:
            new = model.constraint.retract(new)
        return new

    hist = []
    E_cur = float(energy(state))
    for i in range(steps):
        accepted = False
        for _ in range(40):
            trial_state = trial(state, dt)
            E_new = float(energy(trial_state))
            if np.isfinite(E_new) and E_new <= E_cur:
                state, E_cur = trial_state, E_new
                dt = min(dt * 1.05, dt_max)
                accepted = True
                break
            dt *= 0.5
            if dt < dt_min:
                break
        if log_every and (i % log_every == 0):
            hist.append((i, E_cur, dt))
        if not accepted:
            break   # converged/stalled: no descending step exists above dt_min
    hist.append((steps, E_cur, dt))
    return state, hist

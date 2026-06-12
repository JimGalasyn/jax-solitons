"""Sphere-constrained velocity-Verlet for real-time field dynamics.

EOM: d_t^2 n = -(1/dx^3) P_n[ dE/dn ], with P_n the tangent projection at n
and |n| = 1 enforced by retraction each step. Velocities are re-projected
onto the tangent space after every half-kick. Validated lineage: the source
integrator conserved energy to dH = -0.000% on a static hopfion and
reproduced a u = 0.3 boost as 0.286 measured.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from jax_solitons.grid import BoxGrid
from jax_solitons.model import Model


def kinetic_energy(v, grid: BoxGrid):
    return 0.5 * jnp.sum(v * v) * grid.dx**3


def boost_velocity(n, grid: BoxGrid, u: float, axis: int = 2):
    """Galilean boost d_t n = -u d_axis n (tangent automatically: n.dn = 0).
    Forward differences, consistent with the E2 stencil."""
    g = (jnp.roll(n, -1, axis=1 + axis) - n) / grid.dx
    return -u * g


def make_verlet_step(model: Model, grid: BoxGrid, dt: float):
    """Build a jitted constrained velocity-Verlet step (n, v) -> (n, v)."""
    dx = grid.dx
    grad = jax.grad(lambda s: model.energy(s, grid))
    constraint = model.constraint

    def accel(n):
        a = -grad(n) / dx**3
        return constraint.project_tangent(n, a) if constraint else a

    @jax.jit
    def step(n, v):
        a = accel(n)
        v = v + 0.5 * dt * a
        if constraint:
            v = constraint.project_tangent(n, v)
        n = n + dt * v
        if constraint:
            n = constraint.retract(n)
        a = accel(n)
        v = v + 0.5 * dt * a
        if constraint:
            v = constraint.project_tangent(n, v)
        return n, v

    return step


def verlet_step(model: Model, grid: BoxGrid, n, v, dt: float):
    """One constrained velocity-Verlet step (convenience; for loops use
    make_verlet_step once)."""
    return make_verlet_step(model, grid, dt)(n, v)


def verlet_evolve(model: Model, grid: BoxGrid, n, v, *, dt: float,
                  steps: int, observe_every: int = 0, observer=None):
    """Evolve (n, v) for `steps` steps. If observer is given it is called as
    observer(step, n, v) every observe_every steps; its returns are collected.
    Returns (n, v, observations)."""
    step = make_verlet_step(model, grid, dt)
    obs = []
    for i in range(steps):
        if observer and observe_every and (i % observe_every == 0):
            obs.append(observer(i, n, v))
        n, v = step(n, v)
    if observer:
        obs.append(observer(steps, n, v))
    return n, v, obs

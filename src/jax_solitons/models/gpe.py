"""Gross-Pitaevskii model in healing-length units.

    E[psi] = int  1/2 |grad psi|^2  +  g/2 (|psi|^2 - 1)^2

State: complex psi, shape (N, N, N). No manifold constraint; the vacuum is
|psi| = 1 with quantized-circulation vortex lines as the topological
excitations. Imaginary-time relaxation (steppers.splitstep) is the
production relaxer; the autodiff steppers also work on (Re, Im) views.

Kinetic stencil: forward differences, the engine-wide convention (the
source code's diagnostic used central differences; values differ at
O(dx^2), the physics does not).
"""

from __future__ import annotations

import dataclasses

import jax.numpy as jnp

from jax_solitons.grid import BoxGrid
from jax_solitons.model import Model


def _fwd_grads_c(field, dx):
    gx = (jnp.roll(field, -1, 0) - field) / dx
    gy = (jnp.roll(field, -1, 1) - field) / dx
    gz = (jnp.roll(field, -1, 2) - field) / dx
    return gx, gy, gz


@dataclasses.dataclass(frozen=True)
class GPEKineticTerm:
    name: str = "gpe_kinetic"

    def __call__(self, psi, grid: BoxGrid):
        gx, gy, gz = _fwd_grads_c(psi, grid.dx)
        e = jnp.abs(gx) ** 2 + jnp.abs(gy) ** 2 + jnp.abs(gz) ** 2
        return 0.5 * jnp.sum(e) * grid.dx**3


@dataclasses.dataclass(frozen=True)
class GPEPotentialTerm:
    g: float = 1.0
    name: str = "gpe_potential"

    def __call__(self, psi, grid: BoxGrid):
        return 0.5 * self.g * jnp.sum((jnp.abs(psi) ** 2 - 1.0) ** 2) * grid.dx**3


def gpe_model(g: float = 1.0) -> Model:
    """The GPE as a Model configuration (no constraint; vacuum |psi|=1)."""
    return Model(
        name="gpe",
        terms=(GPEKineticTerm(), GPEPotentialTerm(g=g)),
        constraint=None,
        charges=(),
    )

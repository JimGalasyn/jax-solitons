"""The Faddeev-Skyrme model: E = E2 + c4 * E4(area form) on a unit S^2 field.

State: n with shape (3, N, N, N) (unit 3-vector per site).

Discretization choices are load-bearing, not stylistic:
  - E2 uses FORWARD differences: central differences have a null space at
    the Nyquist (checkerboard) mode, so a central-difference energy does
    not penalize sub-grid noise (verified in the source codebase: adding a
    checkerboard LOWERS the central-diff energy and raises the forward-diff
    energy ~27x).
  - E4 uses the Berg-Luscher plaquette solid angle (jax_solitons.topology):
    the naive same-index product carries no topological barrier at any
    resolution.
"""

from __future__ import annotations

import dataclasses

import jax.numpy as jnp

from jax_solitons.grid import BoxGrid
from jax_solitons.model import Model
from jax_solitons.topology import area_form_plaquette, hopf_charge

_PLAQUETTES = ((0, 1), (1, 2), (0, 2))


def n_from_Z(Z1, Z2):
    """CP^1 spinor -> unit n-field, n^a = Z^dag sigma^a Z / Z^dag Z.
    Returns shape (3, ...)."""
    norm = jnp.abs(Z1) ** 2 + jnp.abs(Z2) ** 2
    n1 = 2.0 * jnp.real(jnp.conj(Z1) * Z2) / norm
    n2 = 2.0 * jnp.imag(jnp.conj(Z1) * Z2) / norm
    n3 = (jnp.abs(Z1) ** 2 - jnp.abs(Z2) ** 2) / norm
    return jnp.stack([n1, n2, n3])


def _fwd_grads(field, dx):
    """Forward (nearest-neighbour) difference gradient, periodic."""
    gx = (jnp.roll(field, -1, 0) - field) / dx
    gy = (jnp.roll(field, -1, 1) - field) / dx
    gz = (jnp.roll(field, -1, 2) - field) / dx
    return gx, gy, gz


@dataclasses.dataclass(frozen=True)
class E2Term:
    """Dirichlet (sigma-model) term, forward differences."""

    name: str = "E2"

    def __call__(self, n, grid: BoxGrid):
        dx = grid.dx
        e2 = 0.0
        for a in range(3):
            gx, gy, gz = _fwd_grads(n[a], dx)
            e2 = e2 + gx**2 + gy**2 + gz**2
        return jnp.sum(e2) * dx**3


@dataclasses.dataclass(frozen=True)
class E4AreaFormTerm:
    """Skyrme quartic via the geometric area form: c4 * sum_{i<j} F_ij^2."""

    c4: float = 4.0
    name: str = "E4_areaform"

    def __call__(self, n, grid: BoxGrid):
        dx = grid.dx
        e4 = 0.0
        for (i, j) in _PLAQUETTES:
            Om = area_form_plaquette(n, i, j)   # = F_ij * dx^2
            e4 = e4 + (Om / dx**2) ** 2
        return self.c4 * jnp.sum(e4) * dx**3


class S2Constraint:
    """|n| = 1 pointwise: tangent projection + normalization retraction."""

    def project_tangent(self, n, grad):
        ndotg = jnp.sum(n * grad, axis=0, keepdims=True)
        return grad - ndotg * n

    def retract(self, n):
        return n / jnp.sqrt(jnp.sum(n * n, axis=0, keepdims=True))


def faddeev_model(c4: float = 4.0) -> Model:
    """The bare Faddeev-Skyrme model as a Model configuration."""
    return Model(
        name="faddeev",
        terms=(E2Term(), E4AreaFormTerm(c4=c4)),
        constraint=S2Constraint(),
        charges=(hopf_charge,),
    )

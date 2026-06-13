"""Gauged abelian-Higgs (Ginzburg-Landau) model: a complex Higgs field coupled
to a U(1) gauge field, with quantized-flux vortex lines as the topological
solitons.

State: a real ``(5, N, N, N)`` array ``(Re phi, Im phi, A_x, A_y, A_z)`` -- the
complex Higgs ``phi`` and the U(1) vector potential ``A``. A single stacked array
so the autodiff steppers (arrested flow, projected Adam) apply unchanged, as for
the CP^1 spinor frame.

    E = |D phi|^2 + (1/2) sum_{i<j} F_ij^2 + (lambda/4)(|phi|^2 - v^2)^2 ,
    D_i phi = (d_i - i e A_i) phi ,   F_ij = d_i A_j - d_j A_i .

This is the Paper 16 sec.L_2 / nwt-substrate Lagrangian form, so `lambda` is the
same parameter the NWT consumers pass (m_sigma = sqrt(lambda) v).

**Discretization is the COMPACT (link) lattice-gauge form, so U(1) gauge
invariance is EXACT on the lattice** (the engine's exactness contract, P8) --
not merely O(dx^2). With link ``U_i(x) = exp(i e dx A_i(x))``:

  - covariant difference  ``(U_i(x) phi(x+e_i) - phi(x)) / dx``  is
    gauge-COVARIANT (picks up ``e^{i chi(x)}``), so ``|D phi|^2`` is invariant;
  - the plaquette ``U_i(x) U_j(x+e_i) U_i(x+e_j)^* U_j(x)^* = exp(i e dx^2 F_ij)``
    is gauge-INVARIANT, and ``(1 - Re plaquette)/(e^2 dx^4) -> (1/2) F_ij^2``.

BPS reference (Bogomolny 1976): at the self-dual point a winding-``n`` vortex
saturates the Bogomolny bound with line tension ``2 pi v^2 |n|`` and flux
``Phi = 2 pi n / e`` -- the validation targets (Paper 11, Paper 16 sec.L_2,
``nwt_substrate.condensate.line_tension_BPS``).

CONVENTION NOTE (a real Paper 11 / Paper 16 split, flagged for the NWT side):
in this standard ``|D phi|^2 + (1/2) F^2`` normalization the self-dual point is
``lambda = 2 e^2`` (Paper 11; m_higgs^2 = lambda v^2 == m_gauge^2 = 2 e^2 v^2).
Paper 16's text and nwt-substrate report the self-dual coupling as
``lambda = e^2/2``, which corresponds to a *non-standard* gauge-kinetic
normalization (a relative factor in D or F) -- the same physics, different
bookkeeping. `abelian_higgs_model` defaults to the self-consistent ``2 e^2``;
pass `lam=` explicitly to match a specific paper's convention.
"""

from __future__ import annotations

import dataclasses

import jax.numpy as jnp
import numpy as np

from jax_solitons.grid import BoxGrid
from jax_solitons.model import Model

_PLAQUETTES = ((0, 1), (1, 2), (0, 2))


def unpack(state):
    """Real (5, N, N, N) state -> (complex phi (N,N,N), real A (3,N,N,N))."""
    return state[0] + 1j * state[1], state[2:5]


def links(A, e: float, dx: float):
    """Compact U(1) links U_i(x) = exp(i e dx A_i(x)), shape (3, N, N, N)."""
    return jnp.exp(1j * e * dx * A)


@dataclasses.dataclass(frozen=True)
class CovariantKineticTerm:
    """|D phi|^2 with the gauge-covariant link forward difference."""

    e: float = 1.0
    name: str = "ah_kinetic"

    def __call__(self, state, grid: BoxGrid):
        dx = grid.dx
        phi, A = unpack(state)
        U = links(A, self.e, dx)
        e2 = 0.0
        for i in range(3):
            Dphi = (U[i] * jnp.roll(phi, -1, axis=i) - phi) / dx
            e2 = e2 + jnp.abs(Dphi) ** 2
        return jnp.sum(e2) * dx**3


@dataclasses.dataclass(frozen=True)
class MagneticTerm:
    """(1/2) sum_{i<j} F_ij^2 via the gauge-invariant compact plaquette."""

    e: float = 1.0
    name: str = "ah_magnetic"

    def __call__(self, state, grid: BoxGrid):
        dx = grid.dx
        _, A = unpack(state)
        U = links(A, self.e, dx)
        s = 0.0
        for (i, j) in _PLAQUETTES:
            plaq = (U[i] * jnp.roll(U[j], -1, axis=i)
                    * jnp.conj(jnp.roll(U[i], -1, axis=j)) * jnp.conj(U[j]))
            s = s + (1.0 - jnp.real(plaq))
        # (1 - Re plaq) ~ (1/2)(e dx^2 F)^2  ->  (1/2)F^2 = (1-Re plaq)/(e^2 dx^4);
        # times the dx^3 volume element gives the 1/(e^2 dx) prefactor.
        return jnp.sum(s) / (self.e**2 * dx)


@dataclasses.dataclass(frozen=True)
class HiggsPotentialTerm:
    """(lambda/4) (|phi|^2 - v^2)^2 -- the Paper 16 / nwt-substrate normalization
    (Mexican-hat; vacuum |phi| = v, Higgs-mode mass m_sigma = sqrt(lambda) v)."""

    lam: float = 2.0           # self-dual: lambda = 2 e^2 (e=1 -> 2.0); see model
    v: float = 1.0
    name: str = "ah_potential"

    def __call__(self, state, grid: BoxGrid):
        phi, _ = unpack(state)
        return 0.25 * self.lam * jnp.sum(
            (jnp.abs(phi) ** 2 - self.v**2) ** 2) * grid.dx**3


def magnetic_flux(state, grid: BoxGrid, e: float = 1.0):
    """Vortex winding number = (1/2 pi) of the total xy-plaquette flux, averaged
    over z-slices. Integer for a quantized flux tube along z."""
    _, A = unpack(state)
    U = links(A, e, grid.dx)
    plaq = (U[0] * jnp.roll(U[1], -1, axis=0)
            * jnp.conj(jnp.roll(U[0], -1, axis=1)) * jnp.conj(U[1]))
    phase = jnp.angle(plaq)                       # = e dx^2 F_12, in (-pi, pi]
    return jnp.mean(jnp.sum(phase, axis=(0, 1))) / (2.0 * jnp.pi)


def bps_line_tension(v: float = 1.0, n: int = 1) -> float:
    """Bogomolny line tension of a winding-`n` BPS vortex: ``2 pi v^2 |n|``.

    The topological lower bound ``E >= 2 pi v^2 |n|`` (Bogomolny 1976), saturated
    at the self-dual point. Unlike the self-dual *coupling* (which differs by
    convention -- see the module note), this bound is convention-INDEPENDENT:
    Paper 11, Paper 16 sec.L_2, and ``nwt_substrate.condensate.line_tension_BPS``
    all agree on ``2 pi v^2``. The cross-engine gate locks that shared
    normalization down (tests/test_abelian_higgs_oracle.py).
    """
    return 2.0 * float(np.pi) * v**2 * abs(n)


def abelian_higgs_model(e: float = 1.0, lam: float | None = None,
                        v: float = 1.0) -> Model:
    """Gauged abelian-Higgs as a Model configuration.

    `lam` defaults to the self-dual coupling `2 e^2` (this normalization; see the
    module convention note). No hard constraint: the energy is exactly
    gauge-invariant, so descent moves harmlessly along gauge orbits; Coulomb-gauge
    fixing can be added as a retraction later.
    """
    if lam is None:
        lam = 2.0 * e**2
    return Model(
        name="abelian_higgs",
        terms=(CovariantKineticTerm(e=e), MagneticTerm(e=e),
               HiggsPotentialTerm(lam=lam, v=v)),
        constraint=None,
        charges=(lambda s, g: magnetic_flux(s, g, e=e),),
    )


def vortex_seed(grid: BoxGrid, n: int = 1, e: float = 1.0, v: float = 1.0,
                xi: float | None = None, center=(0.0, 0.0)) -> jnp.ndarray:
    """A straight winding-`n` vortex line along z as a (5, N, N, N) state.

    phi = v * tanh(rho/xi) * e^{i n theta}; the gauge field A circulates the line
    with e*A_theta -> n/rho at large rho (so D phi -> 0 in the vacuum), regular
    at the core. Approximate -- relaxation refines it to the true profile.
    """
    if xi is None:
        xi = 0.25 * grid.L / np.sqrt(2)        # ~ healing length, a few cells
    ax = np.asarray(grid.axis(), float)
    X, Y, _ = np.meshgrid(ax, ax, ax, indexing="ij")
    x, y = X - center[0], Y - center[1]
    rho = np.sqrt(x**2 + y**2) + 1e-12
    theta = np.arctan2(y, x)
    phi = v * np.tanh(rho / xi) * np.exp(1j * n * theta)
    g = 1.0 - np.exp(-(rho / xi) ** 2)            # 0 at core -> 1 in vacuum
    A_theta = (n / e) * g / rho
    Ax = -A_theta * np.sin(theta)
    Ay = A_theta * np.cos(theta)
    Az = np.zeros_like(Ax)
    state = np.stack([phi.real, phi.imag, Ax, Ay, Az])
    return jnp.asarray(state, dtype=grid.dtype)

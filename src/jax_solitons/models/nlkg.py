"""Relativistic nonlinear Klein-Gordon (NLKG) complex-scalar vortex model.

    E[Phi] = int  1/2 |grad Phi|^2  +  lam/4 (|Phi|^2 - F0^2)^2  +  m0/2 |Phi|^2

with the relativistic (second-order-in-time) equation of motion

    d_t^2 Phi = lap Phi - lam (|Phi|^2 - F0^2) Phi - m0 Phi  =  -(1/dx^3) dE/dPhi.

This is the relativistic superfluid substrate of Xiong, Good, Guo, Liu & Huang
(PRD 90, 125019, 2014; arXiv:1408.0779). Unlike the Gross-Pitaevskii model
(models.gpe, first-order Schrodinger flow) it is a genuine WAVE equation, so it
carries Kelvin waves, relativistic vortex propagation, sound radiation and
oscillons. Quantized vortices are the phase singularities (|Phi| -> 0, the phase
sigma = arg Phi winds 2*pi around the core); circulation is conserved.

State representation: a REAL array of shape (2, N, N, N) = (Re Phi, Im Phi). The
real view (rather than a native complex array) is the engine-wide convention for
the autodiff steppers (see models.gpe): jax.grad over a real array is unambiguous
(no Wirtinger convention to track), the velocity-Verlet stepper (steppers.verlet)
runs unmodified with constraint=None, and its kinetic_energy(v) = 1/2 sum v.v
already equals 1/2 sum |d_t Phi|^2. The NLKG field is UNGAUGED with no manifold
constraint -- the topology is free (rings/lines reconnect and untie), which is
the point: this is the dynamical tangle substrate, not a protected static knot.

Kinetic stencil: forward differences, the engine-wide convention -- the grad of
1/2 sum |fwd-grad Phi|^2 is exactly the matching discrete Laplacian, so the Verlet
acceleration reproduces the NLKG EOM with no central-difference Nyquist null space.

NOTE (Derrick): bare NLKG has NO stable finite-size 3D soliton -- a |grad|^2 + V
theory shrinks under x -> s*x, so even knotted loops eventually contract and
reconnect (the leftover energy lingers as oscillons). Persistent standing solitons
need exactly one stabilizer: a Skyrme-Faddeev quartic (models.faddeev E4), a gauge
flux + electric sector (models.abelian_higgs / gauged_faddeev), or continuous
injection. This module is the bare, dynamical substrate on purpose; compose an E4
term onto `terms` to get standing solitons.
"""

from __future__ import annotations

import dataclasses

import jax.numpy as jnp
import numpy as np

from jax_solitons.grid import BoxGrid
from jax_solitons.model import Model


def to_complex(state):
    """(2, N, N, N) real view -> complex (N, N, N) field Phi."""
    return state[0] + 1j * state[1]


def from_complex(phi):
    """Complex (N, N, N) field -> (2, N, N, N) real view, default dtype."""
    return jnp.stack([jnp.real(phi), jnp.imag(phi)])


def _fwd_grads_c(field, dx):
    gx = (jnp.roll(field, -1, 0) - field) / dx
    gy = (jnp.roll(field, -1, 1) - field) / dx
    gz = (jnp.roll(field, -1, 2) - field) / dx
    return gx, gy, gz


@dataclasses.dataclass(frozen=True)
class NLKGKineticTerm:
    name: str = "nlkg_kinetic"

    def __call__(self, state, grid: BoxGrid):
        phi = to_complex(state)
        gx, gy, gz = _fwd_grads_c(phi, grid.dx)
        e = jnp.abs(gx) ** 2 + jnp.abs(gy) ** 2 + jnp.abs(gz) ** 2
        return 0.5 * jnp.sum(e) * grid.dx**3


@dataclasses.dataclass(frozen=True)
class NLKGPotentialTerm:
    """Mexican-hat lam/4 (|Phi|^2 - F0^2)^2; vacuum manifold |Phi| = F0."""

    lam: float = 1.0
    f0sq: float = 1.0
    name: str = "nlkg_potential"

    def __call__(self, state, grid: BoxGrid):
        rho = state[0] ** 2 + state[1] ** 2
        return 0.25 * self.lam * jnp.sum((rho - self.f0sq) ** 2) * grid.dx**3


@dataclasses.dataclass(frozen=True)
class NLKGMassTerm:
    """Explicit mass m0/2 |Phi|^2 (symmetry-breaking; usually 0)."""

    m0: float = 0.0
    name: str = "nlkg_mass"

    def __call__(self, state, grid: BoxGrid):
        rho = state[0] ** 2 + state[1] ** 2
        return 0.5 * self.m0 * jnp.sum(rho) * grid.dx**3


def nlkg_model(lam: float = 1.0, f0sq: float = 1.0, m0: float = 0.0) -> Model:
    """The relativistic NLKG as a Model (real (2,N,N,N) state, no constraint).

    Evolve with steppers.verlet (real-time wave dynamics). The mass term is
    included only when m0 != 0 so the common m0=0 case carries no dead term."""
    terms = [NLKGKineticTerm(), NLKGPotentialTerm(lam=lam, f0sq=f0sq)]
    if m0 != 0.0:
        terms.append(NLKGMassTerm(m0=m0))
    return Model(name="nlkg", terms=tuple(terms), constraint=None, charges=())


# --------------------------------------------------------------------------
# Seeds: quantized vortex rings and lines (multiplied onto the |Phi|=F0 vacuum).
# Returned as the real (2, N, N, N) view at the grid dtype.
# --------------------------------------------------------------------------


def _ring_factor(grid: BoxGrid, R, xi, center, axis="z", sign=1):
    """Unit-amplitude vortex-ring factor: |.| = tanh(d/xi) dips on the core circle
    of radius R, phase = sign*atan2 winds 2*pi around the core. Complex (N,N,N)."""
    X, Y, Z = (np.asarray(c) for c in grid.coords())
    cx, cy, cz = center
    if axis == "z":
        a1, a2, a3 = X - cx, Y - cy, Z - cz
    elif axis == "x":
        a1, a2, a3 = Y - cy, Z - cz, X - cx
    else:
        a1, a2, a3 = Z - cz, X - cx, Y - cy
    rho = np.sqrt(a1**2 + a2**2)
    d = np.sqrt((rho - R) ** 2 + a3**2)
    phase = sign * np.arctan2(a3, rho - R)
    return np.tanh(d / xi) * np.exp(1j * phase)


def _line_factor(grid: BoxGrid, xi, axis="z", offset=(0.0, 0.0), sign=1):
    """Straight vortex line along `axis`, |.| = tanh(r_perp/xi), phase winds 2*pi.
    A persistent box-spanning thread; reconnection shows up as a Kelvin-wave kink."""
    X, Y, Z = (np.asarray(c) for c in grid.coords())
    if axis == "z":
        a, b = X - offset[0], Y - offset[1]
    elif axis == "x":
        a, b = Y - offset[0], Z - offset[1]
    else:
        a, b = Z - offset[0], X - offset[1]
    r = np.sqrt(a**2 + b**2)
    phase = sign * np.arctan2(b, a)
    return np.tanh(r / xi) * np.exp(1j * phase)


def vortex_seed(grid: BoxGrid, rings=(), lines=(), f0: float = 1.0):
    """Build the (2, N, N, N) initial field: |Phi| = f0 vacuum with the given
    vortex `rings` and `lines` multiplied in. Each entry is a dict of the factor
    kwargs, e.g. rings=[dict(R=6.0, xi=1.0, center=(0,0,-6), axis="z", sign=1)].
    """
    phi = np.full(grid.coords()[0].shape, float(f0), dtype=np.complex128)
    for r in rings:
        phi = phi * _ring_factor(grid, **r)
    for ln in lines:
        phi = phi * _line_factor(grid, **ln)
    return jnp.stack(
        [jnp.asarray(phi.real, dtype=grid.dtype), jnp.asarray(phi.imag, dtype=grid.dtype)]
    )


# --------------------------------------------------------------------------
# Circulation: the Xiong vortex detector. Gamma = oint grad(sigma).dl, quantized
# to 2*pi * (enclosed winding). Computed as the sum of wrapped phase increments
# around a rectangular lattice loop -- fully discrete, exact on the lattice.
# --------------------------------------------------------------------------


def _wrap(d):
    """Wrap a phase difference into (-pi, pi]."""
    return (d + jnp.pi) % (2.0 * jnp.pi) - jnp.pi


def circulation(state, grid: BoxGrid, axis: str = "z", center=(0.0, 0.0),
                half_cells: int = 6, plane_index: int | None = None) -> float:
    """oint grad(sigma).dl around a square loop perpendicular to `axis`, centred
    (in the transverse plane) on `center`, with half-width `half_cells` lattice
    cells, taken in the mid-plane (or `plane_index`). Returns ~ 2*pi * n for n
    quanta of circulation threading the loop; the sign follows the winding."""
    phi = to_complex(jnp.asarray(state))
    perp = {"z": 2, "x": 0, "y": 1}[axis]
    if plane_index is None:
        plane_index = grid.N // 2
    s = np.angle(np.moveaxis(np.asarray(phi), perp, 0)[plane_index])  # plane (Na, Nb)
    dx = grid.dx
    ic = int(round(center[0] / dx + grid.N / 2))
    jc = int(round(center[1] / dx + grid.N / 2))
    i0, i1 = ic - half_cells, ic + half_cells
    j0, j1 = jc - half_cells, jc + half_cells

    def ph(i, j):
        return s[i % grid.N, j % grid.N]

    tot = 0.0                              # counter-clockwise four-edge loop
    for i in range(i0, i1):                # +a edge at j0
        tot += _wrap_np(ph(i + 1, j0) - ph(i, j0))
    for j in range(j0, j1):                # +b edge at i1
        tot += _wrap_np(ph(i1, j + 1) - ph(i1, j))
    for i in range(i1, i0, -1):            # -a edge at j1
        tot += _wrap_np(ph(i - 1, j1) - ph(i, j1))
    for j in range(j1, j0, -1):            # -b edge at i0
        tot += _wrap_np(ph(i0, j - 1) - ph(i0, j))
    return float(tot)


def _wrap_np(d):
    return (d + np.pi) % (2.0 * np.pi) - np.pi

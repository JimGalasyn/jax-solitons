"""Gauged Faddeev-Skyrme-Higgs: the L_2 + L_3 sectors of one field theory.

A single complex SU(2) doublet ``psi in C^2`` carries all three roles of the
Paper 16 NWT Lagrangian (reading A: the unit field is *slaved* to ``psi``):

  - its **modulus** ``|psi|`` is the Higgs field (Mexican-hat vacuum |psi| = v);
  - its **overall U(1) phase** is gauged by ``A`` -- quantized-flux vortex lines;
  - its **Bloch direction** ``n^a = psi^dag sigma^a psi / (psi^dag psi)`` is the
    Skyrme-Faddeev unit field -- knotted Hopf solitons.

State: real ``(7, N, N, N)`` = ``(Re psi1, Im psi1, Re psi2, Im psi2, A_x, A_y,
A_z)``. One stacked array so the autodiff steppers apply unchanged (as for the
CP^1 spinor frame and the abelian-Higgs model).

    E = |D_i psi|^2  +  (1/2) sum_{i<j} F_ij^2  +  (lam/4)(|psi|^2 - v^2)^2
        +  c2 (d_i n^a)^2  +  c4 sum_{i<j} F_ij[n]^2  +  theta Q_H[n] ,
    D_i psi = (U_i psi(x+e_i) - psi(x)) / dx ,   U_i = exp(i e dx A_i) .

**The gauge couples ONLY to the overall phase.** ``n^a`` is a U(1)-invariant
bilinear (the gauged phase cancels: ``n^a[e^{i chi} psi] = n^a[psi]``), so
``d_i n^a`` is *already* gauge-invariant -- there is no minimal coupling of ``A``
to the Skyrme sector, exactly as Paper 16 writes ``d_i n``, NOT ``D_i n``. The
L_2 / L_3 coupling is therefore not an explicit cross-term: it is the
**field-sharing** ``n = n[psi]`` (so relaxing psi, A back-reacts on n and vice
versa) plus the shared knotted-tube geometry.

Discretization is the **compact (link) lattice-gauge form**, so U(1) gauge
invariance is EXACT on the lattice (P8), reusing the abelian-Higgs plaquette
machinery. There is no hard constraint: the modulus is dynamical (softly fixed
to v by the potential) and descent moves harmlessly along gauge orbits.

NOTE (core regularization): with a *free* modulus, ``n`` is ill-defined where
``psi -> 0`` (a Higgs vortex core: 0/0). On the lattice we add a tiny ``eps`` to
``psi^dag psi`` so ``n`` and its gradients stay finite; the Skyrme energy of a
psi-zero is then large-but-finite (~log dx). Freezing |psi| = v (a gauged CP^1
+ Skyrme model) removes the singularity entirely and is the numerically robust
variant for deep relaxation -- a follow-up.
"""

from __future__ import annotations

import dataclasses

import jax.numpy as jnp
import numpy as np

from jax_solitons.grid import BoxGrid
from jax_solitons.model import Model
from jax_solitons.models.abelian_higgs import _PLAQUETTES, links, magnetic_flux
from jax_solitons.models.faddeev import E2Term, E4AreaFormTerm
from jax_solitons.topology import hopf_charge

_NEPS = 1e-12


def unpack(state):
    """Real (7, N, N, N) -> (psi1, psi2 complex (N,N,N), A real (3,N,N,N))."""
    return state[0] + 1j * state[1], state[2] + 1j * state[3], state[4:7]


def n_from_doublet(state, eps: float = _NEPS):
    """Unit Skyrme field n^a = psi^dag sigma^a psi / (psi^dag psi), with the
    denominator regularized by `eps` so it stays finite at psi-zeros (vortex
    cores). Shape (3, N, N, N). In the bulk |psi|^2 ~ v^2 >> eps, so n is unit
    to machine precision; only on the eps-shell at a core is |n| < 1."""
    psi1, psi2, _ = unpack(state)
    denom = jnp.abs(psi1) ** 2 + jnp.abs(psi2) ** 2 + eps
    n1 = 2.0 * jnp.real(jnp.conj(psi1) * psi2) / denom
    n2 = 2.0 * jnp.imag(jnp.conj(psi1) * psi2) / denom
    n3 = (jnp.abs(psi1) ** 2 - jnp.abs(psi2) ** 2) / denom
    return jnp.stack([n1, n2, n3])


@dataclasses.dataclass(frozen=True)
class DoubletCovariantKineticTerm:
    """|D_i psi|^2 for the C^2 doublet, gauge-covariant link forward difference
    (the same link U_i acts on BOTH components -- the overall U(1) phase)."""

    e: float = 1.0
    name: str = "gf_kinetic"

    def __call__(self, state, grid: BoxGrid):
        dx = grid.dx
        psi1, psi2, A = unpack(state)
        U = links(A, self.e, dx)
        acc = 0.0
        for i in range(3):
            for psi in (psi1, psi2):
                Dpsi = (U[i] * jnp.roll(psi, -1, axis=i) - psi) / dx
                acc = acc + jnp.abs(Dpsi) ** 2
        return jnp.sum(acc) * dx**3


@dataclasses.dataclass(frozen=True)
class DoubletMagneticTerm:
    """(1/2) sum_{i<j} F_ij^2 via the gauge-invariant compact plaquette (the
    abelian-Higgs magnetic term, on the (7,...) layout's A block)."""

    e: float = 1.0
    name: str = "gf_magnetic"

    def __call__(self, state, grid: BoxGrid):
        dx = grid.dx
        _, _, A = unpack(state)
        U = links(A, self.e, dx)
        s = 0.0
        for (i, j) in _PLAQUETTES:
            plaq = (U[i] * jnp.roll(U[j], -1, axis=i)
                    * jnp.conj(jnp.roll(U[i], -1, axis=j)) * jnp.conj(U[j]))
            s = s + (1.0 - jnp.real(plaq))
        return jnp.sum(s) / (self.e**2 * dx)


@dataclasses.dataclass(frozen=True)
class DoubletHiggsPotentialTerm:
    """(lam/4) (|psi|^2 - v^2)^2 on the doublet modulus |psi|^2 = psi^dag psi."""

    lam: float = 2.0
    v: float = 1.0
    name: str = "gf_potential"

    def __call__(self, state, grid: BoxGrid):
        psi1, psi2, _ = unpack(state)
        mod2 = jnp.abs(psi1) ** 2 + jnp.abs(psi2) ** 2
        return 0.25 * self.lam * jnp.sum((mod2 - self.v**2) ** 2) * grid.dx**3


@dataclasses.dataclass(frozen=True)
class SkyrmeE2Term:
    """c2 (d_i n^a)^2 on the derived (gauge-invariant) unit field n[psi]."""

    c2: float = 1.0
    eps: float = _NEPS
    name: str = "gf_skyrme_e2"

    def __call__(self, state, grid: BoxGrid):
        return self.c2 * E2Term()(n_from_doublet(state, self.eps), grid)


@dataclasses.dataclass(frozen=True)
class SkyrmeE4Term:
    """c4 sum_{i<j} F_ij[n]^2 (Skyrme quartic) via the area form on n[psi]."""

    c4: float = 4.0
    eps: float = _NEPS
    name: str = "gf_skyrme_e4"

    def __call__(self, state, grid: BoxGrid):
        return E4AreaFormTerm(c4=self.c4)(n_from_doublet(state, self.eps), grid)


@dataclasses.dataclass(frozen=True)
class HopfThetaTerm:
    """theta Q_H[n] -- scale-invariant topological term (CPT: theta = 0 or pi)."""

    theta: float = 0.0
    eps: float = _NEPS
    name: str = "gf_hopf_theta"

    def __call__(self, state, grid: BoxGrid):
        return self.theta * hopf_charge(n_from_doublet(state, self.eps), grid)


def hopf_charge_doublet(state, grid: BoxGrid):
    """Hopf charge Q_H[n[psi]] of the coupled state (L_3 topological sector)."""
    return hopf_charge(n_from_doublet(state), grid)


def aspect_ratio(state, grid: BoxGrid, p: int, q: int, v: float = 1.0):
    """Estimate the knot aspect ratio kappa = R / a0 (Paper 16 sec.L_3 target
    pi^2 ~ 9.87) from a relaxed coupled state, via Higgs-core moments.

    The core indicator ``w = max(0, 1 - |psi|^2/v^2)`` concentrates on the Higgs
    flux tube (w -> 1 where psi -> 0, -> 0 in the bulk). Then:
      - major radius   R  = <rho_cyl>_w           (w-weighted cylindrical radius)
      - core volume    V  = integral of w
      - tube length    L  = arc length of the T(p,q) curve at radius (R, R/2)
      - core radius    a0 = sqrt(V / (pi L))       (V ~ L * pi a0^2)
    Returns (kappa, R, a0). A moment estimate at the paper's own altitude ("an
    O(10) number consistent with pi^2 within ~15%"), not a precise core trace.
    """
    psi1, psi2, _ = unpack(state)
    mod2 = jnp.abs(psi1) ** 2 + jnp.abs(psi2) ** 2
    w = jnp.clip(1.0 - mod2 / v**2, 0.0, 1.0)
    X, Y, _Z = grid.coords()
    rho_cyl = jnp.sqrt(jnp.asarray(X) ** 2 + jnp.asarray(Y) ** 2)
    wsum = jnp.sum(w)
    R = float(jnp.sum(w * rho_cyl) / wsum)
    V = float(wsum) * grid.dx**3
    # arc length of T(p,q) at major radius R, minor R/2 (the seed's b = 0.4 R
    # ~ R/2); length is insensitive to the exact minor radius at this altitude.
    s = np.linspace(0.0, 2.0 * np.pi, 4000, endpoint=False)
    b = 0.5 * R
    cx = (R + b * np.cos(q * s)) * np.cos(p * s)
    cy = (R + b * np.cos(q * s)) * np.sin(p * s)
    cz = b * np.sin(q * s)
    curve = np.stack([cx, cy, cz], axis=1)
    L = float(np.sum(np.linalg.norm(
        np.diff(curve, axis=0, append=curve[:1]), axis=1)))
    a0 = float(np.sqrt(V / (np.pi * L)))
    return R / a0, R, a0


def gauged_faddeev_model(e: float = 1.0, lam: float | None = None, v: float = 1.0,
                         c2: float = 1.0, c4: float = 4.0,
                         theta: float = 0.0) -> Model:
    """Coupled L_2 + L_3 gauged Faddeev-Skyrme-Higgs model (Paper 16 L_NWT,
    reading A: n slaved to a single C^2 doublet).

    `lam` defaults to the self-dual coupling 2 e^2 (this normalization; see the
    abelian_higgs convention note). `c2`/`c4` are the Skyrme stiffness and
    quartic (faddeev convention). No hard constraint: the energy is exactly
    gauge-invariant and the modulus is dynamical.
    """
    if e == 0:
        raise ValueError(
            "e (gauge coupling) must be nonzero: the compact-link form has no "
            "e->0 limit (the magnetic term scales as 1/e^2).")
    if lam is None:
        lam = 2.0 * e**2
    terms = [
        DoubletCovariantKineticTerm(e=e),
        DoubletMagneticTerm(e=e),
        DoubletHiggsPotentialTerm(lam=lam, v=v),
        SkyrmeE2Term(c2=c2),
        SkyrmeE4Term(c4=c4),
    ]
    if theta != 0.0:
        terms.append(HopfThetaTerm(theta=theta))
    return Model(
        name="gauged_faddeev",
        terms=tuple(terms),
        constraint=None,
        charges=(lambda s, g: magnetic_flux(s[2:7], g, e=e), hopf_charge_doublet),
    )

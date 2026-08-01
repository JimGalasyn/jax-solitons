"""The SU(2)/S^3 Skyrme model: baryon-number solitons (B=1 nucleon, B=2
deuteron, B=4 alpha, ...).

This complements ``models.faddeev`` (the O(3)/CP^1 field, Hopf charge
pi_3(S^2)) with the OTHER pi_3: an SU(2) field whose solitons are labelled by
the baryon number B = deg(U) in pi_3(S^3) = Z -- the nuclear solitons. It is
the calibration ground truth for the multi-soliton *binding* problem (the
Battye-Sutcliffe B=2 deuteron has a published binding), so our binding methods
(rigid composition, soft-pin polarisation relaxer) can be checked against a
system with known answers before we trust them on the NWT hopfion carrier.

State layout
------------
A real unit 4-vector field ``phi = (phi0, phi1, phi2, phi3)``, ``|phi| = 1``,
shape ``(4, N, N, N)`` (a leading batch axis may be added via vmap, R2). It
represents ``U = phi0 * 1 + i phi_a sigma_a`` in SU(2) ~ S^3.

Energy (the O(4) sigma-model form of the standard Skyrme functional)
-------------------------------------------------------------------
With ``g_i = d_i phi`` (FORWARD differences -- the same load-bearing stencil
discipline as ``models.faddeev``; central differences do not penalise the
checkerboard mode) and the strain metric ``M_ij = g_i . g_j``:

  - L2 (``c2``):  sum_i |g_i|^2                       = -1/2 Tr(L_i L_i)
  - L4 (``c4``):  sum_{i<j} |g_i|^2|g_j|^2 - (g_i.g_j)^2
                  = sum_{i<j} (M_ii M_jj - M_ij^2)    = -1/16 Tr([L_i,L_j])^2
  - L0 (``c0``, ``m_pi``, optional):  1 - phi0        = 1 - 1/2 Tr U

L4 is the second symmetric polynomial of the eigenvalues of M -- the Skyrme
quartic that (as in Faddeev) supplies Derrick stability. The massless model
obeys the Faddeev-Bogomolny bound ``E >= 12 pi^2 sqrt(c2 c4) |B|`` (AM-GM on
the strain eigenvalues); the B=1 hedgehog sits ~1.23x above it.

Baryon number (the exact topological charge)
--------------------------------------------
``baryon_charge`` is computed as the *degree of the piecewise-linear map*
T^3 -> S^3 via a regular-value preimage count (signed coverings of a generic
point), NOT the naive ``eps^{ijk} Tr(L_iL_jL_k)`` (which is not integer-
quantised at finite resolution -- the same failure the area form fixes for the
Hopf index). The count is an exact integer (a finite sum of +-1) and uses only
4x4 determinants, so it needs no spherical-tetrahedron volume (which has no
elementary closed form -- it is a Murakami-Yano dilogarithm). It is *not*
differentiable, but B is a diagnostic, not a loss term: the smooth O(4) Skyrme
quartic above is what carries the descent barrier (exactly as the area-form E4
does for Faddeev, with ``hopf_charge`` the separate diagnostic).
"""

from __future__ import annotations

import dataclasses

import jax.numpy as jnp
import numpy as np

from jax_solitons.grid import BoxGrid
from jax_solitons.model import Model
from jax_solitons.models.faddeev import S2Constraint


def _phi_grads(phi, dx):
    """Forward (nearest-neighbour) difference gradients of the 4-vector field
    along the three spatial axes; each shape (4, N, N, N), periodic."""
    gx = (jnp.roll(phi, -1, axis=1) - phi) / dx
    gy = (jnp.roll(phi, -1, axis=2) - phi) / dx
    gz = (jnp.roll(phi, -1, axis=3) - phi) / dx
    return gx, gy, gz


# --- energy terms (O(4) sigma model) ---------------------------------------

@dataclasses.dataclass(frozen=True)
class E2O4Term:
    """L2 / Dirichlet term: c2 * sum_i |d_i phi|^2, forward differences."""

    c2: float = 1.0
    name: str = "E2_O4"

    def __call__(self, phi, grid: BoxGrid):
        dx = grid.dx
        gx, gy, gz = _phi_grads(phi, dx)
        e2 = jnp.sum(gx * gx + gy * gy + gz * gz, axis=0)
        return self.c2 * jnp.sum(e2) * dx**3


@dataclasses.dataclass(frozen=True)
class E4SkyrmeTerm:
    """L4 / Skyrme quartic as the O(4) invariant
    c4 * sum_{i<j} (M_ii M_jj - M_ij^2),  M_ij = d_i phi . d_j phi."""

    c4: float = 1.0
    name: str = "E4_skyrme"

    def __call__(self, phi, grid: BoxGrid):
        dx = grid.dx
        gx, gy, gz = _phi_grads(phi, dx)
        Mxx = jnp.sum(gx * gx, axis=0)
        Myy = jnp.sum(gy * gy, axis=0)
        Mzz = jnp.sum(gz * gz, axis=0)
        Mxy = jnp.sum(gx * gy, axis=0)
        Mxz = jnp.sum(gx * gz, axis=0)
        Myz = jnp.sum(gy * gz, axis=0)
        e4 = (Mxx * Myy - Mxy**2) + (Mxx * Mzz - Mxz**2) + (Myy * Mzz - Myz**2)
        return self.c4 * jnp.sum(e4) * dx**3


@dataclasses.dataclass(frozen=True)
class E0MassTerm:
    """L0 / pion-mass term: c0 * m_pi^2 * (1 - phi0) = c0 m_pi^2 (1 - 1/2 Tr U)."""

    c0: float = 1.0
    m_pi: float = 0.0
    name: str = "E0_mass"

    def __call__(self, phi, grid: BoxGrid):
        return self.c0 * self.m_pi**2 * jnp.sum(1.0 - phi[0]) * grid.dx**3


class S3Constraint(S2Constraint):
    """|phi| = 1 pointwise on the real 4-vector (the same unit-norm tangent
    projection / normalisation retraction as S^2 and CP^1, one component
    higher -- the projector algebra is dimension-agnostic, axis 0)."""


def skyrme_energy_density(phi, grid: BoxGrid, c2: float = 1.0, c4: float = 1.0,
                          m_pi: float = 0.0, c0: float = 1.0):
    """Pointwise energy density (N, N, N); integrates to the model energy
    (sum * dx^3). The soliton-localisation field for separation scans and the
    binding cross-check, the analog of ``faddeev_energy_density``."""
    dx = grid.dx
    gx, gy, gz = _phi_grads(phi, dx)
    Mxx = jnp.sum(gx * gx, axis=0)
    Myy = jnp.sum(gy * gy, axis=0)
    Mzz = jnp.sum(gz * gz, axis=0)
    Mxy = jnp.sum(gx * gy, axis=0)
    Mxz = jnp.sum(gx * gz, axis=0)
    Myz = jnp.sum(gy * gz, axis=0)
    e2 = c2 * (Mxx + Myy + Mzz)
    e4 = c4 * ((Mxx * Myy - Mxy**2) + (Mxx * Mzz - Mxz**2)
               + (Myy * Mzz - Myz**2))
    e0 = c0 * m_pi**2 * (1.0 - phi[0])
    return e2 + e4 + e0


def skyrme_bound(B, c2: float = 1.0, c4: float = 1.0) -> float:
    """Faddeev-Bogomolny lower bound on the (massless) energy:
    E >= 12 pi^2 sqrt(c2 c4) |B|. The B=1 minimiser sits ~1.232x above it."""
    return 12.0 * np.pi**2 * float(np.sqrt(c2 * c4)) * abs(B)


# --- exact baryon number = degree of the PL map T^3 -> S^3 ------------------
#
# Kuhn (Freudenthal) decomposition of each lattice cube into 6 tetrahedra, all
# sharing the long diagonal 000 -- 111; the two middle vertices follow a
# permutation of the three axes, and the tetrahedron's source orientation is
# that permutation's parity. The image of a tetrahedron's 4 vertices is 4
# points on S^3; a generic target point p is covered (the geodesic image
# simplex contains p) iff solving Phi @ lam = p gives all lam > 0, i.e. (by
# Cramer, division-free) every column-replaced determinant shares the sign of
# det Phi. The signed count over all cubes and tetrahedra is the degree --
# independent of the regular value p (the exactness statement) and an exact
# integer. We pin the global orientation against the B=1 hedgehog (= +1).

_ORIENT = -1.0   # global sign, pinned so the hedgehog reads B = +1

# (parity, v1_offset, v2_offset); v0 = (0,0,0), v3 = (1,1,1) for all six.
_KUHN_TETS = (
    (+1, (1, 0, 0), (1, 1, 0)),   # perm (x,y,z)
    (-1, (1, 0, 0), (1, 0, 1)),   # perm (x,z,y)
    (-1, (0, 1, 0), (1, 1, 0)),   # perm (y,x,z)
    (+1, (0, 1, 0), (0, 1, 1)),   # perm (y,z,x)
    (+1, (0, 0, 1), (1, 0, 1)),   # perm (z,x,y)
    (-1, (0, 0, 1), (0, 1, 1)),   # perm (z,y,x)
)

# Generic regular values (unit 4-vectors away from the vacuum (1,0,0,0) and
# from the poles, so |det| stays off zero except on degenerate simplices).
_PROBE_POINTS = tuple(
    tuple(np.asarray(v, np.float64) / np.linalg.norm(v))
    for v in (
        (0.31, -0.53, 0.67, -0.41),
        (-0.23, 0.71, 0.13, 0.59),
        (0.47, 0.19, -0.61, 0.55),
    )
)


def _corner(phi, off):
    """Cube corner field phi(x + off), periodic; off is a 0/1 triple."""
    di, dj, dk = off
    return jnp.roll(phi, shift=(-di, -dj, -dk), axis=(1, 2, 3))


def _degree_at(phi, p):
    """Signed covering count of the regular value p (a 4-vector); the degree
    of the PL map for this probe point. Exact integer for any closed field."""
    pj = jnp.asarray(p, dtype=phi.dtype)
    v0 = phi                                   # offset (0,0,0)
    v3 = _corner(phi, (1, 1, 1))
    deg = 0.0
    for parity, o1, o2 in _KUHN_TETS:
        v1 = _corner(phi, o1)
        v2 = _corner(phi, o2)
        # Phi columns = the 4 image vertices; shape (..., 4, 4).
        Phi = jnp.stack([jnp.moveaxis(v, 0, -1) for v in (v0, v1, v2, v3)],
                        axis=-1)
        d = jnp.linalg.det(Phi)
        sd = jnp.sign(d)
        same = jnp.abs(d) > 1e-12
        for m in range(4):
            Phim = Phi.at[..., :, m].set(pj)
            same = same & (jnp.sign(jnp.linalg.det(Phim)) == sd)
        deg = deg + parity * jnp.sum(jnp.where(same, sd, 0.0))
    return _ORIENT * float(deg)


def baryon_charge(state, grid: BoxGrid):
    """Exact baryon number B = deg(U: T^3 -> S^3) via a regular-value preimage
    count. Returns the (integer-valued) median over a few generic probe points
    -- their agreement is the topological-invariance statement. Diagnostic,
    not differentiable; the energy's Skyrme quartic is the descent barrier."""
    vals = sorted(round(_degree_at(state, p)) for p in _PROBE_POINTS)
    return float(vals[len(vals) // 2])


# --- the model -------------------------------------------------------------

def skyrme_model(c2: float = 1.0, c4: float = 1.0, m_pi: float = 0.0,
                 c0: float = 1.0) -> Model:
    """The SU(2) Skyrme model as a Model configuration: terms (L2 + L4 [+ L0])
    + S^3 unit constraint + baryon-number charge. Steppers, campaign,
    checkpoint, batch are all unchanged (P1). ``m_pi=0`` (default) is the
    massless model -- enough for topology and structure; a nonzero pion mass
    adds the L0 term (needed for B>=2 binding magnitudes to match nuclei)."""
    terms: list = [E2O4Term(c2=c2), E4SkyrmeTerm(c4=c4)]
    if m_pi != 0.0:
        terms.append(E0MassTerm(c0=c0, m_pi=m_pi))
    return Model(
        name="skyrme",
        terms=tuple(terms),
        constraint=S3Constraint(),
        charges=(baryon_charge,),
    )

"""Tier-1 validation for the SU(2)/S^3 Skyrme model (models.skyrme).

Exact-by-construction identities: the regular-value preimage count is an
integer topological invariant (probe-independent), the rational-map seeds
carry their stated baryon number, and the O(4) sigma-model energy is invariant
under lattice translation and global target rotation. A violation is a bug,
never a tolerance question -- the analog of test_exact.py for the Hopf sector.
See docs/SKYRME.md and VALIDATION.md.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from jax_solitons import BoxGrid
from jax_solitons.models.skyrme import (
    S3Constraint,
    baryon_charge,
    skyrme_bound,
    skyrme_model,
)
from jax_solitons.models.skyrme import _PROBE_POINTS, _degree_at
from jax_solitons.seeds import skyrmion_hedgehog, skyrmion_rational_map

GRID = BoxGrid(N=24, L=8.0)
MODEL = skyrme_model(c2=1.0, c4=1.0)


def _random_smooth_phi(grid: BoxGrid, seed: int, corr=1.2):
    """Band-limited random unit 4-vector field (an arbitrary smooth map
    T^3 -> S^3; adversarial input, nothing like the seeds)."""
    rng = np.random.default_rng(seed)
    _, _, _, K2 = grid.k_vectors()
    filt = np.exp(-np.asarray(K2) * corr**2)
    comps = []
    for _ in range(4):
        w = rng.standard_normal((grid.N,) * 3)
        comps.append(np.real(np.fft.ifftn(np.fft.fftn(w) * filt)))
    phi = np.stack(comps) + np.array([0.2, 0.0, 0.1, 0.05])[:, None, None, None]
    phi = phi / np.sqrt((phi**2).sum(axis=0, keepdims=True))
    return jnp.asarray(phi, dtype=grid.dtype)


# --- baryon number = degree of the PL map T^3 -> S^3 -----------------------

@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_baryon_charge_integer_and_probe_independent(seed):
    """The signed preimage count of a regular value is an EXACT integer and
    is the SAME for every (generic) probe point -- the topological-invariance
    statement, on arbitrary smooth fields, not just solitons. The naive
    eps^{ijk} Tr(L_iL_jL_k) has no such quantization at finite resolution."""
    phi = _random_smooth_phi(GRID, seed)
    degs = [_degree_at(phi, p) for p in _PROBE_POINTS]
    for d in degs:
        assert abs(d - round(d)) < 1e-9, f"degree not integer: {d}"
    assert len({round(d) for d in degs}) == 1, \
        f"degree depends on probe point (not a regular value?): {degs}"


def test_baryon_charge_hedgehog_is_plus_one():
    """The B=1 hedgehog has baryon number exactly +1 (this pins the global
    orientation convention of the count)."""
    assert baryon_charge(skyrmion_hedgehog(GRID), GRID) == 1.0


@pytest.mark.parametrize("B", [1, 2, 3, 4])
def test_baryon_charge_rational_map_degree(B):
    """The Houghton-Manton-Sutcliffe seed R(z)=z^B carries baryon number B."""
    phi = skyrmion_rational_map(GRID, B=B)
    assert baryon_charge(phi, GRID) == float(B)


def test_baryon_charge_translation_invariant():
    """Baryon number is invariant under periodic lattice translation."""
    phi = skyrmion_rational_map(GRID, B=2)
    B0 = baryon_charge(phi, GRID)
    for shift, axes in (((3, 5, 7), (1, 2, 3)), ((11, 2, 1), (1, 2, 3))):
        assert baryon_charge(jnp.roll(phi, shift, axis=axes), GRID) == B0


# --- energy lattice / target symmetries ------------------------------------

def test_energy_translation_invariant():
    """Periodic lattice translations leave E exactly invariant (same summands;
    tolerance is summation-order rounding only)."""
    phi = _random_smooth_phi(GRID, 7)
    E0 = float(MODEL.energy(phi, GRID))
    for shift, axis in ((5, 1), (11, 2), (1, 3)):
        Et = float(MODEL.energy(jnp.roll(phi, shift, axis=axis), GRID))
        assert np.isclose(Et, E0, rtol=1e-5)


def test_energy_global_o4_invariant():
    """Global SO(4) rotations of the SU(2) target leave the energy invariant
    (it is built from O(4) invariants d_i phi . d_j phi). The block 90-degree
    rotation is exact in floats; a generic QR rotation checks the same identity
    off the lattice-friendly axes."""
    phi = _random_smooth_phi(GRID, 8)
    E0 = float(MODEL.energy(phi, GRID))

    # exact block rotation in SO(4): (a,b,c,d) -> (b,-a,d,-c)
    phi90 = jnp.stack([phi[1], -phi[0], phi[3], -phi[2]])
    assert np.isclose(float(MODEL.energy(phi90, GRID)), E0, rtol=1e-6)

    rng = np.random.default_rng(3)
    Q, _ = np.linalg.qr(rng.standard_normal((4, 4)))
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1.0                          # force det +1 (SO(4))
    phiQ = jnp.einsum("ab,bxyz->axyz", jnp.asarray(Q, dtype=phi.dtype), phi)
    assert np.isclose(float(MODEL.energy(phiQ, GRID)), E0, rtol=1e-3)


def test_baryon_charge_so4_invariant():
    """Baryon number is invariant under an orientation-preserving (SO(4))
    rotation of the target S^3."""
    phi = skyrmion_rational_map(GRID, B=2)
    phi90 = jnp.stack([phi[1], -phi[0], phi[3], -phi[2]])
    assert baryon_charge(phi90, GRID) == baryon_charge(phi, GRID)


# --- constraint-manifold algebra (d = 4) -----------------------------------

def test_s3_constraint_algebra():
    """retract is idempotent and lands on S^3; project_tangent is pointwise
    orthogonal to the state and idempotent (the shared unit-norm projector,
    one component above CP^1)."""
    constraint = S3Constraint()
    rng = np.random.default_rng(11)
    x = jnp.asarray(rng.standard_normal((4, 12, 12, 12)))
    g = jnp.asarray(rng.standard_normal((4, 12, 12, 12)))

    r = constraint.retract(x)
    assert np.allclose(np.asarray(jnp.sum(r * r, axis=0)), 1.0, atol=1e-5)
    assert np.allclose(np.asarray(constraint.retract(r)), np.asarray(r),
                       atol=1e-6)

    t = constraint.project_tangent(r, g)
    assert np.abs(np.asarray(jnp.sum(r * t, axis=0))).max() < 1e-5
    t2 = constraint.project_tangent(r, t)
    assert np.allclose(np.asarray(t2), np.asarray(t), atol=1e-5)


# --- Faddeev-Bogomolny bound -----------------------------------------------

@pytest.mark.parametrize("B", [1, 2])
def test_seed_respects_bogomolny_bound(B):
    """Any configuration of baryon number B obeys E >= 12 pi^2 sqrt(c2 c4) |B|;
    the seeds (above the minimiser) must respect it a fortiori."""
    phi = skyrmion_hedgehog(GRID) if B == 1 else skyrmion_rational_map(GRID, B=B)
    E = float(MODEL.energy(phi, GRID))
    assert E >= skyrme_bound(B, 1.0, 1.0), f"B={B}: E={E} below bound"

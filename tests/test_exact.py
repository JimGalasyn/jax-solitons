"""Tier-1 validation: exact mathematical properties.

These tests assert identities that hold EXACTLY (up to float rounding) by
construction of the discretization — quantization of the area form,
lattice symmetries, constraint-manifold algebra. A violation is always a
bug, never a tolerance question. See VALIDATION.md for the full map of
claims -> tests -> references.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from jax_solitons import BoxGrid
from jax_solitons.models import faddeev_model
from jax_solitons.models.faddeev import CP1Constraint, S2Constraint
from jax_solitons.seeds import rational_map_hopfion
from jax_solitons.topology import area_form_plaquette, hopf_charge

# float64: these are EXACT identities, only meaningful at full precision (see
# conftest's x64 enable). In float32 a generic O(3) rotation rounds to ~4e-4 and
# trips the tight tolerances below; in float64 the deviation is ~1e-12.
GRID = BoxGrid(N=24, L=8.0, dtype=jnp.float64)


def _random_smooth_n(grid: BoxGrid, seed: int, corr=1.0):
    """Band-limited random unit n-field (an arbitrary smooth map T^3->S^2,
    nothing like our seeds -- adversarial input for exactness tests)."""
    rng = np.random.default_rng(seed)
    _, _, _, K2 = grid.k_vectors()
    filt = np.exp(-np.asarray(K2) * corr**2)
    comps = []
    for _ in range(3):
        w = rng.standard_normal((grid.N,) * 3)
        comps.append(np.real(np.fft.ifftn(np.fft.fftn(w) * filt)))
    n = np.stack(comps) + np.array([0.1, 0.0, 0.3])[:, None, None, None]
    n = n / np.sqrt((n**2).sum(axis=0, keepdims=True))
    return jnp.asarray(n, dtype=grid.dtype)


# --- area-form flux quantization ------------------------------------------

@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_area_form_flux_quantized_on_random_maps(seed):
    """The Berg-Luscher plaquette is the EXACT solid angle of the spherical
    quadrilateral, so its sum over any closed 2-torus slice is 4*pi times
    the integer degree of that slice map -- for ARBITRARY smooth fields,
    not just solitons. The naive same-index discretization has no such
    quantization at any resolution (the library's reason to exist)."""
    n = _random_smooth_n(GRID, seed)
    for (i, j) in ((0, 1), (1, 2), (0, 2)):
        Om = np.asarray(area_form_plaquette(n, i, j))   # solid angle / plaq
        k_axis = 3 - i - j
        flux = Om.sum(axis=(i, j)) / (4.0 * np.pi)      # (N,) degrees
        frac = np.abs(flux - np.round(flux))
        assert frac.max() < 1e-4, \
            f"plane ({i},{j}): flux not integer, max frac {frac.max():.2e}" \
            f" (axis {k_axis})"


@pytest.mark.parametrize("nm,q", [((1, 1), 1), ((2, 1), 2), ((1, 2), 2)])
def test_hopf_charge_integer_on_seeds(nm, q):
    """Q_H(rational-map seed (n,m)) = n*m on a smoke grid."""
    n_w, m_w = nm
    nf = rational_map_hopfion(GRID, R=2.0, n=n_w, m=m_w)
    assert abs(float(hopf_charge(nf, GRID)) - q) < 0.06


# --- lattice symmetries -----------------------------------------------------

def test_energy_translation_invariant():
    """Periodic lattice translations leave E and Q_H exactly invariant
    (same summands; tolerance is summation-order rounding only)."""
    model = faddeev_model(c4=4.0)
    n = _random_smooth_n(GRID, 7)
    E0 = float(model.energy(n, GRID))
    Q0 = float(hopf_charge(n, GRID))
    for shift, axis in ((5, 1), (11, 2), (1, 3)):
        nt = jnp.roll(n, shift, axis=axis)
        assert np.isclose(float(model.energy(nt, GRID)), E0, rtol=1e-5)
        assert np.isclose(float(hopf_charge(nt, GRID)), Q0, atol=1e-4)


def test_energy_global_o3_invariant():
    """Global rotations of the TARGET S^2 leave E and Q_H invariant.
    The 90-degree rotation is exact in floats; the generic Rodrigues
    rotation checks the same identity off the lattice-friendly axes."""
    model = faddeev_model(c4=4.0)
    n = _random_smooth_n(GRID, 8)
    E0 = float(model.energy(n, GRID))
    Q0 = float(hopf_charge(n, GRID))

    n90 = jnp.stack([-n[1], n[0], n[2]])          # exact 90deg about n3
    assert np.isclose(float(model.energy(n90, GRID)), E0, rtol=1e-6)
    assert np.isclose(float(hopf_charge(n90, GRID)), Q0, atol=1e-4)

    th, ax = 0.7, np.array([1.0, 2.0, 3.0])
    ax = ax / np.linalg.norm(ax)
    K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]],
                  [-ax[1], ax[0], 0]])
    R = np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)
    nR = jnp.einsum("ab,bxyz->axyz", jnp.asarray(R, dtype=n.dtype), n)
    # A target rotation maps S^2 to itself, so |nR| = 1; renormalize to undo the
    # ~5e-4 norm drift that GPU float32 matmul (reduced/TF32 precision) injects via
    # the einsum -- the energy genuinely depends on |n|, and float64 gives an exact
    # match, so this removes a numerical artifact, not a real asymmetry.
    nR = nR / jnp.linalg.norm(nR, axis=0, keepdims=True)
    assert np.isclose(float(model.energy(nR, GRID)), E0, rtol=1e-4)
    assert np.isclose(float(hopf_charge(nR, GRID)), Q0, atol=1e-3)


# --- constraint-manifold algebra -------------------------------------------

@pytest.mark.parametrize("constraint,ncomp", [(S2Constraint(), 3),
                                              (CP1Constraint(), 4)])
def test_constraint_algebra(constraint, ncomp):
    """retract is idempotent and lands on the manifold; project_tangent is
    pointwise orthogonal to the state and idempotent."""
    rng = np.random.default_rng(11)
    x = jnp.asarray(rng.standard_normal((ncomp, 12, 12, 12)))
    g = jnp.asarray(rng.standard_normal((ncomp, 12, 12, 12)))

    r = constraint.retract(x)
    nrm = np.asarray(jnp.sum(r * r, axis=0))
    assert np.allclose(nrm, 1.0, atol=1e-6)
    assert np.allclose(np.asarray(constraint.retract(r)), np.asarray(r),
                       atol=1e-6)

    t = constraint.project_tangent(r, g)
    ortho = np.asarray(jnp.sum(r * t, axis=0))
    assert np.abs(ortho).max() < 1e-5
    t2 = constraint.project_tangent(r, t)
    assert np.allclose(np.asarray(t2), np.asarray(t), atol=1e-5)

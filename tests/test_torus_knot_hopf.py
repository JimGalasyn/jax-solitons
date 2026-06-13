"""The Q_H = p*m topological lock for T(p, q) torus-knot hopfions (Paper 16
sec.L_3, Whitehead/Rybakov-2015 reduction).

This gates the L_3 sector of the NWT Lagrangian: a finite-size knotted soliton
whose preimage core is a (p, q) torus knot with phase winding m carries Hopf
charge Q_H = p*m. The seed (seeds.torus_knot_spinor) builds the genuine
torus-knot tube -- the seed geometry the L_2+L_3 coupled-model program needs --
and this test asserts the topological charge lands on p*m, measured by the
engine's own area-form hopf_charge (no external oracle).

The charge SIGN is the Hopf theta = 0/pi (CPT) convention of Paper 16, so the
lock is on |Q_H|. A small finite-box deficit is expected (the paper reports
0.93 vs 1 on 96^3 for the basic hopfion); we gate on rounding to the integer.
"""
from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import pytest

import numpy as np

from jax_solitons.grid import BoxGrid
from jax_solitons.seeds import (
    _closed_rmf,
    torus_knot_hopfion,
    torus_knot_hopfion_cp1,
    torus_knot_spinor,
)
from jax_solitons.topology import hopf_charge

# Paper 16 anchors + a higher-m check.  (p, q, m, expected |Q_H| = p*m)
LOCK_CASES = [
    (2, 1, 3, 6),   # "electron" winding T(2,1), m=3
    (1, 4, 2, 2),   # "proton" winding T(1,4), m=2
    (2, 3, 1, 2),   # trefoil T(2,3), m=1
    (2, 3, 2, 4),   # trefoil T(2,3), m=2
    (3, 2, 1, 3),   # T(3,2), m=1
]

N, L = 80, 20.0


@pytest.fixture(scope="module")
def grid():
    return BoxGrid(N=N, L=L, dtype=jnp.float64)


@pytest.mark.parametrize("p,q,m,expected", LOCK_CASES)
def test_torus_knot_hopf_charge_locks_to_pm(grid, p, q, m, expected):
    """|Q_H| rounds to p*m, and the deficit from the integer is small."""
    n = torus_knot_hopfion(grid, p, q, m)
    Q = float(hopf_charge(n, grid))
    assert round(abs(Q)) == expected, f"T({p},{q}) m={m}: Q_H={Q:.3f}, want |{expected}|"
    assert abs(abs(Q) - expected) < 0.15, f"finite-box deficit too large: {Q:.3f}"


def test_torus_knot_seed_is_unit_field(grid):
    """The seed lies on S^2 (CP^1 spinor is unit-norm by construction)."""
    n = torus_knot_hopfion(grid, 2, 3, 1)
    norm = jnp.sqrt(jnp.sum(n ** 2, axis=0))
    assert float(jnp.max(jnp.abs(norm - 1.0))) < 1e-6


def test_torus_knot_requires_coprime(grid):
    """T(p, q) is a knot only for coprime (p, q)."""
    with pytest.raises(ValueError, match="gcd"):
        torus_knot_spinor(grid, 2, 4, 1)


def test_torus_knot_seed_reaches_vacuum(grid):
    """Far from the tube the field is the north-pole vacuum n = +z."""
    n = torus_knot_hopfion(grid, 2, 3, 1)
    # the box corner is the farthest point from the centered knot
    assert float(n[2, 0, 0, 0]) > 0.99


def test_torus_knot_cp1_state_shape(grid):
    """The CP^1 spinor state is (4, N, N, N) = (Re Z1, Im Z1, Re Z2, Im Z2)."""
    z = torus_knot_hopfion_cp1(grid, 2, 3, 1)
    assert z.shape == (4, grid.N, grid.N, grid.N)


def test_torus_knot_rejects_bad_geometry(grid):
    """Tube geometry must satisfy w < b < R (else the tube self-intersects)."""
    with pytest.raises(ValueError, match="w < b < R"):
        torus_knot_spinor(grid, 2, 3, 1, R=5.0, b=1.0, w=2.0)   # w > b


def test_torus_knot_rejects_oversized_tube(grid):
    """The whole tube (R + b + w) must fit inside the periodic box (< L/2)."""
    with pytest.raises(ValueError, match="does not fit"):
        torus_knot_spinor(grid, 2, 3, 1, R=8.0, b=3.0, w=2.0)   # 13 > L/2 = 10


def test_closed_rmf_z_aligned_seed():
    """A y-z-plane circle has t[0] = +z, exercising the z-aligned seed fallback
    (the default [0,0,1] seed is parallel to t[0], so it must swap to [1,0,0]).
    The returned frame must still be orthonormal and normal to the tangent."""
    S = 200
    th = np.linspace(0.0, 2.0 * np.pi, S, endpoint=False)
    g = np.stack([np.zeros(S), np.cos(th), np.sin(th)], axis=1)
    t = np.stack([np.zeros(S), -np.sin(th), np.cos(th)], axis=1)
    r, u = _closed_rmf(g, t)
    assert np.allclose(np.linalg.norm(r, axis=1), 1.0, atol=1e-6)
    assert np.max(np.abs(np.einsum("ij,ij->i", r, t))) < 1e-6   # r perp t
    assert np.max(np.abs(np.einsum("ij,ij->i", r, u))) < 1e-6   # r perp u


def test_closed_rmf_degenerate_segment_guard():
    """When the reflected tangent already equals the next tangent (v2 ~ 0), the
    frame falls back to the single-reflection result instead of dividing by ~0.
    Crafted so the tangent is perpendicular to every chord: the first Householder
    reflection is the identity, so tL == t[i+1] exactly and c2 = 0."""
    g = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 2.0, 0.0]])  # chord +y
    t = np.array([[1.0, 0.0, 0.0]] * 3)                                 # tangent +x
    r, u = _closed_rmf(g, t)
    assert np.all(np.isfinite(r)) and np.all(np.isfinite(u))            # no NaN
    assert np.allclose(np.linalg.norm(r, axis=1), 1.0, atol=1e-6)
    assert np.max(np.abs(np.einsum("ij,ij->i", r, t))) < 1e-6

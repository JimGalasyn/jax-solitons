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

from jax_solitons.grid import BoxGrid
from jax_solitons.seeds import torus_knot_hopfion, torus_knot_spinor
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

"""A seed curve through a lattice site is a degenerate IC, and must not run.

`prof(d) = tanh(d/core)` is exactly 0 where a seed sample lands on a grid site, so
|phi| = 0 there: a vortex core pinned to a lattice point. For phi2 that leaves
`a = arg phi2` -- the field `axion_grad` differentiates -- undefined exactly where
the L3 coupling reads it, and the run diverges.

Whether it happens is decided by whether R/dx is an integer in BINARY floating
point, so it is not reproducible from the nominal numbers: L=38.4 and
L=38.400000000000006 are the same box to 15 digits and differ here. Measured
2026-08-07 across six configurations, phi2 coincidence predicted divergence with
no exceptions (N=48/96/320 diverged, N=64/128/192 did not).
"""
import numpy as np
import pytest

import jax
jax.config.update("jax_enable_x64", True)

from jax_solitons.ehn.knot_batch import (                     # noqa: E402
    LatticeCoincidence, LatticeCoincidenceWarning, _assert_off_lattice, build_ic,
)

OK = np.array([[1.0, 2.0], [3.0, 4.0]])                       # no zeros anywhere


def test_clean_geometry_passes():
    _assert_off_lattice(OK, OK, N=48, L=38.4, R=9.6)          # must not raise


def test_phi2_coincidence_is_refused():
    bad = np.array([[1.0, 0.0], [3.0, 4.0]])
    with pytest.raises(LatticeCoincidence, match="phi2"):
        _assert_off_lattice(OK, bad, N=48, L=38.4, R=9.6)


def test_the_refusal_names_the_cause_and_the_remedy():
    """A campaign hits this hours in, on a rented box. The message has to carry
    what happened and what to change, not just that something is wrong."""
    bad = np.array([[0.0, 1.0], [3.0, 4.0]])
    with pytest.raises(LatticeCoincidence) as e:
        _assert_off_lattice(OK, bad, N=320, L=256.0, R=64.0)
    m = str(e.value)
    assert "R=64.0" in m and "dx=0.8" in m                    # the actual geometry
    assert "(0, 0)" in m                                      # which site
    assert "R += dx/2" in m                                   # what to do
    assert "floating point" in m                              # why it is not obvious


def test_phi1_coincidence_only_warns():
    """phi1's phase is never differentiated by axion_grad, and N=64/N=128 carry a
    phi1 coincidence and run clean -- so refusing it would reject working
    configurations. It is still degenerate, so it is not silent."""
    bad = np.array([[1.0, 0.0], [3.0, 4.0]])
    with pytest.warns(LatticeCoincidenceWarning, match="phi1"):
        _assert_off_lattice(bad, OK, N=64, L=51.2, R=12.8)


def test_phi2_wins_when_both_coincide():
    bad = np.array([[0.0, 1.0], [3.0, 4.0]])
    with pytest.raises(LatticeCoincidence, match="phi2"):
        _assert_off_lattice(bad, bad, N=48, L=38.4, R=9.6)


@pytest.mark.parametrize("L,R,refused", [
    (38.4, 9.6, False),                    # exact: min dist 1.8e-15, runs clean
    (38.400000000000006, 9.600000000000001, True),   # 1 ULP up: lands on a site
])
def test_build_ic_end_to_end(L, R, refused):
    """The 1-ULP pair that started this: same box to 15 digits, opposite fates."""
    if refused:
        with pytest.raises(LatticeCoincidence):
            build_ic(48, L, 3, R, 2.0, n=400)
    else:
        p1, p2 = build_ic(48, L, 3, R, 2.0, n=400)
        assert np.abs(np.asarray(p2)).min() > 0                # no pinned core


def test_solid_angle_degenerate_branch_is_deterministic():
    """arctan2(0, ±0.0) is 0 or pi by the sign of a zero. The curve's solid angle
    is genuinely undefined ON the curve, so the value is a convention -- it just
    has to be the SAME convention every run."""
    from jax_solitons.ehn.knot_batch import _solid_angle_phase
    # a degenerate curve: both vertices at the evaluation point
    X = np.zeros((2, 2, 2)); Y = np.zeros_like(X); Z = np.zeros_like(X)
    curve = np.zeros((2, 3))
    a = _solid_angle_phase([curve], np.array([0.0, 0.0, 1.0]), X, Y, Z)
    b = _solid_angle_phase([curve], np.array([0.0, 0.0, 1.0]), X, Y, Z)
    assert np.array_equal(a, b) and np.all(a == 0.0)

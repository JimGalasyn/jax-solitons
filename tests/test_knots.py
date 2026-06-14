"""Core-curve knot ID: the determinant on parametric torus knots, and the
implicit-curve tracer on an analytic field whose {g1=0,g2=0} is a known circle."""

import numpy as np
import pytest

from jax_solitons.knots import (
    core_curves_from_psi,
    identify_knot,
    torus_knot,
    trace_implicit_curve,
)


def test_identify_torus_knot_determinants():
    pytest.importorskip("pyknotid")
    for p, q, expect in [(1, 1, 1), (2, 3, 3), (2, 5, 5), (2, 7, 7)]:
        info = identify_knot(torus_knot(p, q))
        assert info["determinant"] == expect, (p, q, info)
    # the label maps onto the carrier ladder
    assert "cinquefoil" in identify_knot(torus_knot(2, 5))["carrier"]


def test_trace_implicit_curve_recovers_a_circle():
    """{g1 = z, g2 = sqrt(x^2+y^2) - R} has zero set = a circle of radius R in the
    z=0 plane — an unknot loop the tracer should close. (No pyknotid needed.)"""
    N, L, R = 48, 8.0, 2.0
    ax = np.linspace(-L / 2, L / 2, N, endpoint=False)
    X, Y, Z = np.meshgrid(ax, ax, ax, indexing="ij")
    loops = trace_implicit_curve(Z, np.sqrt(X**2 + Y**2) - R, (ax, ax, ax),
                                 seed_tol=0.30)
    assert loops, "tracer found no closed loop"
    c = max(loops, key=len)
    assert abs(np.linalg.norm(c[:, :2], axis=1).mean() - R) < 0.3   # radius ~R
    assert np.abs(c[:, 2]).max() < 0.5                              # in z~0 plane


def test_trace_implicit_circle_is_unknot():
    pytest.importorskip("pyknotid")
    N, L, R = 48, 8.0, 2.0
    ax = np.linspace(-L / 2, L / 2, N, endpoint=False)
    X, Y, Z = np.meshgrid(ax, ax, ax, indexing="ij")
    loops = trace_implicit_curve(Z, np.sqrt(X**2 + Y**2) - R, (ax, ax, ax))
    assert identify_knot(max(loops, key=len))["determinant"] == 1   # unknot


def test_core_curves_from_psi_is_a_front_end():
    """psi front end traces {Re psi=0, Im psi=0}; a straight phase-winding line
    psi = (x + i y) has its vortex on the z-axis (an open line, no closed loop in
    a box) — so this just checks the front end runs and returns a list."""
    N, L = 24, 6.0
    ax = np.linspace(-L / 2, L / 2, N, endpoint=False)
    X, Y, _ = np.meshgrid(ax, ax, ax, indexing="ij")
    psi = X + 1j * Y
    assert isinstance(core_curves_from_psi(psi, (ax, ax, ax)), list)

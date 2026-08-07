"""The Hopf-clasped trefoil pair: the geometry, and the charge-sector arithmetic.

Ported 2026-08-07 from a retired private program. Pure curve maths -- no field,
no grid -- so it needs only numpy and the invariants package.

The physics claim underneath, inherited and NOT re-tested here: continuous
descent cannot cross linking classes, so a linked pair has to be born linked.
These tests pin the geometry and the bookkeeping, not that claim.
"""
import numpy as np
import pytest

from jax_solitons.invariants.curves import (
    hopf_clasped_trefoils, torus_knot_curve,
)
from jax_solitons.invariants.linking_invariants import (
    gauss_linking_number, linking_matrix, writhe,
)


# -- the geometry -------------------------------------------------------------
def test_the_clasp_is_a_single_hopf_link():
    """lk = -1, not -4. The naive deep torus-interlock gives 4 because each
    (2,3) trefoil winds its longitude twice; only the shallow 90-degree clasp
    gives one."""
    a, b = hopf_clasped_trefoils()
    assert gauss_linking_number(a, b) == pytest.approx(-1.0, abs=2e-3)


def test_the_clasp_survives_being_rescaled():
    """R and r set the shape; the placement constants are in units of R, so the
    clasp has to hold when the pair is resized -- otherwise every experiment is
    pinned to one box."""
    for R, r in [(1.5, 0.55), (2.2, 0.8), (5.0, 1.5)]:
        a, b = hopf_clasped_trefoils(R=R, r=r)
        assert gauss_linking_number(a, b) == pytest.approx(-1.0, abs=2e-3), (R, r)


@pytest.mark.parametrize("sep_scale,linked", [
    (0.85, True), (1.00, True), (1.20, True),      # the measured window
    (1.30, False), (1.70, False), (2.30, False),   # the source claimed 2.3 held
])
def test_the_breathing_window_is_narrow_and_the_inherited_range_was_wrong(
        sep_scale, linked):
    """The port's source recorded "lk stays -1 over s ~ 0.85-2.3+". Swept and
    measured, it holds to s = 1.20 and is gone by 1.30.

    This is the test that matters most in the file. A separation scan run on the
    inherited range would spend most of its points on an UNLINKED pair while
    labelling them linked -- and the resulting E(d) curve would look like a
    binding measurement rather than the topology change it actually is.
    """
    a, b = hopf_clasped_trefoils(sep_scale=sep_scale)
    lk = gauss_linking_number(a, b)
    if linked:
        assert lk == pytest.approx(-1.0, abs=2e-3)
    else:
        assert abs(lk + 1.0) > 0.5


def test_the_clasp_opens_through_a_non_integer_reading():
    """At s = 1.25 the Gauss integral returns ~ +3.4, which is not a linking
    number of anything -- the curves are passing through each other there.
    Pinned so that a scan crossing this point is recognised as passing through
    a curve intersection rather than measuring a link."""
    a, b = hopf_clasped_trefoils(sep_scale=1.25)
    lk = gauss_linking_number(a, b)
    assert abs(lk - round(lk)) > 0.1        # not near ANY integer


def test_clasped_and_separated_pairs_sit_in_different_charge_sectors():
    """THE result the collision question turns on, and it is arithmetic.

    Helicity of a set of unit-circulation tubes is
    H = sum_i Sl_i + 2 sum_{i<j} Lk_ij, and for a Faddeev-Skyrme field that H is
    the Hopf charge. The self-terms are framing-dependent, so absolute H is --
    but the two configurations here are the SAME two curves rigidly moved, so
    every Sl_i cancels in the difference and what is left is exact:

        dH = 2 * (lk_clasped - lk_separated) = 2 * (-1 - 0) = -2

    Measured: writhe is -3.2778 for all four curves (separated A, separated B,
    clasped A, clasped B), so sum(Sl) = -6.5555 in both, and H goes -6.5555 ->
    -8.5557.

    Consequence, where Q_H is a genuine homotopy invariant: the two
    configurations are two units apart and NO continuous evolution connects
    them. A collision of two free trefoils cannot produce the clasped pair at
    fixed charge, whatever the boost or impact parameter -- they would have to
    be born linked. In GPE, by contrast, helicity is not conserved at all
    (reconnections change it -- see the relaxation test in
    test_linked_vortex_pair.py, which measures the link being shed), so GPE
    cannot be used to test the claim either way.
    """
    a = torus_knot_curve(2, 3, R=2.2, r=0.8, n_points=480)
    sep = [a - np.array([6.0, 0, 0]), a + np.array([6.0, 0, 0])]
    cl = list(hopf_clasped_trefoils(R=2.2, r=0.8, n_points=480))

    lone = writhe(a)
    for c in sep + cl:                       # rigid motions only: Sl must cancel
        assert writhe(c) == pytest.approx(lone, rel=1e-9)

    lk_sep = linking_matrix(sep)[0, 1]
    lk_cl = linking_matrix(cl)[0, 1]
    assert lk_sep == pytest.approx(0.0, abs=2e-3)
    assert lk_cl == pytest.approx(-1.0, abs=2e-3)

    h_sep = 2 * lone + 2 * lk_sep
    h_cl = 2 * lone + 2 * lk_cl
    assert h_cl - h_sep == pytest.approx(-2.0, abs=1e-2)


"""Writhe: the geometric half of self-linking, and the half a curve can supply.

Needed for the helicity ledger H = sum(Sl_i) + 2 sum(Lk_ij) that decides whether
two configurations sit in the same charge sector. Lk was already here; Wr was
not, and it was being written ad hoc in analysis scripts.
"""
import numpy as np
import pytest

from jax_solitons.invariants.curves import (
    hopf_clasped_trefoils, torus_knot_curve,
)
from jax_solitons.invariants.linking_invariants import writhe


def _circle(n=400, R=1.0):
    t = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    return np.stack([R * np.cos(t), R * np.sin(t), np.zeros_like(t)], axis=1)


def test_a_planar_curve_has_zero_writhe():
    """The one case with an exact answer: a plane curve's writhe vanishes
    identically, because dr1 x dr2 is normal to the plane and r1 - r2 lies in
    it. Not approximately -- the integrand is zero pointwise."""
    assert writhe(_circle()) == pytest.approx(0.0, abs=1e-12)


def test_writhe_is_odd_under_reflection():
    """Wr measures handedness, so a mirror image must return exactly minus it.
    Catches a transposed cross product, which a magnitude-only check would not."""
    c = torus_knot_curve(2, 3, R=2.2, r=0.8, n_points=480)
    mirrored = c.copy()
    mirrored[:, 2] *= -1
    assert writhe(mirrored) == pytest.approx(-writhe(c), rel=1e-9)


def test_writhe_is_invariant_under_rigid_motion():
    """Rotation and translation must not move it -- this is what lets a helicity
    ledger cancel the self-terms between two configurations of the same curves,
    which is the whole reason the ledger is computable."""
    c = torus_knot_curve(2, 3, R=2.2, r=0.8, n_points=480)
    w0 = writhe(c)
    ang = 0.7
    rot = np.array([[np.cos(ang), -np.sin(ang), 0.0],
                    [np.sin(ang), np.cos(ang), 0.0],
                    [0.0, 0.0, 1.0]])
    moved = (rot @ c.T).T + np.array([3.0, -1.5, 0.9])
    assert writhe(moved) == pytest.approx(w0, rel=1e-9)


def test_writhe_is_invariant_under_reparametrisation():
    """Rolling the sample index is the same curve. Guards the periodic
    along-curve distance used by `skip`: computing it as |i-j| instead of the
    wrapped minimum would break exactly here, at the seam."""
    c = torus_knot_curve(2, 3, R=2.2, r=0.8, n_points=480)
    assert writhe(np.roll(c, 137, axis=0)) == pytest.approx(writhe(c), rel=1e-9)


def test_writhe_is_scale_invariant():
    """Wr is dimensionless: the 1/|r|^3 kernel against two dr's and a (r1-r2)."""
    c = torus_knot_curve(2, 3, R=2.2, r=0.8, n_points=480)
    assert writhe(10.0 * c) == pytest.approx(writhe(c), rel=1e-9)


@pytest.mark.parametrize("n", [240, 480, 960])
@pytest.mark.parametrize("skip", [1, 2, 4])
def test_writhe_is_converged_in_sampling_and_in_skip(n, skip):
    """A number that moved with the discretisation would make any ledger built
    on it meaningless, so the stability is pinned rather than assumed. Measured
    spread across this grid is in the 4th decimal."""
    c = torus_knot_curve(2, 3, R=2.2, r=0.8, n_points=n)
    assert writhe(c, skip=skip) == pytest.approx(-3.278, abs=5e-3)


def test_the_clasp_does_not_change_either_curve_s_writhe():
    """`hopf_clasped_trefoils` builds its second component by rotating and
    translating the first, so both components must writhe identically to a
    lone trefoil. This is the assumption that lets the sector comparison reduce
    to the mutual linking: if the clasp changed the self-terms, the difference
    between the configurations would not be 2*dLk.
    """
    lone = writhe(torus_knot_curve(2, 3, R=2.2, r=0.8, n_points=480))
    a, b = hopf_clasped_trefoils(R=2.2, r=0.8, n_points=480)
    assert writhe(a) == pytest.approx(lone, rel=1e-9)
    assert writhe(b) == pytest.approx(lone, rel=1e-9)


def test_agrees_with_pyknotid_s_writhing_integral():
    """An independent implementation, gated on the optional extra.

    Why ours is the one that ships rather than a delegation to pyknotid, which
    does compute this:

      - pyknotid is the optional `[knots]` extra, and this module is documented
        numpy-only. `gauss_linking_number` is likewise implemented here rather
        than delegated; a core invariant should not disappear in a base install.
      - measured at n=480, ours takes 0.03 s against pyknotid's 15.3 s for the
        integral method -- 500x, and a helicity ledger calls this once per curve
        per configuration.
      - pyknotid's default `projections` method is an approximation (averaged
        planar writhe, ~1.4% off here); only its `integral` method is comparable.
      - ours converges faster in sample count: stable in the 4th decimal from
        n=240 to n=960, where pyknotid's integral still moves in the 2nd.

    What it IS good for is exactly this: two implementations agreeing is the
    check that caught a factor-2 error in `vortex_topology.total_helicity`.
    """
    pytest.importorskip("pyknotid")
    from jax_solitons.knots import _knot_class

    c = torus_knot_curve(2, 3, R=2.2, r=0.8, n_points=480)
    theirs = float(_knot_class()(c.copy()).writhe(method="integral"))
    assert writhe(c) == pytest.approx(theirs, rel=5e-3)


def test_too_few_samples_raises_rather_than_returning_a_number():
    """With skip=2 a 5-point curve has every pair excluded, and the honest sum
    of nothing is not 0.0 -- it is undefined. Returning 0.0 would read as a
    planar curve."""
    with pytest.raises(ValueError, match="too few"):
        writhe(_circle(n=5), skip=2)

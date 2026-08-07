"""total_helicity must equal the core curve's writhe. It used to return half.

The single-trefoil case is the one with no place to hide: helicity of one closed
tube is its self-linking, there is no cross term, and `linking_invariants.writhe`
computes the same number independently from the curve. Two implementations of one
quantity disagreeing by a factor of two is how the bug surfaced -- reading the
code did not, because the offending `0.5` carried a comment explaining itself.
"""
import numpy as np
import pytest

import jax
jax.config.update("jax_enable_x64", True)

from jax_solitons.grid import BoxGrid                              # noqa: E402
from jax_solitons.invariants.curves import (                       # noqa: E402
    hopf_clasped_trefoils, torus_knot_curve,
)
from jax_solitons.invariants.linking_invariants import (           # noqa: E402
    linking_matrix, writhe,
)
from jax_solitons.seeds import superflow_seed                      # noqa: E402
from jax_solitons.vortex_topology import total_helicity            # noqa: E402

NUDGE = np.array([0.0413, 0.0237, 0.0119])       # general position, see seeds.py
NPTS = 3000


@pytest.mark.parametrize("N,ratio", [(72, 0.876), (96, 1.047)])
def test_helicity_of_one_tube_is_its_writhe(N, ratio):
    """The factor-2 regression test, and the measurement that found it.

    H = sum_i Wr_i + 2 sum_{i<j} Lk_ij. For ONE tube that is just Wr, so the
    field measurement and the curve measurement must agree. Before the fix this
    read -1.72 against the curve's -3.28 -- a ratio of 0.52, not a discretisation
    error -- because the ORDERED double sum was being halved. The (i,j) and (j,i)
    halves are exactly what supply the factor 2 on the cross terms.

    Both resolutions are pinned because the regularisation error is not monotone
    in N (0.88 at 72, 1.05 at 96), and a single-resolution test would read as a
    tighter guarantee than this meter can give. The band is wide enough to hold
    both and nowhere near wide enough to admit 0.5.
    """
    grid = BoxGrid(N=N, L=12.0, dtype=np.float64)
    curve = torus_knot_curve(2, 3, R=2.2, r=0.8, n_points=NPTS) + NUDGE
    psi = np.asarray(superflow_seed(grid, [curve], core=0.7))
    measured = total_helicity(psi, grid.dx, grid.L) / writhe(curve)
    assert measured == pytest.approx(ratio, abs=0.05)
    assert measured > 0.7, "a halved convention would land near 0.5"


def test_the_halved_convention_would_fail_this():
    """Guards the guard: confirm the band above actually excludes H/2, so the
    test could fail if the 0.5 came back. A regression test that passes under the
    bug it names is decoration."""
    grid = BoxGrid(N=96, L=12.0, dtype=np.float64)
    curve = torus_knot_curve(2, 3, R=2.2, r=0.8, n_points=NPTS) + NUDGE
    psi = np.asarray(superflow_seed(grid, [curve], core=0.7))
    halved = 0.5 * total_helicity(psi, grid.dx, grid.L) / writhe(curve)
    assert not (halved > 0.7), f"the guard is too loose: halved ratio {halved:.3f}"


def test_the_regularisation_does_not_cancel_between_configurations():
    """A correction to this module's own earlier guidance, pinned so it stays
    corrected.

    `total_helicity`'s docstring used to say to prefer DIFFERENCES between
    configurations over absolute numbers, on the assumption that the
    regularisation error cancels. Measured, it does not. The clasped pair reads
    within 0.2% of its predicted -8.555, but the separated pair under-reads its
    -6.555 by ~12%, so the difference comes out near -2.8 where the curves say
    exactly -2.0.

    The reason is mechanical: the near-pair exclusion depends on how close the
    tubes are to each other, and clasped-versus-separated differ in precisely
    that. So the curves stay the authority for a NUMBER (writhe and
    gauss_linking_number are exact); the field meter is for sign and shape.
    """
    a = torus_knot_curve(2, 3, R=2.2, r=0.8, n_points=NPTS)
    sep = [a - np.array([4.0, 0, 0]) + NUDGE, a + np.array([4.0, 0, 0]) + NUDGE]
    cl = [c + NUDGE for c in hopf_clasped_trefoils(R=2.2, r=0.8, n_points=NPTS)]
    grid = BoxGrid(N=108, L=18.0, dtype=np.float64)

    assert linking_matrix(sep)[0, 1] == pytest.approx(0.0, abs=2e-3)
    assert linking_matrix(cl)[0, 1] == pytest.approx(-1.0, abs=2e-3)

    h_sep = total_helicity(np.asarray(superflow_seed(grid, sep, core=0.7)),
                           grid.dx, grid.L)
    h_cl = total_helicity(np.asarray(superflow_seed(grid, cl, core=0.7)),
                          grid.dx, grid.L)
    exact_sep = 2 * writhe(a)
    exact_cl = 2 * writhe(a) + 2 * linking_matrix(cl)[0, 1]

    # the clasped configuration is read well ...
    assert h_cl == pytest.approx(exact_cl, rel=0.03)
    # ... the separated one is not, and that is the point
    assert abs(h_sep / exact_sep - 1.0) > 0.05, (
        f"separated now reads {h_sep:.3f} against {exact_sep:.3f} -- if the "
        "regularisation improved, tighten this and revisit the docstring")
    # so the difference is NOT the curves' -2.0
    assert abs((h_cl - h_sep) - (-2.0)) > 0.3

"""A linked pair of vortex knots: seed it, measure it, and know when it opened.

Ported 2026-08-07 from a retired private program's GPE binding/linking scripts.
What came over is the geometry (a single Hopf clasp between two trefoils), the
Nore-Abid-Brachet sound/vortex energy split, and the same-box binding protocol.
What did NOT come over is that program's pseudovorticity linking measure --
`vortex_topology.linking_number` already keys on the phase winding, which is
integer-clean and needs no calibration against a known-lk seed.

The physics claim under all of it, inherited and NOT re-tested here: continuous
descent cannot cross linking classes, so a linked pair has to be born linked.
These tests pin the instrument, not that claim.
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
    gauss_linking_number,
)
from jax_solitons.seeds import superflow_seed                      # noqa: E402
from jax_solitons.vortex_topology import (                         # noqa: E402
    _label_lines, core_separation, kinetic_decomposition,
    link_binding_energy, linking_number, vortex_skeleton,
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


# -- seed and read back --------------------------------------------------------
GRID = BoxGrid(N=72, L=12.0, dtype=np.float64)     # seeding is O(n_points * N^3)


@pytest.fixture(scope="module")
def seeded_pair():
    """A clasped pair on a grid that actually resolves the clasp."""
    a, b = hopf_clasped_trefoils(R=2.2, r=0.8, n_points=240)
    return np.asarray(superflow_seed(GRID, [a, b], core=0.7))


@pytest.fixture(scope="module")
def seeded_single():
    c = torus_knot_curve(2, 3, R=2.2, r=0.8, n_points=240)
    return np.asarray(superflow_seed(GRID, [c], core=0.7))


def test_the_seeded_field_carries_the_curves_link(seeded_pair):
    """The end-to-end contract: curve lk -> seeded field -> measured lk. Two
    separate components, and the linking the geometry specified."""
    n_lines, lk, sizes = linking_number(seeded_pair, GRID.dx, GRID.L)
    assert n_lines == 2, sizes
    assert lk == pytest.approx(-1.0, abs=0.05)


def test_a_single_trefoil_seeds_as_one_dominant_line(seeded_single):
    """The control: no clasp, so no second component carrying real structure.

    Asserted as DOMINANCE rather than `n_lines == 1`, because that stricter form
    is resolution-dependent in a way that would make this a grid-stabilized
    test: the same curve at N=96 throws an extra 6-segment fragment (against 378
    real ones) that clears the default min_seg=6 cut. The fragment is noise, it
    moves with the grid, and a linking measurement that paired the real line
    with it would read ~0 and look like an answer. Dominance holds at both
    resolutions and says the thing that is actually true.
    """
    P, _, C = vortex_skeleton(seeded_single)
    _, big, sizes = _label_lines(C, seeded_single.shape, 6)
    assert len(big) >= 1
    assert sizes[big[0]] > 0.95 * sizes[1:].sum()


def test_core_separation_is_finite_and_smaller_than_the_box(seeded_pair):
    sep = core_separation(seeded_pair, GRID.dx, GRID.L)
    assert np.isfinite(sep)
    assert 0.0 < sep < GRID.L / 2


def test_core_separation_is_nan_when_there_is_only_one_line(seeded_single):
    """Not an error: two lines merging into one is what a reconnection looks
    like frame to frame, and the caller needs to see it as a gap in the series
    rather than a number."""
    assert np.isnan(core_separation(seeded_single, GRID.dx, GRID.L, min_seg=40))


# -- the energy meters ---------------------------------------------------------
def test_a_seeded_vortex_pair_is_mostly_incompressible(seeded_pair):
    """A freshly seeded field is vortex flow with very little sound; that is what
    makes the later rise of the compressible part readable as radiation rather
    than as seeding noise."""
    e_inc, e_comp = kinetic_decomposition(seeded_pair, GRID.dx)
    assert e_inc > 0 and e_comp > 0
    assert e_comp < 0.1 * e_inc


def test_kinetic_decomposition_sends_irrotational_flow_to_the_compressible_bin():
    """A sinusoidal phase, psi = exp(i a sin(kx)), gives u = a k cos(kx) x-hat --
    a pure gradient, so ALL of its kinetic energy must land in the compressible
    bin. The check that fails if the Helmholtz projection is transposed.

    NOT a uniform phase ramp: that puts u entirely at k = 0, where the
    decomposition has no direction to project onto and the convention assigns
    the zero mode to the incompressible part. Uniform flow is the one case this
    meter cannot classify, which is worth knowing before reading a boosted run.
    """
    n, L = 32, 8.0
    dx = L / n
    ax = np.arange(n) * dx
    X = np.meshgrid(ax, ax, ax, indexing="ij")[0]
    psi = np.exp(1j * 0.3 * np.sin(2 * np.pi * X / L)).astype(np.complex128)
    e_inc, e_comp = kinetic_decomposition(psi, dx)
    assert e_comp > 0
    assert abs(e_inc) < 1e-5 * e_comp


def test_binding_energy_sign_convention():
    """Positive = the cluster sits below its separated constituents."""
    assert link_binding_energy(e_cluster=190.0, e_single=100.0,
                               n_constituents=2) == pytest.approx(10.0)
    assert link_binding_energy(e_cluster=210.0, e_single=100.0,
                               n_constituents=2) == pytest.approx(-10.0)


# -- the inherited result, reproduced -----------------------------------------
def test_imaginary_time_relaxation_unlinks_the_pair(seeded_pair):
    """The recorded result this port had to reproduce before the instrument
    could be trusted: GPE gradient flow does not hold the link, it reconnects
    and sheds it. Measured here at N=96: lk -1.00 -> 0.00 within 10 steps, while
    the incompressible (vortex) energy drains into the compressible (sound) bin.

    So the GPE leg is a dead end for a bound linked state, and the value of these
    primitives is as instruments for the models where knots ARE minimizers.
    """
    n, dx = GRID.N, GRID.dx
    k1 = 2.0 * np.pi * np.fft.fftfreq(n, d=dx)
    KX, KY, KZ = np.meshgrid(k1, k1, k1, indexing="ij")
    k2 = KX ** 2 + KY ** 2 + KZ ** 2

    psi = seeded_pair.copy()
    _, lk0, _ = linking_number(psi, dx, GRID.L)
    e_inc0, _ = kinetic_decomposition(psi, dx)
    assert lk0 == pytest.approx(-1.0, abs=0.05)

    for _ in range(10):
        psi = psi * np.exp(-0.05 * (np.abs(psi) ** 2 - 1.0))
        psi = np.fft.ifftn(np.fft.fftn(psi) * np.exp(-0.05 * k2))
        psi = psi * np.exp(-0.05 * (np.abs(psi) ** 2 - 1.0))

    _, lk1, _ = linking_number(psi, dx, GRID.L)
    e_inc1, _ = kinetic_decomposition(psi, dx)
    assert abs(lk1) < 0.1, f"expected the link to be shed, got lk={lk1}"
    assert e_inc1 < 0.5 * e_inc0        # vortex energy went somewhere

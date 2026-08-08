"""Seeding a linked pair into a field, and the meters that read it back.

The curve geometry and the charge-sector arithmetic live in
test_clasped_trefoils.py; this file starts from those curves and is about the
FIELD -- does the seed carry the link, is it periodic, and where does the energy
sit.

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
from jax_solitons.seeds import superflow_seed                      # noqa: E402
from jax_solitons.vortex_topology import (                         # noqa: E402
    _label_lines, core_separation, kinetic_decomposition,
    link_binding_energy, linking_number, vortex_skeleton,
)


# -- seed and read back --------------------------------------------------------
GRID = BoxGrid(N=72, L=12.0, dtype=np.float64)     # seeding is O(n_points * N^3)


# Off-lattice nudge. The plaquette deposition counts face piercings, so a curve
# running along a lattice plane pierces nothing there and its line fails to
# close -- refused, by design. An irrational-looking sub-cell offset puts the
# geometry in general position, and is the same remedy build_ic_torus uses.
NUDGE = np.array([0.0413, 0.0237, 0.0119])
# Dense sampling: the deposition walks segment by segment, so segments must be
# much shorter than dx (= 1/6 here).
NPTS = 3000


@pytest.fixture(scope="module")
def seeded_pair():
    """A clasped pair on a grid that actually resolves the clasp."""
    a, b = hopf_clasped_trefoils(R=2.2, r=0.8, n_points=NPTS)
    return np.asarray(superflow_seed(GRID, [a + NUDGE, b + NUDGE], core=0.7))


@pytest.fixture(scope="module")
def seeded_single():
    c = torus_knot_curve(2, 3, R=2.2, r=0.8, n_points=NPTS) + NUDGE
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


def test_core_separation_is_nan_on_a_field_with_no_vortices_at_all():
    """The empty-skeleton branch, which the single-line test does NOT reach.

    `min_seg` filtering and finding no phase winding anywhere are separate exits
    returning the same value, and only the first was covered. A vortex-free field
    is what a fully decayed run looks like, and it must read as a gap in the
    series like any other -- not 0.0, which is the value for two coincident
    lines and would say the opposite of what happened.

    Deliberately tiny and uniform: no seeding, so this costs nothing.
    """
    psi = np.ones((16, 16, 16), dtype=np.complex128)
    assert np.isnan(core_separation(psi, GRID.dx, GRID.L))


# -- the energy meters ---------------------------------------------------------
def test_the_seed_is_periodic(seeded_pair):
    """The predecessor of this test asserted the OPPOSITE, deliberately: the
    free-space (solid-angle) seed was not periodic, and the note left on it said
    to restore the `e_comp < 0.1 * e_inc` assertion if the seeder was ever fixed.
    It has been, so both are back.

    The seam is now well under one interior step on every axis, where the
    free-space construction was at 143% / 13% / 140%. What remains is the
    discretisation of the tanh profile, not a phase discontinuity.
    """
    psi = np.asarray(seeded_pair)
    for ax in range(3):
        seam = float(np.abs(np.take(psi, 0, axis=ax)
                            - np.take(psi, -1, axis=ax)).max())
        interior = float(np.abs(np.diff(psi, axis=ax)).max())
        assert seam < 0.25 * interior, (
            f"axis {ax}: seam {seam:.3e} vs interior step {interior:.3e}")


def test_a_seeded_vortex_pair_is_mostly_incompressible(seeded_pair):
    """A freshly seeded field is vortex flow with very little sound; that is what
    makes the later rise of the compressible part readable as radiation rather
    than as seeding noise.

    This is the assertion the non-periodic seed could not support. With the
    free-space phase the spectral gradient rang on the seam and gave comp/inc =
    0.61 -- all seam, no sound. The plaquette-basis seed gives 0.018.
    """
    e_inc, e_comp = kinetic_decomposition(seeded_pair, GRID.dx)
    assert e_inc > 0 and e_comp > 0
    assert e_comp < 0.1 * e_inc, f"comp/inc = {e_comp / e_inc:.3f}"


def test_a_degenerate_geometry_is_refused_rather_than_seeded():
    """A ring lying exactly in a lattice plane pierces no face of the family it
    is parallel to, so its deposited vortex line does not close. That is not a
    small error -- the field would carry an OPEN vortex line, which is not a
    configuration at all. Same condition and same remedy as #97's
    LatticeCoincidence.
    """
    from jax_solitons.seeds import DegenerateSeedGeometry
    g = BoxGrid(N=48, L=12.0, dtype=np.float64)
    t = np.linspace(0.0, 2 * np.pi, 2000, endpoint=False)
    flat = np.stack([3.0 * np.cos(t), 3.0 * np.sin(t), np.zeros_like(t)], axis=1)
    with pytest.raises(DegenerateSeedGeometry, match="do not close"):
        superflow_seed(g, [flat], core=0.7)


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


# -- periodicity fixes from Copilot's review on #99 ----------------------------
def test_core_separation_uses_the_periodic_box():
    """It takes L and documents periodic behaviour; it used to ignore L entirely.
    Two lines placed just either side of the boundary are CLOSE, not L apart."""
    import numpy as np
    from jax_solitons.vortex_topology import _periodic_centroid
    L = 10.0
    # points straddling the +-L/2 seam: true centroid is the seam, not the origin
    pts = np.array([[4.9, 0.0, 0.0], [-4.9, 0.0, 0.0]])
    c = _periodic_centroid(pts, L)
    assert abs(abs(c[0]) - 5.0) < 1e-6, f"circular mean landed at {c[0]:.3f}"
    assert abs(np.mean(pts[:, 0])) < 1e-9      # the arithmetic mean says 0.0 -- wrong


def test_kinetic_decomposition_gradient_is_periodic():
    """np.gradient is one-sided at the edges; this function then projects
    spectrally. A field that is smooth ACROSS the seam must not manufacture
    compressible energy there."""
    import numpy as np
    from jax_solitons.vortex_topology import kinetic_decomposition
    n, L = 32, 2 * np.pi
    dx = L / n
    x = np.arange(n) * dx
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    # a pure phase ramp with an integer number of periods: perfectly periodic,
    # uniform |psi|, so the flow is uniform and ENTIRELY incompressible
    psi = np.exp(1j * X).astype(np.complex128)
    inc, comp = kinetic_decomposition(psi, dx)
    assert comp / (inc + comp) < 1e-6, (
        f"periodic field produced {comp/(inc+comp):.2e} compressible fraction")

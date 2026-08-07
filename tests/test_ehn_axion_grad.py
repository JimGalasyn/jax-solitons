"""The three ∂a discretizations, and the branch cut that separates them.

`axion_grad` is the campaign's independent variable (NLINK_LADDER.md): the whole
N_link ladder is one arm per mode with everything else frozen. So these are
contract tests on what each mode IS, not on physics -- an arm that quietly
computed a different mode than its label would invalidate a campaign rather than
fail a test.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from jax_solitons.ehn.energy import (axion_grad, curlA,                 # noqa: E402
                                     E_L3_electric, rho_L3)
from jax_solitons.ehn.knot_batch import build_ic                 # noqa: E402

DX = 0.5


def _pure_phase(a):
    """A unit-modulus field with prescribed phase -- isolates the phase handling
    from any modulus weighting."""
    return jnp.exp(1j * jnp.asarray(a))


def _ramp_x(n=16, slope=0.3):
    """Phase rising linearly along x by `slope` PER SITE. Returns the field and the
    true ∂_x a, which is slope/DX -- sites are DX apart, so phase-per-site is not
    phase-per-unit-length. The branch cut is an artefact of representing this field
    in (-pi, pi], not a feature of it.

    `jnp.roll` also wraps the domain edge, so rows 0 and n-1 carry a periodic seam
    in EVERY mode. That seam is a boundary condition, not a branch cut, and the
    tests below drop it so the two are not confused.
    """
    x = np.arange(n)[:, None, None] * np.ones((1, n, n))
    return _pure_phase(slope * x), slope / DX


def test_naive_matches_wrapped_away_from_the_cut():
    """With no wrap in the stencil the two agree: `wrapped` differs from `naive`
    only in taking the angle of the ratio instead of the difference of angles."""
    p2 = _pure_phase(0.2 * np.arange(8)[:, None, None] * np.ones((1, 8, 8)))
    gw = axion_grad(p2, DX, 0.05, "wrapped")[0]
    gn = axion_grad(p2, DX, 0.05, "naive")[0]
    interior = slice(2, 6)                       # rows whose 3-point stencil is clean
    assert np.allclose(np.asarray(gw)[interior], np.asarray(gn)[interior], atol=1e-9)


def test_naive_blows_up_on_the_branch_cut_and_wrapped_does_not():
    """THE reason this arm exists. arctan2 is principal-valued, so where the phase
    crosses ±pi two neighbours differ by ~2pi and the naive difference reports
    ~2pi/2dx -- a spurious O(1/dx) gradient on a whole sheet. `wrapped` sees the
    same field and returns the true slope."""
    p2, want = _ramp_x(n=16, slope=0.3)          # phase crosses +pi at x = 11
    gw = np.asarray(axion_grad(p2, DX, 0.05, "wrapped")[0])[1:-1]   # drop the seam
    gn = np.asarray(axion_grad(p2, DX, 0.05, "naive")[0])[1:-1]

    assert np.allclose(gw, want, atol=1e-9)      # wrapped: exact, cut and all
    cut = np.abs(gn - want) > 1.0                # naive: wrong on the cut sheet
    assert cut.any(), "no branch cut in this field -- the test proves nothing"
    # the sheet is codimension-1: whole (y, z) planes at the crossing, not specks
    assert {int(i) for i in np.unique(np.nonzero(cut)[0])} == {9, 10}
    # the error is exactly one winding spread over the 2dx stencil: 2pi/(2dx).
    assert np.allclose(np.abs(gn[cut] - want), np.pi / DX, atol=1e-9)
    assert np.allclose(gn[~cut], want, atol=1e-9)    # and right everywhere else


def test_naive_is_modulus_blind_like_wrapped():
    """Both phase-based modes ignore |phi2|; only bilinear is modulus-weighted.
    This is what makes `naive` a test of the BRANCH CUT specifically, rather than
    a second test of modulus suppression."""
    p2, slope = _ramp_x(n=12, slope=0.2)
    damped = p2 * (0.1 + 0.9 * np.random.default_rng(0).random(p2.shape))
    for mode in ("wrapped", "naive"):
        g0 = np.asarray(axion_grad(p2, DX, 0.05, mode)[0])
        g1 = np.asarray(axion_grad(damped, DX, 0.05, mode)[0])
        assert np.allclose(g0, g1, atol=1e-9), f"{mode} moved with the modulus"


def test_bilinear_is_modulus_suppressed():
    """The contrast case, so the above is not vacuous."""
    p2, _ = _ramp_x(n=12, slope=0.2)
    g_unit = np.asarray(axion_grad(p2, DX, 0.05, "bilinear")[0])
    g_small = np.asarray(axion_grad(0.05 * p2, DX, 0.05, "bilinear")[0])
    assert np.abs(g_small).max() < 0.5 * np.abs(g_unit).max()


@pytest.mark.parametrize("mode", ["bilinear", "wrapped", "naive"])
def test_every_mode_returns_three_components(mode):
    p2, _ = _ramp_x(n=8)
    g = axion_grad(p2, DX, 0.05, mode)
    assert len(g) == 3 and all(x.shape == p2.shape for x in g)


def test_unknown_mode_raises_rather_than_defaulting_to_bilinear():
    """`agrad` is recorded in the manifest. A silent fall-through would file a
    bilinear run under a misspelled arm name with nothing in the artifacts to
    catch it -- safe with two modes, a trap with three."""
    p2, _ = _ramp_x(n=8)
    with pytest.raises(ValueError, match="unknown agrad"):
        axion_grad(p2, DX, 0.05, "wrappd")


# -- does a mode carry winding at all? ----------------------------------------
@pytest.fixture(scope="module")
def campaign_ic():
    """The campaign's own IC -- the claim is about what the ladder actually ran on,
    not a synthetic winding. Module-scoped because build_ic is O(n.N^3) in numpy
    and dominates this file's runtime."""
    N, L, R, core = 48, 38.4, 9.6, 2.0
    _, p2 = build_ic(N, L, 3, R, core, n=400)
    return N, L, L / N, jnp.asarray(p2)


def test_circulation_is_the_property_that_separates_the_modes(campaign_ic):
    """The physical content of "a is a compact angle" is that grad(a) has
    circulation 2*pi*(winding). This is the test that says whether an arm can
    carry the L3 charge at all, rather than how well.

    `naive` scores EXACTLY zero, and not by accident: a = arctan2(Im, Re) is
    single-valued, so summing its central difference around a closed periodic loop
    telescopes to 0 identically. The branch-cut sheet is not noise on top of a good
    gradient -- it is precisely the term that cancels the smooth winding.

    The consequence, stated narrowly: net(rho) = 0 IDENTICALLY under naive, for
    uniform B and for any divergence-free B, while |rho| is the LARGEST of the
    three modes. So rho cannot represent a winding-derived charge under naive --
    and EHN's floor is a statement about integral-rho being locked to N_link, so
    the arm cannot exhibit the mechanism the campaign exists to test. That is what
    voids the naive ladder arm.

    What this does NOT establish: that the L3 coupling is inert. The engine forms
    `eelec = 0.5*C*sum(rho*s)` with a spatially varying A0 (relax.py), and
    sum(rho) = 0 does not imply sum(rho*s) = 0 -- the weighting can pick out the
    cancelling spikes instead of cancelling with them. An earlier version of this
    docstring said "the L3 coupling contributes nothing"; that was wrong, and
    `test_vanishing_net_rho_does_not_bound_the_L3_energy` below pins the
    replacement with both A0s defined in-tree rather than quoting numbers from a
    field the reader cannot reconstruct.
    """
    N, L, dx, p2 = campaign_ic

    def max_circ(mode):
        """Closed periodic x-loops at every (y, z): sum(grad_x a)*dx / 2pi."""
        gx = np.asarray(axion_grad(p2, dx, 0.05, mode)[0])
        return np.abs(gx.sum(axis=0) * dx / (2 * np.pi)).max()

    assert max_circ("naive") == pytest.approx(0.0, abs=1e-12)   # no winding, at all
    assert max_circ("wrapped") == pytest.approx(1.0, abs=1e-9)  # exactly one
    assert max_circ("bilinear") > 1e-3        # leaky but nonzero: it does carry some


def test_net_rho_vanishes_under_naive_for_a_divergence_free_B(campaign_ic):
    """The general statement is about DIVERGENCE-FREE B, not uniform B: for
    single-valued `a`, integral(B.grad a) = -integral(a div B) = 0. B = curl A is
    the physically relevant case (it is what the engine's rho_L3 is fed), and a
    uniform B would only test the weaker corollary.

    `abs_rho`, not `mag` -- the engine's energy dict already uses "mag" for the
    MAGNETIC energy, and the two readings of that word invite exactly the wrong
    conclusion from a results table.
    """
    N, L, dx, p2 = campaign_ic
    g = jnp.asarray(np.linspace(-L / 2, L / 2, N, endpoint=False))
    X, Y, Z = jnp.meshgrid(g, g, g, indexing="ij")
    B = curlA(jnp.sin(2 * np.pi * Y / L), jnp.cos(2 * np.pi * Z / L),
              jnp.sin(2 * np.pi * X / L), dx)

    net, abs_rho = {}, {}
    for m in ("wrapped", "bilinear", "naive"):
        r = rho_L3(p2, B, dx, 0.05, m)
        net[m] = float(jnp.sum(r))
        abs_rho[m] = float(jnp.sum(jnp.abs(r)))

    assert net["naive"] == pytest.approx(0.0, abs=1e-9)   # exactly, not approximately
    assert abs(net["wrapped"]) > 1e2                      # curl A: ~-413, not ~3495
    assert abs(net["bilinear"]) > 1e2
    # the cancellation made visible: biggest local values, zero net
    assert abs_rho["naive"] > abs_rho["wrapped"]


def test_vanishing_net_rho_does_not_bound_the_L3_energy(campaign_ic):
    """net(rho) = 0 says nothing about the energy the engine actually forms.

    `E_L3_electric` is 0.5*C*sum(rho*s) with a spatially varying A0, so
    sum(rho) = 0 does not imply sum(rho*s) = 0: the weighting can pick out the
    cancelling spikes instead of cancelling with them. Under naive those spikes sit
    on the branch-cut sheet, and whether they survive depends entirely on how A0
    correlates with that sheet -- so the L3 energy is configuration-dependent and
    NOT bounded by the vanishing net.

    This test exists because the docstring above used to assert the opposite ("the
    L3 coupling contributes nothing"), and then, once corrected, quoted two numbers
    from an A0 that lived nowhere in the tree. A measurement a reader cannot
    reproduce from the repo is the same failure this file is about, one level up.
    Both A0s are defined here, deterministically, and the two rows bracket the
    claim: naive is the SMALLEST contributor under one and the LARGEST under the
    other.
    """
    N, L, dx, p2 = campaign_ic
    g = jnp.asarray(np.linspace(-L / 2, L / 2, N, endpoint=False))
    X, Y, Z = jnp.meshgrid(g, g, g, indexing="ij")
    B = curlA(jnp.sin(2 * np.pi * Y / L), jnp.cos(2 * np.pi * Z / L),
              jnp.sin(2 * np.pi * X / L), dx)
    C, eps_a = 400.0, 0.05

    smooth = jnp.cos(2 * np.pi * X / L) * jnp.cos(2 * np.pi * Y / L)
    rough = jnp.asarray(np.random.default_rng(0).standard_normal((N, N, N)))

    def eelec(s, mode):
        return float(E_L3_electric(p2, B, s, dx, C, eps_a, mode))

    for s, label in ((smooth, "smooth"), (rough, "rough")):
        assert abs(eelec(s, "naive")) > 1.0, f"{label}: naive L3 energy is not zero"

    # smooth A0 varies slowly across the cut sheet, so the spikes largely cancel
    assert abs(eelec(smooth, "naive")) < abs(eelec(smooth, "wrapped"))
    # a rough A0 decorrelates them, and the largest |rho| of the three wins
    assert abs(eelec(rough, "naive")) > abs(eelec(rough, "wrapped"))

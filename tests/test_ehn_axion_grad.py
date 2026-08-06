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

from jax_solitons.ehn.energy import axion_grad                  # noqa: E402

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

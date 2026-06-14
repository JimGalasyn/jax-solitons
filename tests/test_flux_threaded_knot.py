"""The L_2 + L_3 coupled seed: a flux-threaded T(p,q) knot (Paper 16 sec.L_3).

A single doublet psi = rho e^{i chi} zeta carrying BOTH a Higgs flux vortex
(modulus -> 0 on the knot curve) AND a Hopf texture of charge Q_H = p*m, ready
to relax in models.gauged_faddeev. These gate the seed's structure; the deep
relaxation + kappa = R/a0 extraction is a (GPU-tier) campaign, not a unit test.
"""
from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest

from jax_solitons.grid import BoxGrid
from jax_solitons.models.gauged_faddeev import (
    aspect_ratio,
    gauged_faddeev_model,
    hopf_charge_doublet,
    unpack,
)
from jax_solitons.seeds import flux_threaded_knot_seed

N, L = 64, 20.0
LOCK_CASES = [(2, 3, 1, 2), (2, 1, 1, 2), (1, 4, 2, 2)]


@pytest.fixture(scope="module")
def grid():
    return BoxGrid(N=N, L=L, dtype=jnp.float64)


def test_seed_shape_and_finite_energy(grid):
    """The coupled seed is a (7, N, N, N) state with finite coupled energy."""
    s = flux_threaded_knot_seed(grid, 2, 3, 1)
    assert s.shape == (7, N, N, N)
    E = float(gauged_faddeev_model(e=1.0, v=1.0).energy(s, grid))
    assert np.isfinite(E)


@pytest.mark.parametrize("p,q,m,expected", LOCK_CASES)
def test_hopf_charge_locks_to_pm(grid, p, q, m, expected):
    """The Hopf charge of n[psi] locks to |Q_H| = p*m even on the coupled
    doublet (n depends only on the Bloch direction, not the modulus/flux)."""
    s = flux_threaded_knot_seed(grid, p, q, m)
    assert round(abs(float(hopf_charge_doublet(s, grid)))) == expected


def test_higgs_flux_tube_core(grid):
    """A Higgs vortex core runs along the knot: |psi| -> 0 on the curve and -> v
    in the bulk (the flux tube the L_2 sector threads through the knot)."""
    s = flux_threaded_knot_seed(grid, 2, 3, 1, v=1.0)
    psi1, psi2, _ = unpack(s)
    mod = np.sqrt(np.asarray(jnp.abs(psi1) ** 2 + jnp.abs(psi2) ** 2))
    assert mod.min() < 0.1            # core: psi -> 0 on the knot curve
    assert mod.max() == pytest.approx(1.0, abs=0.05)   # bulk vacuum |psi| = v


def test_requires_coprime_and_nonzero_e(grid):
    with pytest.raises(ValueError, match="gcd"):
        flux_threaded_knot_seed(grid, 2, 4, 1)
    with pytest.raises(ValueError, match="nonzero"):
        flux_threaded_knot_seed(grid, 2, 3, 1, e=0.0)


def test_aspect_ratio_is_a_sane_measurement(grid):
    """kappa = R/a0 is a sane positive O(10) measurement of the engine's own
    soliton. The theory TARGET is not asserted here -- it lives in nwt-substrate
    (see the oracle gate below), per the engine<->theory firewall."""
    kappa, R, a0 = aspect_ratio(flux_threaded_knot_seed(grid, 2, 3, 1), grid, 2, 3)
    assert R > 0 and a0 > 0
    assert 1.0 < kappa < 30.0


def test_aspect_ratio_target_comes_from_the_oracle(grid):
    """Cross-engine firewall: the kappa TARGET is the substrate aspect ratio
    nwt_substrate.isa.constants.KAPPA_MACKEN ~ 9.844 (the alpha-derived closed
    form, Paper 17), NOT a value rolled locally. Self-skips when the oracle
    (optional `oracle` extra) is absent.

    The measured-vs-target EQUIVALENCE gate is the tuned relaxation campaign
    (Paper 16's f_pi^2/e_Sk^2 at the BPS point + resolution convergence); here
    we lock down only that the engine's aspect_ratio is the same observable and
    that the target is sourced from the oracle, not hard-coded."""
    consts = pytest.importorskip("nwt_substrate.isa.constants")
    kappa_target = consts.KAPPA_MACKEN
    # the oracle's own closed-form identity: kappa^2 * sqrt2 = 1/alpha
    assert kappa_target**2 * np.sqrt(2.0) == pytest.approx(
        1.0 / consts.ALPHA_SUBSTRATE, rel=1e-9)
    # the engine measures the same kind of quantity (positive aspect ratio)
    kappa_meas, _, _ = aspect_ratio(
        flux_threaded_knot_seed(grid, 2, 3, 1), grid, 2, 3)
    assert kappa_meas > 0 and 5.0 < kappa_target < 15.0


def test_seed_rejects_nonpositive_xi(grid):
    """xi (Higgs healing length) is a divisor in the modulus/gauge profiles;
    a non-positive value must fail loudly, not produce a broken seed (P9)."""
    with pytest.raises(ValueError, match="xi"):
        flux_threaded_knot_seed(grid, 2, 3, 1, xi=0.0)


def test_aspect_ratio_rejects_nonpositive_v(grid):
    """v=0 would divide-by-zero the core indicator; reject it early."""
    with pytest.raises(ValueError, match="must be positive"):
        aspect_ratio(flux_threaded_knot_seed(grid, 2, 3, 1), grid, 2, 3, v=0.0)


def test_aspect_ratio_rejects_coreless_state(grid):
    """A filled-in state (|psi| = v everywhere) has no flux tube to measure;
    aspect_ratio must raise rather than return inf/NaN."""
    flat = jnp.zeros((7, grid.N, grid.N, grid.N), grid.dtype).at[0].set(1.0)
    with pytest.raises(ValueError, match="no Higgs-core weight"):
        aspect_ratio(flat, grid, 2, 3, v=1.0)

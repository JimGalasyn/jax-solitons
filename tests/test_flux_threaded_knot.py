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


def test_aspect_ratio_is_order_ten(grid):
    """kappa = R/a0 on the seed is a sane positive O(10) number (the Paper 16
    sec.L_3 observable; the target pi^2 ~ 9.87 is a relaxed-equilibrium claim)."""
    kappa, R, a0 = aspect_ratio(flux_threaded_knot_seed(grid, 2, 3, 1), grid, 2, 3)
    assert R > 0 and a0 > 0
    assert 1.0 < kappa < 30.0

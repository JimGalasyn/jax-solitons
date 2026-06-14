"""Coupled L_2 + L_3 gauged Faddeev-Skyrme-Higgs model (Paper 16 L_NWT,
reading A: the unit field n is slaved to a single C^2 doublet psi).

Two correctness pillars:

  1. EXACT lattice gauge invariance (P8) -- the headline property that proves
     the L_2 / L_3 coupling is built right: the compact-link energy is invariant
     under psi -> e^{i chi} psi, U_i -> e^{i chi(x)} U_i e^{-i chi(x+e_i)}, to
     machine precision (not merely O(dx^2)).

  2. Decoupling limits -- the coupled model must reproduce its two parents:
     with the Skyrme couplings off and psi2 = 0 it IS the gauged abelian-Higgs
     model; the Skyrme sub-terms on a unit doublet ARE the Faddeev energy.
"""
from __future__ import annotations

import numpy as np
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import pytest

from jax_solitons.grid import BoxGrid
from jax_solitons.models import abelian_higgs_model, faddeev_cp1_model
from jax_solitons.models.gauged_faddeev import (
    SkyrmeE2Term,
    SkyrmeE4Term,
    gauged_faddeev_model,
    hopf_charge_doublet,
    n_from_doublet,
)
from jax_solitons.seeds import rational_map_hopfion_cp1

N, L = 16, 12.0


@pytest.fixture(scope="module")
def grid():
    return BoxGrid(N=N, L=L, dtype=jnp.float64)


def _random_state(seed):
    """A generic (7, N, N, N) state: random doublet + gauge field."""
    rng = np.random.default_rng(seed)
    s = rng.standard_normal((7, N, N, N))
    # offset the modulus off zero so |psi| ~ O(1) (a generic, non-degenerate
    # configuration -- gauge invariance must hold regardless)
    s[0] += 1.0
    return jnp.asarray(s)


def _gauge_transform(state, chi, e, dx):
    """psi -> e^{i chi} psi ;  A_i -> A_i + (chi(x) - chi(x+e_i)) / (e dx)."""
    psi1 = (state[0] + 1j * state[1]) * jnp.exp(1j * chi)
    psi2 = (state[2] + 1j * state[3]) * jnp.exp(1j * chi)
    A = state[4:7]
    A_new = [A[i] + (chi - jnp.roll(chi, -1, axis=i)) / (e * dx) for i in range(3)]
    return jnp.stack([jnp.real(psi1), jnp.imag(psi1),
                      jnp.real(psi2), jnp.imag(psi2), *A_new])


def test_gauge_invariance_exact(grid):
    """Total energy is invariant under a lattice gauge transform to ~machine
    precision -- the compact-link exactness contract, all six couplings on."""
    e = 1.3
    model = gauged_faddeev_model(e=e, v=1.0, c2=1.0, c4=4.0, theta=0.7)
    state = _random_state(0)
    rng = np.random.default_rng(1)
    chi = jnp.asarray(rng.standard_normal((N, N, N)))      # arbitrary gauge fn

    E0 = float(model.energy(state, grid))
    E1 = float(model.energy(_gauge_transform(state, chi, e, grid.dx), grid))
    assert abs(E1 - E0) < 1e-9 * (1.0 + abs(E0)), f"E0={E0}, E1={E1}"


def test_n_is_gauge_invariant_pointwise(grid):
    """n^a = psi^dag sigma^a psi / |psi|^2 is unchanged by the U(1) phase, so
    d_i n needs no covariant derivative (Paper 16 writes d_i n, not D_i n)."""
    state = _random_state(2)
    chi = jnp.asarray(np.random.default_rng(3).standard_normal((N, N, N)))
    n0 = n_from_doublet(state)
    n1 = n_from_doublet(_gauge_transform(state, chi, 1.0, grid.dx))
    assert float(jnp.max(jnp.abs(n1 - n0))) < 1e-10


def test_reduces_to_abelian_higgs(grid):
    """With the Skyrme couplings off (c2 = c4 = 0) and psi2 = 0, the coupled
    model is exactly the gauged abelian-Higgs model on (psi1, A)."""
    e, lam, v = 1.0, 2.0, 1.0
    rng = np.random.default_rng(4)
    psi1 = rng.standard_normal((2, N, N, N))
    A = rng.standard_normal((3, N, N, N))
    coupled = jnp.asarray(np.concatenate(
        [psi1, np.zeros((2, N, N, N)), A]))           # (7,...), psi2 = 0
    ah = jnp.asarray(np.concatenate([psi1, A]))       # (5,...)

    e_coupled = float(gauged_faddeev_model(e=e, lam=lam, v=v, c2=0.0, c4=0.0)
                      .energy(coupled, grid))
    e_ah = float(abelian_higgs_model(e=e, lam=lam, v=v).energy(ah, grid))
    assert e_coupled == pytest.approx(e_ah, rel=1e-12)


def test_skyrme_terms_match_faddeev(grid):
    """The L_3 sub-terms (Skyrme E2 + E4) on a unit doublet equal the Faddeev
    energy on the same CP^1 spinor -- the L_3 sector reuses faddeev faithfully."""
    z = rational_map_hopfion_cp1(grid, R=2.5, n=1, m=1)   # unit (4,N,N,N)
    state = jnp.concatenate([z, jnp.zeros((3, N, N, N), z.dtype)])  # A = 0
    c4 = 4.0
    e_skyrme = float(SkyrmeE2Term(c2=1.0)(state, grid)
                     + SkyrmeE4Term(c4=c4)(state, grid))
    e_fadd = float(faddeev_cp1_model(c4=c4).energy(z, grid))
    assert e_skyrme == pytest.approx(e_fadd, rel=1e-9)


def test_charges_and_e_zero_guard(grid):
    """Charges evaluate finitely; e = 0 is rejected (no compact-link limit)."""
    model = gauged_faddeev_model(e=1.0)
    state = _random_state(5)
    for charge in model.charges:
        assert np.isfinite(float(charge(state, grid)))
    assert np.isfinite(float(hopf_charge_doublet(state, grid)))
    with pytest.raises(ValueError, match="nonzero"):
        gauged_faddeev_model(e=0.0)

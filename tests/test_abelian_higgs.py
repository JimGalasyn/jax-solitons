"""Gauged abelian-Higgs model: the exactness gates (P8) + sanity.

The load-bearing test is EXACT lattice U(1) gauge invariance — the compact
(link) discretization's reason to exist, the gauge analogue of the area form's
exact quantization for Faddeev.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from jax_solitons.grid import BoxGrid  # noqa: E402
from jax_solitons.models.abelian_higgs import (  # noqa: E402
    abelian_higgs_model,
    unpack,
    vortex_seed,
)
from jax_solitons.steppers.adam import adam_flow  # noqa: E402


def _gauge_transform(state, chi, e, dx):
    """Apply a local U(1) gauge transform: phi -> e^{i chi} phi,
    A_i += (chi(x) - chi(x+e_i))/(e dx)  (the compact-link convention)."""
    phi, A = (np.asarray(x) for x in unpack(np.asarray(state)))
    phi2 = phi * np.exp(1j * chi)
    A2 = A.copy()
    for i in range(3):
        A2[i] = A[i] + (chi - np.roll(chi, -1, axis=i)) / (e * dx)
    return jnp.asarray(np.stack([phi2.real, phi2.imag, A2[0], A2[1], A2[2]]))


@pytest.mark.parametrize("e", [1.0, 1.7])
def test_energy_exactly_gauge_invariant(e):
    """The whole reason for the compact link form: U(1) gauge invariance is
    EXACT on the lattice, for an arbitrary field and coupling (not O(dx^2))."""
    g = BoxGrid(N=16, L=8.0, dtype=jnp.float64)
    model = abelian_higgs_model(e=e)
    rng = np.random.default_rng(0)
    s = jnp.asarray(rng.standard_normal((5, 16, 16, 16)))
    chi = rng.standard_normal((16, 16, 16))
    s2 = _gauge_transform(s, chi, e, g.dx)
    E0 = float(model.energy(s, g))
    assert abs(float(model.energy(s2, g)) - E0) < 1e-9 * abs(E0)


def test_magnetic_flux_is_gauge_invariant():
    g = BoxGrid(N=16, L=8.0, dtype=jnp.float64)
    model = abelian_higgs_model(e=1.3)
    rng = np.random.default_rng(1)
    s = jnp.asarray(rng.standard_normal((5, 16, 16, 16)))
    s2 = _gauge_transform(s, rng.standard_normal((16, 16, 16)), 1.3, g.dx)
    flux = model.charges[0]
    assert abs(float(flux(s2, g)) - float(flux(s, g))) < 1e-9


def test_vacuum_has_zero_energy():
    g = BoxGrid(N=12, L=6.0, dtype=jnp.float64)
    vac = np.zeros((5, 12, 12, 12))
    vac[0] = 1.0                                   # phi = v = 1, A = 0
    assert abs(float(abelian_higgs_model().energy(jnp.asarray(vac), g))) < 1e-12


def test_relaxation_lowers_vortex_energy():
    g = BoxGrid(N=20, L=10.0, dtype=jnp.float64)
    model = abelian_higgs_model(e=1.0)             # self-dual lam = 2
    s = vortex_seed(g, n=1, e=1.0)
    E0 = float(model.energy(s, g))
    sr, _ = adam_flow(model, s, g, lr=2e-3, steps=200)
    assert float(model.energy(sr, g)) < E0

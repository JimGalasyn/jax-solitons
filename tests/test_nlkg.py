"""Tier-1 tests for the relativistic NLKG model (models.nlkg).

Mirrors the test_gpe idiom: module-level GRID/MODEL, scalars via float(),
np.allclose comparisons. The headline contracts are (1) the |Phi|=F0 vacuum is
a zero-energy fixed point, (2) the velocity-Verlet real-time evolution conserves
energy to < 0.1% (the relativistic wave EOM is Hamiltonian), (3) the autodiff
acceleration reproduces the NLKG EOM, and (4) vortex circulation is quantized to
2*pi.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from jax_solitons.grid import BoxGrid
from jax_solitons.models import nlkg
from jax_solitons.steppers.verlet import kinetic_energy, make_verlet_step

GRID = BoxGrid(N=32, L=16.0, dtype=jnp.float64)
MODEL = nlkg.nlkg_model(lam=1.0, f0sq=1.0)


def test_vacuum_is_zero_energy_fixed_point():
    s = nlkg.vortex_seed(GRID, f0=1.0)               # uniform |Phi|=1
    assert float(MODEL.energy(s, GRID)) == pytest.approx(0.0, abs=1e-9)
    force = jax.grad(lambda x: MODEL.energy(x, GRID))(s)
    assert float(jnp.max(jnp.abs(force))) == pytest.approx(0.0, abs=1e-9)


def test_state_roundtrip_real_complex_view():
    rng = np.random.default_rng(0)
    phi = jnp.asarray(rng.standard_normal((8, 8, 8)) + 1j * rng.standard_normal((8, 8, 8)))
    assert np.allclose(np.asarray(nlkg.to_complex(nlkg.from_complex(phi))), np.asarray(phi))


def test_mass_term_added_only_when_nonzero():
    assert len(nlkg.nlkg_model(m0=0.0).terms) == 2
    m = nlkg.nlkg_model(m0=0.5)
    assert len(m.terms) == 3
    # m0/2 |Phi|^2 over uniform |Phi|=1 vacuum = 0.5*m0*V
    s = nlkg.vortex_seed(GRID, f0=1.0)
    vol = GRID.L**3
    assert float(m.energy(s, GRID)) == pytest.approx(0.5 * 0.5 * vol, rel=1e-6)


def test_autodiff_acceleration_matches_nlkg_eom():
    """-(1/dx^3) dE/dPhi must equal lap Phi - lam(|Phi|^2-F0^2)Phi on a smooth
    field, where lap is the matching (forward-difference) discrete Laplacian.
    Compared against the spectral Laplacian, they agree to the O(dx^2) stencil
    error -- crucially NOT O(1), which a wrong sign / conjugation would give."""
    rng = np.random.default_rng(1)
    KX, KY, KZ, K2 = GRID.k_vectors()
    raw = 1.0 + 0.1 * (rng.standard_normal((32,) * 3) + 1j * rng.standard_normal((32,) * 3))
    phi = jnp.fft.ifftn(jnp.fft.fftn(jnp.asarray(raw)) * jnp.exp(-0.5 * np.asarray(K2)))
    s = nlkg.from_complex(phi).astype(jnp.float64)

    a = -jax.grad(lambda x: MODEL.energy(x, GRID))(s) / GRID.dx**3
    a_complex = a[0] + 1j * a[1]
    lap = jnp.fft.ifftn(-K2 * jnp.fft.fftn(phi))
    a_spectral = lap - (jnp.abs(phi) ** 2 - 1.0) * phi
    rel = float(jnp.linalg.norm(a_complex - a_spectral) / jnp.linalg.norm(a_spectral))
    assert rel < 0.05


def test_energy_conserved_under_verlet():
    """Relativistic wave dynamics is Hamiltonian: a settled vortex ring evolved
    by velocity-Verlet conserves total energy (potential + kinetic) to < 0.1%."""
    grid = BoxGrid(N=48, L=20.0, dtype=jnp.float64)
    model = nlkg.nlkg_model(lam=1.0, f0sq=1.0)
    s = nlkg.vortex_seed(grid, rings=[dict(R=5.0, xi=1.0, center=(0, 0, 0),
                                           axis="z", sign=1)])
    gE = jax.jit(jax.grad(lambda x: model.energy(x, grid)))
    dtau = 0.1 * grid.dx**2
    for _ in range(300):                              # imaginary-time settle
        s = s - dtau * gE(s) / grid.dx**3
    v = jnp.zeros_like(s)
    dt = 0.4 * grid.dx / np.sqrt(3.0)
    step = make_verlet_step(model, grid, dt)

    def E():
        return float(model.energy(s, grid) + kinetic_energy(v, grid))

    e0 = E()
    for _ in range(1500):
        s, v = step(s, v)
    assert abs(E() - e0) / abs(e0) < 1e-3


@pytest.mark.parametrize("sign", [+1, -1])
def test_circulation_quantized_on_straight_vortex(sign):
    """oint grad(sigma).dl around a straight winding-`sign` vortex = sign * 2*pi."""
    s = nlkg.vortex_seed(GRID, lines=[dict(xi=1.0, axis="z", sign=sign)])
    gamma = nlkg.circulation(s, GRID, axis="z", center=(0.0, 0.0), half_cells=8)
    assert gamma == pytest.approx(sign * 2.0 * np.pi, abs=1e-6)


def test_circulation_zero_with_no_vortex():
    s = nlkg.vortex_seed(GRID, f0=1.0)                # vacuum: no winding
    gamma = nlkg.circulation(s, GRID, axis="z", center=(0.0, 0.0), half_cells=8)
    assert gamma == pytest.approx(0.0, abs=1e-6)

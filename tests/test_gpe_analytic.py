"""Tier-2 validation: GPE against closed-form solutions.

Engine convention (models/gpe.py + steppers/splitstep.py):
    i d_t psi = -1/2 grad^2 psi + g (|psi|^2 - 1) psi
so the uniform vacuum psi = 1 is stationary, the healing length is
1/sqrt(g), and linearizing psi = 1 + delta gives the Bogoliubov
dispersion  omega(k) = k sqrt(g + k^2/4)  — the classic validation for
any GPE code. The planar dark soliton is psi = tanh(sqrt(g) x) with
energy (4/3) sqrt(g)... per unit area at g=1: 4/3.
"""

import jax.numpy as jnp
import numpy as np

from jax_solitons import BoxGrid
from jax_solitons.models import gpe_model
from jax_solitons.steppers import splitstep_evolve


def test_bogoliubov_dispersion():
    """Measured oscillation frequency of a small density wave matches
    omega(k) = k sqrt(g + k^2/4) to ~1%."""
    g = 1.0
    grid = BoxGrid(N=32, L=8.0)
    x = np.asarray(grid.axis())
    for m in (1, 2):                       # two modes, two omegas
        k = 2.0 * np.pi * m / grid.L
        omega = k * np.sqrt(g + k**2 / 4.0)
        psi0 = jnp.asarray((1.0 + 0.01 * np.cos(k * x))[:, None, None]
                           * np.ones((1, grid.N, grid.N)),
                           dtype=jnp.complex64)
        cosk = jnp.asarray(np.cos(k * x)[:, None, None])

        amps, times = [], []
        dt, every, steps = 0.004, 25, 2000

        def obs(i, psi, cosk=cosk):
            amps.append(float(jnp.sum(cosk * (jnp.abs(psi) ** 2 - 1.0))))
            times.append(i * dt)
            return 0

        splitstep_evolve(grid, psi0, dt=dt, steps=steps, g=g,
                         imaginary_time=False, observe_every=every,
                         observer=obs)
        a, t = np.asarray(amps), np.asarray(times)
        # linear-theory standing wave: a(t) = a0 cos(omega t); fit omega by
        # least squares over a frequency scan around the analytic value
        scan = omega * np.linspace(0.9, 1.1, 401)
        resid = [np.linalg.norm(a - a[0] * np.cos(w * t)) for w in scan]
        w_fit = scan[int(np.argmin(resid))]
        assert abs(w_fit - omega) / omega < 0.01, \
            f"mode m={m}: omega fit {w_fit:.4f} vs analytic {omega:.4f}"


def test_dark_soliton_profile_energy_and_stationarity():
    """A planar kink-antikink pair (separation >> healing length) has the
    analytic energy 2 * (4/3) * area and is stationary in real time."""
    g = 1.0
    grid = BoxGrid(N=64, L=16.0)
    x = np.asarray(grid.axis())
    prof = np.tanh(x + grid.L / 4) * np.tanh(grid.L / 4 - x)
    psi0 = jnp.asarray(prof[:, None, None] * np.ones((1, grid.N, grid.N)),
                       dtype=jnp.complex64)

    model = gpe_model(g=g)
    E = float(model.energy(psi0, grid))
    E_analytic = 2.0 * (4.0 / 3.0) * grid.L**2
    assert abs(E - E_analytic) / E_analytic < 0.02, \
        f"kink-pair energy {E:.2f} vs analytic {E_analytic:.2f}"

    # real-time stationarity: density drift stays at rounding scale
    psi1, _ = splitstep_evolve(grid, psi0, dt=0.005, steps=400, g=g,
                               imaginary_time=False)
    drift = float(jnp.max(jnp.abs(jnp.abs(psi1) ** 2 - jnp.abs(psi0) ** 2)))
    assert drift < 5e-3, f"dark soliton not stationary: drift {drift:.2e}"


def test_imaginary_time_relaxes_to_vacuum():
    """Imaginary-time split-step drives a perturbed state to the |psi| = 1
    vacuum (the relaxer's fixed point is the analytic ground state)."""
    grid = BoxGrid(N=24, L=8.0)
    rng = np.random.default_rng(3)
    psi0 = jnp.asarray(1.0 + 0.2 * rng.standard_normal((24, 24, 24))
                       + 0.2j * rng.standard_normal((24, 24, 24)),
                       dtype=jnp.complex64)
    psi1, _ = splitstep_evolve(grid, psi0, dt=0.02, steps=600, g=1.0,
                               imaginary_time=True)
    dens_err = float(jnp.max(jnp.abs(jnp.abs(psi1) ** 2 - 1.0)))
    assert dens_err < 1e-3, f"vacuum not reached: {dens_err:.2e}"
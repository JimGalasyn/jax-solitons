"""GPE model + split-step smoke tests.

The stepper was additionally cross-validated against the source numpy
split-step (max |src - port| = 9e-16 after 10 steps at N=32/fp64).
"""

import jax.numpy as jnp
import numpy as np
import pytest

from jax_solitons import BoxGrid
from jax_solitons.models import gpe_model
from jax_solitons.steppers import splitstep_evolve

GRID = BoxGrid(N=24, L=10.0)
MODEL = gpe_model(g=1.0)


def test_vacuum_is_fixed_point():
    psi = jnp.ones((GRID.N,) * 3, dtype=jnp.complex64)
    assert float(MODEL.energy(psi, GRID)) == pytest.approx(0.0, abs=1e-6)
    out, _ = splitstep_evolve(GRID, psi, dt=0.1, steps=10)
    assert np.allclose(np.asarray(out), np.asarray(psi), atol=1e-6)


def test_imaginary_time_relaxes_noise():
    rng = np.random.default_rng(3)
    psi = jnp.asarray(
        1.0 + 0.3 * rng.standard_normal((GRID.N,) * 3)
        + 0.3j * rng.standard_normal((GRID.N,) * 3)
    )
    e0 = float(MODEL.energy(psi, GRID))
    energies = []
    out, _ = splitstep_evolve(
        GRID, psi, dt=0.05, steps=60, observe_every=10,
        observer=lambda i, p: energies.append(float(MODEL.energy(p, GRID))),
    )
    e1 = float(MODEL.energy(out, GRID))
    assert e1 < 0.05 * e0, f"did not relax: {e0:.1f} -> {e1:.1f}"
    assert all(b <= a * 1.001 for a, b in zip(energies, energies[1:])), \
        "energy not (near-)monotone in imaginary time"

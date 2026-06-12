"""Smoke tests for the working core (grid, model protocol, run config)."""

import jax.numpy as jnp

from jax_solitons import BoxGrid, Model, RunConfig


def test_grid_basics():
    g = BoxGrid(N=16, L=8.0)
    assert abs(g.dx - 0.5) < 1e-12
    X, Y, Z = g.coords()
    assert X.shape == (16, 16, 16)
    assert X.dtype == jnp.float32
    g64 = BoxGrid(N=16, L=8.0, dtype=jnp.float64)
    assert g64.axis().dtype in (jnp.float64, jnp.float32)  # float32 if x64 disabled


def test_model_energy_sums_terms():
    class Quadratic:
        name = "quad"

        def __call__(self, state, grid):
            return jnp.sum(state**2)

    class Quartic:
        name = "quart"

        def __call__(self, state, grid):
            return jnp.sum(state**4)

    g = BoxGrid(N=4, L=4.0)
    m = Model(name="toy", terms=(Quadratic(), Quartic()))
    state = jnp.full((4, 4, 4), 2.0, dtype=g.dtype)
    # 64 cells * (4 + 16)
    assert abs(float(m.energy(state, g)) - 64 * 20.0) < 1e-3


def test_runconfig_roundtrip_and_hash():
    cfg = RunConfig(model="faddeev", N=96, L=18.0, steps=400, dt=0.02,
                    params={"c4": 6.0, "Q": -11})
    cfg2 = RunConfig.from_json(cfg.to_json())
    assert cfg2 == cfg
    assert cfg.config_hash() == cfg2.config_hash()
    # hash must be sensitive to params
    cfg3 = RunConfig(model="faddeev", N=96, L=18.0, steps=400, dt=0.02,
                     params={"c4": 6.0, "Q": -12})
    assert cfg3.config_hash() != cfg.config_hash()
    assert cfg.run_name().startswith("faddeev_N96_")

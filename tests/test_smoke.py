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


def test_vmap_batch_dynamics_matches_per_sample():
    """R2: a vmapped Verlet step over a batch of fields is identical to
    stepping each field individually (the batch axis is free)."""
    import jax
    import numpy as np

    from jax_solitons.models import faddeev_model
    from jax_solitons.seeds import rational_map_hopfion
    from jax_solitons.steppers.verlet import make_verlet_step

    g = BoxGrid(N=16, L=6.0)
    model = faddeev_model(c4=4.0)
    step = make_verlet_step(model, g, dt=0.005)
    batch_step = jax.jit(jax.vmap(step))

    seeds = jnp.stack([rational_map_hopfion(g, R=r) for r in (1.5, 1.8)])
    vels = jnp.zeros_like(seeds)
    bn, bv = seeds, vels
    for _ in range(3):
        bn, bv = batch_step(bn, bv)
    for k, r in enumerate((1.5, 1.8)):
        n, v = seeds[k], vels[k]
        for _ in range(3):
            n, v = step(n, v)
        assert np.allclose(np.asarray(bn[k]), np.asarray(n), atol=1e-6), \
            f"vmap mismatch for sample {k} (R={r})"

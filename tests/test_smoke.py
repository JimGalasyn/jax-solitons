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


def test_cp1_frame_matches_n_frame_on_seed():
    """The CP^1 spinor frame is the same physics: energy and Hopf charge of
    the spinor model on the spinor seed match the n-frame model on the
    n-frame seed (both built from the same rational map)."""
    import numpy as np

    from jax_solitons.models import faddeev_cp1_model, faddeev_model
    from jax_solitons.models.faddeev import n_from_state
    from jax_solitons.seeds import rational_map_hopfion, rational_map_hopfion_cp1
    from jax_solitons.topology import hopf_charge

    g = BoxGrid(N=24, L=8.0)
    nf = rational_map_hopfion(g, R=2.0)
    z = rational_map_hopfion_cp1(g, R=2.0)
    assert z.shape == (4, 24, 24, 24)
    assert np.allclose(np.asarray(n_from_state(z)), np.asarray(nf), atol=1e-6)

    mn = faddeev_model(c4=4.0)
    mz = faddeev_cp1_model(c4=4.0)
    assert np.isclose(float(mz.energy(z, g)), float(mn.energy(nf, g)),
                      rtol=1e-6)
    qz = float(mz.charges[0](z, g))
    qn = float(hopf_charge(nf, g))
    assert np.isclose(qz, qn, atol=1e-6)
    # retraction restores |Z| = 1
    drifted = z * 1.7
    back = mz.constraint.retract(drifted)
    nrm = np.asarray(jnp.sum(back**2, axis=0))
    assert np.allclose(nrm, 1.0, atol=1e-6)


def test_faddeev_energy_density_integrates_to_energy():
    """sum(density) * dx^3 == model.energy, by construction and forever."""
    import numpy as np

    from jax_solitons.models import faddeev_energy_density, faddeev_model
    from jax_solitons.seeds import rational_map_hopfion

    g = BoxGrid(N=24, L=8.0)
    nf = rational_map_hopfion(g, R=2.0)
    e = faddeev_energy_density(nf, g, c4=4.0)
    assert e.shape == (24, 24, 24)
    assert np.isclose(float(jnp.sum(e)) * g.dx**3,
                      float(faddeev_model(c4=4.0).energy(nf, g)), rtol=1e-5)


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

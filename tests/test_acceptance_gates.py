"""Acceptance gates for the engine port (the regression contract).

These encode the validation chain of the source research codebase.
Live gates run smoke-sized (CPU CI);
the port was additionally cross-validated bit-identically against the
source engine at N=64/x64 (seed E and Q_H matched to the last digit).
Remaining skips flip as their modules land; a gate that lands must never
regress.

Reference values from validated runs of the source engine:
  - unit hopfion: area-form relaxation holds Q_H ~ 0.998 at the E ~ 1152 min
  - VK ratio: E(Q=2)/E(Q=1) = 1.604 (published 1.623), exponent ~ Q^{3/4}
  - trefoil Q=7 smoke: core-curve knot determinant 3 held through relaxation
  - persistence: real-time leapfrog conserves dH to ~0.00% with det held
"""

import jax.numpy as jnp
import pytest

from jax_solitons import BoxGrid
from jax_solitons.models import faddeev_model
from jax_solitons.seeds import rational_map_hopfion
from jax_solitons.steppers import arrested_flow, kinetic_energy, verlet_evolve
from jax_solitons.topology import hopf_charge

pytestmark = pytest.mark.acceptance

# Smoke configuration: small enough for CPU CI, large enough that the
# area-form charge is honest (~8 grid points across the seed tube radius).
GRID = BoxGrid(N=32, L=8.0)
MODEL = faddeev_model(c4=4.0)
SEED_R = 2.0


@pytest.fixture(scope="module")
def seed():
    return rational_map_hopfion(GRID, R=SEED_R)


@pytest.fixture(scope="module")
def relaxed(seed):
    state, hist = arrested_flow(MODEL, seed, GRID, dt=2e-4, steps=150,
                                log_every=25)
    return state, hist


def test_gate_unit_hopfion_charge_held(seed, relaxed):
    """Area-form Q_H reads ~1 on the rational-map seed and is HELD through
    monotone descent (the naive discretization unwinds here)."""
    q0 = float(hopf_charge(seed, GRID))
    assert abs(q0 - 1.0) < 0.05, f"seed charge dishonest: {q0}"
    state, hist = relaxed
    q1 = float(hopf_charge(state, GRID))
    assert abs(q1 - 1.0) < 0.05, f"charge not held through flow: {q1}"
    energies = [h[1] for h in hist]
    assert all(b <= a + 1e-6 for a, b in zip(energies, energies[1:])), \
        "flow not monotone"
    assert energies[-1] < 0.8 * energies[0], "flow did not descend"


@pytest.mark.skip(reason="needs near-converged Q=1 and Q=2 relaxations; "
                  "too slow for CPU CI smoke — runs in the GPU validation tier")
def test_gate_vk_q1_q2_ratio():
    """E(Q=2)/E(Q=1) in [1.55, 1.70] (published 1.623; engine 1.604)."""


def test_gate_core_tracer_closed_ring(relaxed):
    """The tracer finds the relaxed Q=1 hopfion core {n1=0, n2=0, n3>0} as
    exactly ONE closed loop with a sane circumference (a planar ring of
    radius ~R has length ~2*pi*R)."""
    import numpy as np

    from jax_solitons.measure import curve_length, trace_curves

    n, _ = relaxed
    cores = trace_curves(n[0], n[1], GRID, mask=np.asarray(n[2]) > 0)
    assert len(cores) == 1, f"expected 1 core loop, got {len(cores)}"
    ln = curve_length(cores[0])
    assert 6.0 < ln < 18.0, f"core circumference implausible: {ln:.2f}"


@pytest.mark.skip(reason="determinant needs the knot-identification "
                  "(Alexander) integration; the tracer half is live above")
def test_gate_trefoil_q7_determinant_held():
    """Smoke-sized Q_H=7 trefoil: core-curve determinant 3 held through relax."""


def test_gate_persistence_energy_conservation(relaxed):
    """Real-time constrained Verlet: |dH/H| small and Q_H held over a short
    run started from the (partially) relaxed hopfion."""
    n0, _ = relaxed
    v0 = jnp.zeros_like(n0)

    def H(n, v):
        return float(MODEL.energy(n, GRID)) + float(kinetic_energy(v, GRID))

    h0 = H(n0, v0)
    # dt=0.005: drift is retraction-dominated on a partially-relaxed smoke
    # state and scales ~linearly with dt (measured -1.4e-3 over 320 steps,
    # halving with dt); 160 steps gives ~3x margin under the 2e-3 gate.
    n1, v1, _ = verlet_evolve(MODEL, GRID, n0, v0, dt=0.005, steps=160)
    h1 = H(n1, v1)
    assert abs((h1 - h0) / h0) < 2e-3, f"energy drift {(h1-h0)/h0:+.2e}"
    q1 = float(hopf_charge(n1, GRID))
    assert abs(q1 - 1.0) < 0.05, f"charge not held in dynamics: {q1}"


def test_gate_checkpoint_restart_determinism(relaxed, tmp_path):
    """A run restarted from a mid-run checkpoint reproduces the uninterrupted
    trajectory bit-identically at fixed dtype and device count."""
    import numpy as np

    from jax_solitons.runs import RunConfig, load_checkpoint, save_checkpoint
    from jax_solitons.steppers.verlet import make_verlet_step

    n0, _ = relaxed
    v0 = jnp.zeros_like(n0)
    step = make_verlet_step(MODEL, GRID, dt=0.005)

    # uninterrupted: 40 steps, checkpoint written at 20
    cfg = RunConfig(model="faddeev", N=GRID.N, L=GRID.L, dt=0.005, steps=40)
    n, v = n0, v0
    for i in range(40):
        if i == 20:
            save_checkpoint(tmp_path / "ckpt.npz", {"n": n, "v": v}, cfg, i)
        n, v = step(n, v)
    ref_n, ref_v = np.asarray(n), np.asarray(v)

    # restarted: load at 20, run the remaining 20
    state, cfg2, step_no = load_checkpoint(tmp_path / "ckpt.npz")
    assert cfg2 == cfg and step_no == 20
    n, v = state["n"], state["v"]
    for _ in range(20):
        n, v = step(n, v)
    assert np.array_equal(np.asarray(n), ref_n), "restart not bit-identical (n)"
    assert np.array_equal(np.asarray(v), ref_v), "restart not bit-identical (v)"

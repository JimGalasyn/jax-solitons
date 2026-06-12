"""GPU validation tier: physics-level gates too slow for CPU CI.

Run explicitly on a GPU box:

    SOLITON_GPU_TIER=1 pytest tests/test_gpu_tier.py -v -s

Reference values (source engine, N=108, L=18, c4=4, x64, deep relax):
minimal E/E1 = {1, 1.604, 2.252, 2.817} for Q=1..4 versus published
{1, 1.623, 2.183, 2.662}; VK exponent fit p=0.736 vs 3/4.
"""

import os

import jax
import jax.numpy as jnp
import pytest

# The GPU tier is the CERTIFY step of the hunt-fp32 / certify-x64 protocol.
jax.config.update("jax_enable_x64", True)

from jax_solitons import BoxGrid
from jax_solitons.models import faddeev_cp1_model, faddeev_model
from jax_solitons.models.faddeev import E2Term, E4AreaFormTerm, n_from_state
from jax_solitons.seeds import rational_map_hopfion_cp1
from jax_solitons.steppers.adam import adam_flow
from jax_solitons.topology import hopf_charge

pytestmark = pytest.mark.skipif(
    os.environ.get("SOLITON_GPU_TIER") != "1",
    reason="GPU tier: set SOLITON_GPU_TIER=1 on a GPU box",
)

GRID = BoxGrid(N=96, L=18.0, dtype=jnp.float64)
C4 = 4.0
MODEL = faddeev_model(c4=C4)
MODEL_CP1 = faddeev_cp1_model(c4=C4)


def _relax(n_winding, m_winding, adam_steps=40000):
    # Deep relaxation runs in the CP^1 SPINOR frame (the source engine's
    # frame): projected Adam on the n-field freezes the soft Derrick
    # scaling mode (E2/E4 plateaus ~0.68 with E creeping up at any constant
    # lr), while the spinor frame reaches the virial plateau in ~2k steps.
    # 40k still matters: a second descent phase runs ~12k-22k (the source
    # lesson "15k = scouting, 40k = converged" holds in this engine too).
    z = rational_map_hopfion_cp1(GRID, R=3.5, n=n_winding, m=m_winding)
    z, _ = adam_flow(MODEL_CP1, z, GRID, lr=2e-3, steps=adam_steps)
    state = n_from_state(z)
    E = float(MODEL.energy(state, GRID))
    Q = float(hopf_charge(state, GRID))
    e2 = float(E2Term()(state, GRID))
    e4 = float(E4AreaFormTerm(c4=C4)(state, GRID))
    return E, Q, e2 / e4


def test_gate_vk_q1_q2_ratio():
    """E(Q=2)/E(Q=1) in [1.55, 1.70] (published 1.623; source engine 1.604),
    with charges honest and virial ratios at the Derrick point (gate 5)."""
    E1, Q1, vir1 = _relax(1, 1)
    print(f"\nQ=1: E={E1:.1f}  Q_H={Q1:+.4f}  E2/E4={vir1:.3f}")
    assert abs(Q1 - 1.0) < 0.05
    E2_, Q2, vir2 = _relax(2, 1)   # azimuthal winding is the cheap direction
    print(f"Q=2: E={E2_:.1f}  Q_H={Q2:+.4f}  E2/E4={vir2:.3f}")
    assert abs(Q2 - 2.0) < 0.10
    ratio = E2_ / E1
    print(f"VK ratio E(2)/E(1) = {ratio:.4f}  (band [1.55, 1.70])")
    assert 1.55 < ratio < 1.70, f"VK ratio out of band: {ratio:.4f}"
    # Derrick depth gate: the spinor-frame relaxation reaches the virial
    # point (source-engine lattice-normal base ~0.91; measured here at 40k:
    # Q=1 E=1108.05 vir=0.929, Q=2 E=1782.69 vir=1.007, ratio 1.609). The
    # n-frame plateau at 0.655/0.572 that forced the old [0.5, 1.5] sanity
    # band was a coordinate artifact, now resolved.
    assert 0.8 < vir1 < 1.2 and 0.8 < vir2 < 1.2, \
        f"virial off the Derrick point: {vir1:.3f}, {vir2:.3f}"

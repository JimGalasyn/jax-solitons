"""Acceptance gates for the engine port (the regression contract).

These encode the validation chain of the source research codebase
(LIBRARY_MIGRATION amendment 4). Each gate is skipped until the module it
exercises lands; a gate that lands must never regress.

Gate values come from validated runs of the private research engine:
  - unit hopfion: area-form relaxation holds Q_H = 0.998 at the E ~ 1152 min
  - VK ratio: E(Q=2)/E(Q=1) = 1.604 (published 1.623), exponent ~ Q^{3/4}
  - trefoil Q=7 smoke: carrier determinant 3 held through relaxation
  - persistence: real-time leapfrog conserves dH to ~0.00% with det held
"""

import pytest

pytestmark = pytest.mark.acceptance


@pytest.mark.skip(reason="awaiting port: models.faddeev + steppers.anf + topology.areaform")
def test_gate_unit_hopfion_charge_held():
    """Relax a Q=1 rational-map seed; area-form Q_H must stay within 2% of 1."""


@pytest.mark.skip(reason="awaiting port: models.faddeev + steppers.anf")
def test_gate_vk_q1_q2_ratio():
    """E(Q=2)/E(Q=1) in [1.55, 1.70] (published 1.623; engine 1.604)."""


@pytest.mark.skip(reason="awaiting port: seeds + measure.tracer (knot identification)")
def test_gate_trefoil_q7_determinant_held():
    """Smoke-sized Q_H=7 trefoil: core-curve determinant 3 held through relax."""


@pytest.mark.skip(reason="awaiting port: steppers.verlet (real-time dynamics)")
def test_gate_persistence_energy_conservation():
    """Short real-time run: |dH/H| < 1e-3 and topology checks pass at 2 checkpoints."""


@pytest.mark.skip(reason="awaiting port: runs.checkpoint (orbax)")
def test_gate_checkpoint_restart_determinism():
    """A run restarted from a mid-run checkpoint must reproduce the uninterrupted
    trajectory bit-identically at fixed dtype and device count."""

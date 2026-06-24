"""Pytest session config.

Enable JAX float64. The Tier-1 *exactness* suite (``test_exact.py``) asserts
identities that hold EXACTLY up to float rounding (area-form quantization,
lattice symmetries, O(3) target invariance). In the float32 "scouting" default
those identities are only good to ~1e-3 — e.g. a *generic* O(3) rotation of the
energy rounds to ~4e-4, which silently exceeds the suite's tight tolerances and
makes a pass/fail depend on the platform's XLA rounding rather than on
correctness. Enabling x64 here lets those tests run on a float64 grid where the
identities are genuinely exact (deviation ~1e-12), deterministically on every
machine. It only *enables* the capability — grids still default to float32, so
every other (float32) test is unaffected.
"""
import jax

jax.config.update("jax_enable_x64", True)

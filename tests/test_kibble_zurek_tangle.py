"""The one seed in the library that places nothing.

Every other seed puts a structure in by hand; this one is a random field whose
phase winds by accident. So the tests that matter are not "does it run" but "does
it actually produce a resolved vortex tangle, and does kcut control it" -- a seed
that returned smooth noise with no zeros would pass a shape check and be useless.
"""
import numpy as np
import pytest

import jax
jax.config.update("jax_enable_x64", True)

from jax_solitons.grid import BoxGrid                          # noqa: E402
from jax_solitons.seeds import kibble_zurek_tangle             # noqa: E402
from jax_solitons.vortex_topology import vortex_skeleton       # noqa: E402

G = BoxGrid(N=48, L=24.0)


def _n_core_segments(psi):
    P, _, _ = vortex_skeleton(np.asarray(psi))
    return len(P)


def test_deterministic_in_seed():
    """A stochastic seed has to promise this or nothing built on it reproduces."""
    a = np.asarray(kibble_zurek_tangle(G, kcut=1.4, seed=1))
    b = np.asarray(kibble_zurek_tangle(G, kcut=1.4, seed=1))
    assert np.array_equal(a, b)


def test_seed_actually_changes_the_field():
    a = np.asarray(kibble_zurek_tangle(G, kcut=1.4, seed=1))
    b = np.asarray(kibble_zurek_tangle(G, kcut=1.4, seed=2))
    assert not np.array_equal(a, b)


def test_normalised_to_the_vacuum_manifold():
    psi = np.asarray(kibble_zurek_tangle(G, kcut=1.4, seed=0))
    assert np.mean(np.abs(psi)) == pytest.approx(1.0, rel=1e-7)  # FFT accumulation


def test_it_actually_makes_vortices():
    """THE test. A band-limited random field whose phase did not wind would be a
    smooth blob -- it would pass every other check here and be worthless."""
    psi = kibble_zurek_tangle(G, kcut=1.4, seed=0)
    assert _n_core_segments(psi) > 0


def test_kcut_controls_the_tangle_density():
    """Correlation length ~ 1/kcut, so a larger kcut must give MORE core.
    Averaged over seeds: a single realisation fluctuates, and a test that
    happened to pass on one draw would not be testing the scaling."""
    lo = np.mean([_n_core_segments(kibble_zurek_tangle(G, kcut=0.8, seed=s))
                  for s in range(4)])
    hi = np.mean([_n_core_segments(kibble_zurek_tangle(G, kcut=2.0, seed=s))
                  for s in range(4)])
    assert hi > 2 * lo, f"kcut scaling not seen: {lo:.0f} -> {hi:.0f} segments"


def test_cores_are_resolved_not_single_cell_noise():
    """The Gaussian envelope exists to keep the phase smooth below 1/kcut. Without
    it the 'vortices' are one-cell phase noise and no detector means anything.
    Neighbouring phase differences should mostly be small compared to pi."""
    psi = np.asarray(kibble_zurek_tangle(G, kcut=1.4, seed=0))
    th = np.angle(psi)
    d = np.abs((np.diff(th, axis=0) + np.pi) % (2 * np.pi) - np.pi)
    assert np.median(d) < 0.5, f"phase is not smooth: median jump {np.median(d):.2f}"


def test_nonpositive_kcut_raises():
    with pytest.raises(ValueError, match="kcut"):
        kibble_zurek_tangle(G, kcut=0.0)

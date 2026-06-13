"""Cross-engine equivalence: this engine vs. the pure-numpy oracle.

The Faddeev-Skyrme area-form energy and Whitehead Hopf charge are implemented
*independently* here (JAX) and in the ``nwt-substrate`` reference package
(numpy). They are meant to agree bit-closely -- that is the "validated
bit-identically against the source research engine" claim. This test makes that
claim **live**: it builds one identical hopfion field and asserts both engines'
energy and Hopf charge agree, so the two implementations cannot silently drift.

The oracle (``nwt-substrate``) is an optional, git-installed dependency (see the
``test`` extra in pyproject); when it is absent the whole module is skipped.
"""
from __future__ import annotations

import numpy as np
import pytest

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

# Oracle (pure-numpy reference); skip the whole module if not installed.
oracle = pytest.importorskip("nwt_substrate.solitons.faddeev")
from nwt_substrate.solitons import BoxGrid as OracleGrid

from jax_solitons.grid import BoxGrid
from jax_solitons.models.faddeev import faddeev_model
from jax_solitons.topology import hopf_charge

# A small, certifiable (fp64) Q_H = 1 hopfion seed. N is kept modest so the
# test is a CPU-CI citizen; the agreement is resolution-independent.
N, L, C4, R = 24, 16.0, 4.0, 3.5


def _oracle_seed_field():
    """Build the rational-map hopfion in the oracle and return (oracle_grid, n)
    with n the (3, N, N, N) unit field as a numpy float64 array."""
    g = OracleGrid(N=N, L=L)
    Z1, Z2 = oracle.rational_hopfion(g, R=R, n=1, m=1)
    n1, n2, n3 = oracle.n_from_Z(Z1, Z2)
    return g, np.stack([n1, n2, n3])


def test_faddeev_energy_matches_oracle():
    """E2 + c4*E4 on an identical field must agree across engines."""
    g, n = _oracle_seed_field()
    e_oracle = oracle.faddeev_energy(n[0], n[1], n[2], g.dx, C4)

    grid = BoxGrid(N=N, L=L, dtype=jnp.float64)
    e_engine = float(faddeev_model(c4=C4).energy(jnp.asarray(n), grid))

    assert e_engine == pytest.approx(e_oracle, rel=1e-10)


def test_hopf_charge_matches_oracle():
    """The Whitehead Hopf charge (area form + spectral A) must agree, sign
    included -- both use the +1j (curl A = +B) textbook convention."""
    g, n = _oracle_seed_field()
    q_oracle = oracle.whitehead_hopf_charge(n[0], n[1], n[2], g)

    grid = BoxGrid(N=N, L=L, dtype=jnp.float64)
    q_engine = float(hopf_charge(jnp.asarray(n), grid))

    assert q_engine == pytest.approx(q_oracle, abs=1e-9)
    assert round(q_engine) == 1  # sanity: it is a Q_H = +1 seed

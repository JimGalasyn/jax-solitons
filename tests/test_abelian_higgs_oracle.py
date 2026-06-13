"""Cross-engine gate for the gauge sector: jax-solitons and nwt-substrate must
agree on the BPS line-tension normalization mu_BPS = 2 pi v^2 |n|.

The Faddeev oracle gate compares energy + Hopf charge on an *identical field*
(bit-tight). The abelian-Higgs oracle (nwt_substrate.condensate) is a 1D radial /
analytic BPS module with no 3D-lattice energy, so the clean cross-engine
invariant is the topological Bogomolny line tension -- which is
convention-independent and therefore immune to the self-dual-coupling
discrepancy flagged in models/abelian_higgs.py. (A field-level lattice-energy
comparison is a deeper follow-up; see TODO.)

Self-skips when nwt-substrate is absent (it is the optional `oracle` extra).
"""

import math

import pytest

nwtc = pytest.importorskip("nwt_substrate.condensate")

from jax_solitons.models.abelian_higgs import bps_line_tension  # noqa: E402

HBARC_EV_FM = 197.3269804e6        # hbar c, eV*fm (CODATA)


def test_bps_line_tension_coefficient_matches_oracle():
    """Both engines: mu_BPS / v^2 == 2 pi (the Bogomolny normalization).

    nwt-substrate returns mu in eV/fm with the hbar*c factor restored; dividing
    it back out must recover 2 pi, independent of the gauge coupling e.
    """
    for v_eV in (0.3e6, 0.510998928e6, 2.0e6):
        for e in (1.0, 1.4):
            p = nwtc.AbelianHiggsParams(v_phi_eV=v_eV, e_gauge=e)   # BPS default
            coeff = nwtc.line_tension_BPS(p) * HBARC_EV_FM / v_eV**2
            assert abs(coeff - 2 * math.pi) < 1e-4                  # oracle == 2pi
    # jax-solitons agrees on the same coefficient (exactly, by construction)
    assert abs(bps_line_tension(1.0, 1) - 2 * math.pi) < 1e-12


def test_bps_line_tension_v_squared_scaling_agrees():
    """Both engines scale the line tension as v^2 |n| -- a units-free check."""
    p1 = nwtc.AbelianHiggsParams(v_phi_eV=1.0e6, e_gauge=1.0)
    p2 = nwtc.AbelianHiggsParams(v_phi_eV=2.0e6, e_gauge=1.0)
    oracle_ratio = nwtc.line_tension_BPS(p2) / nwtc.line_tension_BPS(p1)
    js_ratio = bps_line_tension(2.0, 1) / bps_line_tension(1.0, 1)
    assert abs(oracle_ratio - 4.0) < 1e-6 and abs(js_ratio - 4.0) < 1e-12
    # winding scales linearly in jax-solitons
    assert abs(bps_line_tension(1.5, 3) - 2 * math.pi * 1.5**2 * 3) < 1e-9

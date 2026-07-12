"""Tests for jax_solitons.topology.colored_jones (Rosso-Jones cabled states)."""

import cmath
import math

import pytest

from jax_solitons.topology.colored_jones import (
    quantum_integer,
    torus_channels,
    cabling_space_dimension,
    cabled_state,
    colored_jones,
    colored_jones_qtrace,
)

LEVEL = 5
ORD = LEVEL + 2  # 7


def _jones_torus_at_zeta7(m, p):
    """Ordinary (N=2) Jones polynomial of T(m,p) evaluated at t = exp(2 pi i/7).

    V(t) = t^{(m-1)(p-1)/2} (1 - t^{m+1} - t^{p+1} + t^{m+p}) / (1 - t^2).
    Reference for the colored-Jones ratio gate.
    """
    t = cmath.exp(2j * math.pi / ORD)
    pref = t ** ((m - 1) * (p - 1) / 2)
    num = 1 - t ** (m + 1) - t ** (p + 1) + t ** (m + p)
    return pref * num / (1 - t ** 2)


def test_quantum_integer_level5():
    # [2] = 2 cos(pi/7) at level 5
    assert quantum_integer(2).real == pytest.approx(2 * math.cos(math.pi / 7), abs=1e-12)
    # level-5 truncation wraps: [5] = [2], [6] = [1], [7] = 0
    assert quantum_integer(5).real == pytest.approx(quantum_integer(2).real, abs=1e-12)
    assert abs(quantum_integer(7)) == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("strands,power", [(2, 3), (2, 5), (3, 5), (3, 7), (3, 4)])
def test_state_sum_equals_qtrace(strands, power):
    """colored_jones (state-sum sum_J d_J c_J) == direct quantum trace."""
    spin = 0.5 if strands != 3 or power != 4 else 1.0  # one spin-1 case (T(3,4))
    a = colored_jones(spin, strands, power)
    b = colored_jones_qtrace(spin, strands, power)
    assert abs(a - b) < 1e-7


@pytest.mark.parametrize("strands,power", [(2, 3), (2, 5), (3, 5), (3, 7)])
def test_jones_torus_ratio_gate(strands, power):
    """At colour spin 1/2, |colored_jones| ratios reproduce |jones_torus| at zeta_7.

    The decisive gate: includes T(3,5) (spin-1/2 multiplicity 2), the cabling that
    breaks any construction missing the cyclic-multiplicity structure.
    """
    # reference knot for jones_torus (T(7,3) presented as 3 strands, power 7)
    ref_m, ref_p = (7, 3) if (strands, power) == (3, 7) else (strands, power)
    eng = abs(colored_jones(0.5, strands, power))
    eng0 = abs(colored_jones(0.5, 2, 3))                  # normalise to T(2,3)
    jt = abs(_jones_torus_at_zeta7(ref_m, ref_p))
    jt0 = abs(_jones_torus_at_zeta7(2, 3))
    assert eng / eng0 == pytest.approx(jt / jt0, abs=1e-4)


def test_overall_normalization_is_qdim2():
    """Engine = unreduced colored Jones = [N] x reduced; |engine|/|jones_torus| = [2]."""
    eng = abs(colored_jones(0.5, 2, 3))
    jt = abs(_jones_torus_at_zeta7(2, 3))
    assert eng / jt == pytest.approx(quantum_integer(2).real, abs=1e-3)


def test_T35_multiplicity_case():
    """T(3,5): spin-1/2 appears with multiplicity 2 -> channel set {1/2, 3/2}."""
    ch = torus_channels(0.5, 3)
    assert ch.get(0.5) == 2 and ch.get(1.5) == 1
    # and it lands at the trefoil value at zeta_7 (the famous coincidence)
    assert abs(colored_jones(0.5, 3, 5)) == pytest.approx(abs(colored_jones(0.5, 2, 3)),
                                                          abs=1e-4)


@pytest.mark.parametrize("n_q,q,expected", [
    (2, 3, 8), (2, 5, 32), (2, 7, 128), (3, 4, 81), (5, 3, 125),
])
def test_cabling_space_dimension_equals_nqq(n_q, q, expected):
    """n_q^q = dim(V_{n_q}^{tensor q}) = sum_J (2J+1) mult_J  (the Paper 7 n_q^q identity)."""
    assert cabling_space_dimension(n_q, q) == expected
    spin = (n_q - 1) / 2
    dim = sum((round(2 * J) + 1) * mult for J, mult in torus_channels(spin, q).items())
    assert dim == expected

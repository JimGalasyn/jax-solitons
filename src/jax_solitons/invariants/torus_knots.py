"""
Torus knot T(p, q) topology: classification and invariants.

The (p, q) torus knot wraps p times around the major axis and q times
around the minor axis of a torus.  Properties:

  - Genus:      g = (p-1)(q-1)/2  for gcd(p, q) = 1 (knot)
                g = 0             for gcd(p, q) > 1 (link, multi-component)
  - Self-linking:  Lk = pq
  - Crossing number:  c = min(p(q-1), q(p-1))  for p, q >= 2

"""

from __future__ import annotations

from math import gcd


def is_torus_knot(p: int, q: int) -> bool:
    """T(p, q) is a knot (single component) iff gcd(p, q) = 1."""
    return gcd(p, q) == 1


def seifert_genus(p: int, q: int) -> int:
    """
    Seifert genus of T(p, q).

    For gcd(p, q) = 1: g = (p-1)(q-1)/2.
    For multi-component links (gcd > 1): genus is not simply defined; return 0.
    """
    if gcd(p, q) > 1:
        return 0
    return (p - 1) * (q - 1) // 2


def crossing_number(p: int, q: int) -> int:
    """
    Standard crossing number of T(p, q).

    For p = 1 or q = 1 (unknot when gcd = 1): 0.
    For p, q >= 2: min(p(q-1), q(p-1)).
    """
    if p <= 1 or q <= 1:
        return 0
    return min(p * (q - 1), q * (p - 1))


def hopf_charge(p: int, m: int) -> int:
    """Hopf charge Q_H = p * m of the (p, m) torus winding."""
    return p * m


def knot_family(p: int, q: int) -> str:
    """
    Classify (p, q) into a named torus-knot family.

    Returns one of: "unknot", "Hopf", "trefoil", "cinquefoil",
    "heptafoil", "septafoil", or generic "T(p,q)".
    """
    if p == 1 or q == 1:
        return "unknot"
    if (p, q) in {(2, 2), (1, 2), (2, 1)}:
        return "Hopf"
    if {p, q} == {2, 3}:
        return "trefoil"
    if {p, q} == {2, 5}:
        return "cinquefoil"
    if {p, q} == {2, 7}:
        return "heptafoil"
    if {p, q} == {2, 9}:
        return "septafoil"
    return f"T({p},{q})"



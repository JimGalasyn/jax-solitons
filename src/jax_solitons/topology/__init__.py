"""Knot/link invariants: colored Jones (quantum integers at CS level k),
Gauss linking numbers, torus-knot combinatorics, and closed-curve generators.

Extracted 2026-07-12 from the retired nwt-substrate package; pure computational
knot theory, no dependence on that program's physics. Complements
jax_solitons.knots (curve tracing/identification in relaxed fields): seed a
knot, relax it in a model, measure its invariants — one library.
"""
from jax_solitons.topology.colored_jones import *      # noqa: F401,F403
from jax_solitons.topology.linking_invariants import * # noqa: F401,F403
from jax_solitons.topology.torus_knots import *        # noqa: F401,F403
from jax_solitons.topology.curves import (             # noqa: F401
    torus_knot_curve, hopf_link_curves, torus_xyz,
)

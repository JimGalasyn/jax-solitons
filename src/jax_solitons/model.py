"""The Model abstraction (design requirement R1).

A model is:
  - a field state (a PyTree, with an optional leading batch axis — R2),
  - an energy that is a SUM of composable local terms,
  - a manifold constraint (projection + retraction),
  - topological diagnostics (charges).

Forces are obtained by jax.grad of the total energy composed with the
constraint's tangent projection (the egrad->rgrad pattern); steppers never
see individual terms.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Callable, Protocol, runtime_checkable

from jax_solitons.grid import BoxGrid

# A field state is any PyTree; concrete models document their layout.
State = Any


@runtime_checkable
class EnergyTerm(Protocol):
    """One local term of an energy functional, e.g. E2, c4*E4, |B|^2, g|psi|^4."""

    name: str

    def __call__(self, state: State, grid: BoxGrid) -> Any:
        """Return the scalar energy of this term."""
        ...


@runtime_checkable
class Constraint(Protocol):
    """A manifold constraint, e.g. |n|=1 (S^2), |Z|=1 (CP^1), Coulomb gauge."""

    def project_tangent(self, state: State, grad: State) -> State:
        """Project a Euclidean gradient onto the tangent space at `state`."""
        ...

    def retract(self, state: State) -> State:
        """Pull a drifted state back onto the manifold (e.g. renormalize)."""
        ...


@dataclasses.dataclass(frozen=True)
class Model:
    """A field theory = energy terms + constraint + charges.

    Hybrid theories (gauged Faddeev, GPE+Skyrme, ...) are configurations of
    this dataclass, not new code.
    """

    name: str
    terms: tuple[EnergyTerm, ...]
    constraint: Constraint | None = None
    charges: tuple[Callable[[State, BoxGrid], Any], ...] = ()

    def energy(self, state: State, grid: BoxGrid):
        total = 0.0
        for term in self.terms:
            total = total + term(state, grid)
        return total

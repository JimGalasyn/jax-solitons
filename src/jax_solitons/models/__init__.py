"""Concrete field theories, expressed as Model configurations."""

from jax_solitons.models.faddeev import (
    E2Term,
    E4AreaFormTerm,
    S2Constraint,
    faddeev_model,
    n_from_Z,
)
from jax_solitons.models.gpe import GPEKineticTerm, GPEPotentialTerm, gpe_model

__all__ = [
    "E2Term",
    "E4AreaFormTerm",
    "GPEKineticTerm",
    "GPEPotentialTerm",
    "S2Constraint",
    "faddeev_model",
    "gpe_model",
    "n_from_Z",
]

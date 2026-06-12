"""Concrete field theories, expressed as Model configurations."""

from jax_solitons.models.faddeev import (
    E2Term,
    E4AreaFormTerm,
    S2Constraint,
    faddeev_model,
    n_from_Z,
)

__all__ = [
    "E2Term",
    "E4AreaFormTerm",
    "S2Constraint",
    "faddeev_model",
    "n_from_Z",
]

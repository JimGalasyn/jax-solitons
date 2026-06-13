"""Concrete field theories, expressed as Model configurations."""

from jax_solitons.models.abelian_higgs import (
    CovariantKineticTerm,
    HiggsPotentialTerm,
    MagneticTerm,
    abelian_higgs_model,
    magnetic_flux,
    vortex_seed,
)
from jax_solitons.models.faddeev import (
    CP1Constraint,
    CP1Term,
    E2Term,
    E4AreaFormTerm,
    S2Constraint,
    faddeev_cp1_model,
    faddeev_energy_density,
    faddeev_model,
    hopf_charge_cp1,
    n_from_state,
    n_from_Z,
)
from jax_solitons.models.gpe import GPEKineticTerm, GPEPotentialTerm, gpe_model

__all__ = [
    "CP1Constraint",
    "CP1Term",
    "CovariantKineticTerm",
    "E2Term",
    "E4AreaFormTerm",
    "GPEKineticTerm",
    "GPEPotentialTerm",
    "HiggsPotentialTerm",
    "MagneticTerm",
    "S2Constraint",
    "abelian_higgs_model",
    "magnetic_flux",
    "vortex_seed",
    "faddeev_cp1_model",
    "faddeev_energy_density",
    "faddeev_model",
    "gpe_model",
    "hopf_charge_cp1",
    "n_from_state",
    "n_from_Z",
]

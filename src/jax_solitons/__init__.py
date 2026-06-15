"""jax-solitons: a general JAX engine for classical field-theory solitons.

Pre-alpha. The API will change without notice until 0.1.
"""

__version__ = "0.0.3"

from jax_solitons.grid import BoxGrid
from jax_solitons.model import EnergyTerm, Model
from jax_solitons.runs import RunConfig

__all__ = ["BoxGrid", "EnergyTerm", "Model", "RunConfig", "__version__"]

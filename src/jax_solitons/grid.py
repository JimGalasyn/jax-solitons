"""Periodic box grid with explicit dtype.

Design rule (R3): precision is a per-grid parameter, never a global flag.
The sharding spec will live here too once the multi-device layer lands.
"""

from __future__ import annotations

import dataclasses

import jax.numpy as jnp
import numpy as np


@dataclasses.dataclass(frozen=True)
class BoxGrid:
    """A periodic cubic box: N points per axis over physical extent L.

    dtype governs every array built from this grid. fp32 is the scouting
    default on consumer GPUs (memory-bandwidth-bound workloads); certify
    results in float64.
    """

    N: int
    L: float
    dtype: jnp.dtype = jnp.float32

    @property
    def dx(self) -> float:
        return self.L / self.N

    def axis(self) -> jnp.ndarray:
        """Cell-centered 1D coordinate axis, periodic over [-L/2, L/2)."""
        return jnp.asarray(
            (np.arange(self.N) + 0.5) * self.dx - self.L / 2.0, dtype=self.dtype
        )

    def coords(self) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Meshgrid of 3D coordinates, indexing='ij'."""
        ax = self.axis()
        return jnp.meshgrid(ax, ax, ax, indexing="ij")

    def kaxis(self) -> jnp.ndarray:
        """Angular wavenumbers for FFTs on this grid."""
        return jnp.asarray(
            2.0 * np.pi * np.fft.fftfreq(self.N, d=self.dx), dtype=self.dtype
        )

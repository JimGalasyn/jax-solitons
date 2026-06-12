"""Initial conditions. First citizen: the smooth rational-map hopfion seed.

Why rational-map and not the analytic stereographic seed: the analytic seed
lifts n to the CP^1 spinor through the south-pole patch, which is 0/0 exactly
on the soliton core ring, and reaches vacuum only like 1/r^2 (a seam leak on
a periodic box). Both push the seed OUTSIDE the soliton basin. The
rational-map construction (Battye-Sutcliffe / Hietarinta-Salo) composes
smooth maps with a compact C^2 profile, so the field is exactly vacuum
beyond the tube radius w.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from jax_solitons.grid import BoxGrid
from jax_solitons.models.faddeev import n_from_Z


def rational_map_spinor(grid: BoxGrid, R=3.5, w=None, n=1, m=1,
                        center=(0.0, 0.0, 0.0)):
    """Smooth rational-map hopfion CP^1 spinor of charge Q_H = n*m.

    Base map (inverse stereographic R^3 -> S^3):
        phi1 = atan2(y, x)                azimuthal about the z-axis
        phi2 = atan2(r^2 - R^2, 2 R z)    meridional around the core ring
    Degree-(n, m) map on the Hopf fibre (smooth because each amplitude
    vanishes where its phase is ill-defined):
        Z1 = cos(lam) e^{i n phi1},  Z2 = sin(lam) e^{i m phi2}
    Compact profile in the minor radius d = sqrt((rho-R)^2 + z^2):
        lam(d) = (pi/2) smootherstep(d/w), exactly vacuum for d >= w.

    Regularity requires w <= R (else the tube reaches the z-axis and phi1
    re-singularizes). R is the core-ring major radius, w the tube radius.
    """
    if w is None:
        w = 0.85 * R
    if w > R:
        raise ValueError(f"need w <= R for axis regularity (w={w}, R={R})")
    X, Y, Z = (np.asarray(c, dtype=np.float64) for c in grid.coords())
    x0, y0, z0 = center
    x, y, z = X - x0, Y - y0, Z - z0
    rho = np.sqrt(x**2 + y**2)
    r2 = x**2 + y**2 + z**2
    phi1 = np.arctan2(y, x)
    phi2 = np.arctan2(r2 - R**2, 2.0 * R * z)   # oriented so Q_H = +n*m
    d = np.sqrt((rho - R) ** 2 + z**2)
    t = np.clip(d / w, 0.0, 1.0)
    s = t**3 * (10.0 - 15.0 * t + 6.0 * t**2)   # smootherstep (C^2)
    lam = 0.5 * np.pi * s
    Z1 = np.cos(lam) * np.exp(1j * n * phi1)
    Z2 = np.sin(lam) * np.exp(1j * m * phi2)
    nrm = np.sqrt(np.abs(Z1) ** 2 + np.abs(Z2) ** 2)
    return Z1 / nrm, Z2 / nrm


def rational_map_hopfion(grid: BoxGrid, R=3.5, w=None, n=1, m=1,
                         center=(0.0, 0.0, 0.0)) -> jnp.ndarray:
    """Rational-map hopfion as a unit n-field, shape (3, N, N, N), in the
    grid dtype."""
    Z1, Z2 = rational_map_spinor(grid, R=R, w=w, n=n, m=m, center=center)
    nf = n_from_Z(jnp.asarray(Z1), jnp.asarray(Z2))
    return nf.astype(grid.dtype)

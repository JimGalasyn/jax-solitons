"""Closed-curve generators for knots and links (torus knots, Hopf link).

Companion to linking_invariants (its docstring examples use these).
"""
from __future__ import annotations

import numpy as np

DEFAULT_R_MAJOR = 1.5
DEFAULT_R_MINOR = 0.55


def torus_xyz(u, v, R: float = DEFAULT_R_MAJOR, r: float = DEFAULT_R_MINOR):
    """
    Parameterise a torus T^2 in R^3.

    Parameters
    ----------
    u, v : float or array-like
        Toroidal and poloidal angles in [0, 2 pi).
    R, r : float
        Major and minor radii.

    Returns
    -------
    (x, y, z) : tuple of arrays
        Cartesian coordinates of the torus surface.
    """
    x = (R + r * np.cos(v)) * np.cos(u)
    y = (R + r * np.cos(v)) * np.sin(u)
    z = r * np.sin(v)
    return x, y, z


def torus_knot_curve(p: int, q: int, *,
                     R: float = DEFAULT_R_MAJOR, r: float = DEFAULT_R_MINOR,
                     n_points: int = 800, closed: bool = False) -> np.ndarray:
    """
    Return the (p, q) torus-knot curve as an array of 3-D points.

    The knot winds ``p`` times toroidally and ``q`` times poloidally on the
    torus of radii (R, r): the curve is ``torus_xyz(p t, q t)`` for
    t in [0, 2 pi).  (p, q) = (2, 3) is the trefoil, (2, 5) the cinquefoil,
    (1, q)/(p, 1) unknots.

    Parameters
    ----------
    p, q : int
        Toroidal and poloidal winding numbers.
    R, r : float
        Major and minor radii of the embedding torus.
    n_points : int
        Number of samples along the curve.
    closed : bool
        If True, repeat the first point at the end so the returned polyline
        is explicitly closed (useful for plotting); if False (default), the
        ``n_points`` samples cover [0, 2 pi) without duplication (useful as a
        point cloud, e.g. for nearest-distance queries).

    Returns
    -------
    np.ndarray
        Array of shape ``(n_points, 3)`` (or ``(n_points + 1, 3)`` if
        ``closed``).
    """
    endpoint = bool(closed)
    t = np.linspace(0.0, 2 * np.pi, n_points, endpoint=False)
    if endpoint:
        t = np.append(t, 0.0)
    x, y, z = torus_xyz(p * t, q * t, R, r)
    return np.stack([x, y, z], axis=1)


def hopf_link_curves(*, R: float = DEFAULT_R_MAJOR, r: float = DEFAULT_R_MINOR,
                     n_points: int = 800) -> tuple[np.ndarray, np.ndarray]:
    """
    Return the two component curves of a Hopf link (linking number 1).

    The Hopf link is two unknots that each pass
    through the other's disc exactly once.  Component A is the core circle of
    radius R in the z = 0 plane; component B is an identical circle rotated
    90 degrees about the x-axis and shifted by R along x, so the two rings
    interlock once.

    Returns
    -------
    (A, B) : tuple of np.ndarray
        Two arrays of shape ``(n_points, 3)``.
    """
    t = np.linspace(0.0, 2 * np.pi, n_points, endpoint=False)
    A = np.stack([R * np.cos(t), R * np.sin(t), np.zeros_like(t)], axis=1)
    B = np.stack([R * np.cos(t) + R, np.zeros_like(t), R * np.sin(t)], axis=1)
    return A, B


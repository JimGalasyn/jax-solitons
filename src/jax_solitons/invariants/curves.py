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


def _rot_y(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def hopf_clasped_trefoils(*, R: float = DEFAULT_R_MAJOR,
                          r: float = DEFAULT_R_MINOR,
                          n_points: int = 480, sep_scale: float = 1.0,
                          ) -> tuple[np.ndarray, np.ndarray]:
    """
    Two (2, 3) trefoils joined by a SINGLE Hopf clasp: lk = -1.

    THE PLACEMENT IS NOT OBVIOUS, and the obvious one is wrong. Interlocking
    the two tori the way ``hopf_link_curves`` interlocks two circles gives
    ``lk = +/-4``, not +/-1: each (2, 3) trefoil winds its torus longitude
    twice, so a deep torus-interlock multiplies the linking (1 x 2 x 2 = 4).
    A single clasp needs a SHALLOW one instead -- rotate the second trefoil 90
    degrees so its hole axis is perpendicular to the first's, then offset its
    centre by ``(1.10 R, 0, 1.09 R)`` so that exactly one strand pair clasps.
    Those two constants were found by sweeping placement and measuring the
    Gauss integral; they are not derived, and moving them is not free.

    ``sep_scale`` scales that offset and is the natural breathing coordinate
    for an interaction scan: it varies separation while holding the link class
    fixed, which isolates energy-versus-separation from any change of topology.

    THE USABLE RANGE IS NARROW, and narrower than the source this was ported
    from claimed. That source recorded "lk stays -1 over s ~ 0.85-2.3+"; swept
    and measured here, lk = -1 holds only over::

        R=1.5, r=0.55 (this module's defaults)   s in [0.85, 1.20]
        R=5.0, r=1.5  (the source's shape)       s in [0.75, 1.15]

    and at s = 1.25 the Gauss integral returns +3.42 -- a non-integer, i.e. the
    two curves are passing through each other there, not smoothly unclasping.
    By s = 1.30 the link is gone. A scan that assumed the wide range would run
    most of its points on an UNLINKED pair while reporting them as linked,
    which is the exact failure the breathing coordinate exists to prevent.

    The range depends on the (R, r) shape, so measure it for yours rather than
    trusting either number above::

        from jax_solitons.invariants.linking_invariants import (
            gauss_linking_number)
        a, b = hopf_clasped_trefoils(sep_scale=s)
        gauss_linking_number(a, b)          # assert ~ -1 at every scan point

    Parameters
    ----------
    R, r : float
        Major and minor radii of each trefoil's embedding torus.
    n_points : int
        Samples per component.
    sep_scale : float
        Multiplier on the centre-to-centre offset. 1.0 is the reference clasp.

    Returns
    -------
    (A, B) : tuple of np.ndarray
        Two ``(n_points, 3)`` trefoil curves, recentred so the pair's combined
        centroid is at the origin.
    """
    A = torus_knot_curve(2, 3, R=R, r=r, n_points=n_points)
    B = (_rot_y(np.pi / 2) @ A.T).T + sep_scale * np.array([1.10 * R, 0.0, 1.09 * R])
    centre = np.vstack([A, B]).mean(axis=0)
    return A - centre, B - centre


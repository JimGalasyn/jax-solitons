"""
Linking invariants of closed space curves: Gauss linking number, pairwise
linking matrix, Borromean reference rings, Milnor (Massey/Borromean)
indeterminacy, and rotational link symmetry.

These are pure-geometry measurements on point-sampled closed curves in R^3 --
the kind of curves `diagrams.torus_knot_curve` and `diagrams.hopf_link_curves`
produce.  They are the numeric primitive for "binding = Hopf linking of
linked-solitons picture: e.g. two loops joined by a single Hopf clasp
(lk = -1), and the alpha particle is four trefoils at a tetrahedron with every
pair clasped -- the complete graph K_4, lk = +1 on all six edges, total +6.

The module is deliberately rendering-free (numpy only) so a separate portrait /
compendium library can build particle and nucleus pictures on top of it.

Quick start
-----------
    from nwt_substrate.diagrams import hopf_link_curves
    from nwt_substrate.topology import gauss_linking_number
    a, b = hopf_link_curves()
    gauss_linking_number(a, b)          # ~ +/-1

References
----------
Gauss (1833), the linking integral.  Milnor, "Isotopy of links" (1957) for the
higher-order invariants and their indeterminacy.
"""
from __future__ import annotations

from math import gcd

import numpy as np


# ---------------------------------------------------------------------------
# Pairwise linking
# ---------------------------------------------------------------------------
def gauss_linking_number(curve_a: np.ndarray, curve_b: np.ndarray) -> float:
    """Gauss linking number of two closed curves.

    Evaluates the double contour integral

        Lk = 1/(4 pi) * oint oint (r_a - r_b) . (dr_a x dr_b) / |r_a - r_b|^3

    by midpoint discretisation.  The result is a topological invariant and, for
    well-separated curves, a near-integer (the signed number of times one curve
    threads the other); its sign encodes the clasp handedness.

    Parameters
    ----------
    curve_a, curve_b : np.ndarray
        Closed curves of shape (n, 3), sampled once around the loop with NO
        repeated endpoint (the closing segment from the last point back to the
        first is supplied internally).  This matches `torus_knot_curve` and
        `hopf_link_curves` with their default `closed=False`.

    Returns
    -------
    float
        The Gauss linking number (real-valued; round for the integer invariant).
    """
    a = np.asarray(curve_a, dtype=float)
    b = np.asarray(curve_b, dtype=float)
    da = np.roll(a, -1, axis=0) - a            # segment vectors (closed)
    db = np.roll(b, -1, axis=0) - b
    ma = a + da / 2                            # segment midpoints
    mb = b + db / 2
    total = 0.0
    for i in range(len(a)):                    # loop over a, vectorise over b
        rv = ma[i] - mb
        cross = np.cross(da[i], db)
        denom = np.linalg.norm(rv, axis=1) ** 3 + 1e-12
        total += np.sum(np.einsum("jk,jk->j", rv, cross) / denom)
    return float(total / (4.0 * np.pi))


def linking_matrix(curves: list[np.ndarray]) -> np.ndarray:
    """Symmetric matrix of pairwise Gauss linking numbers.

    Parameters
    ----------
    curves : list of np.ndarray
        n closed curves, each shape (m, 3) (see `gauss_linking_number`).

    Returns
    -------
    np.ndarray
        (n, n) symmetric array with zero diagonal; entry (i, j) is
        `gauss_linking_number(curves[i], curves[j])`.  The total linking number
        of the link is `L.sum() / 2`.
    """
    n = len(curves)
    L = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            L[i, j] = L[j, i] = gauss_linking_number(curves[i], curves[j])
    return L


def link_deletion_test(curves: list[np.ndarray]) -> list[np.ndarray]:
    """Linking matrix of the remaining curves after deleting each one in turn.

    The diagnostic that separates Borromean from "anti-Borromean" links: for
    Borromean rings every returned sub-matrix is ~0 (delete any ring and the
    rest fall apart), whereas for a pairwise-saturated link such as the alpha's
    K_4 the sub-matrices keep their nonzero entries (delete one and the rest
    stay linked).

    Returns
    -------
    list of np.ndarray
        Element i is the (n-1, n-1) `linking_matrix` of `curves` with curve i
        removed.
    """
    out = []
    for drop in range(len(curves)):
        rest = [c for k, c in enumerate(curves) if k != drop]
        out.append(linking_matrix(rest))
    return out


# ---------------------------------------------------------------------------
# Higher-order linking: the Milnor indeterminacy
# ---------------------------------------------------------------------------
def milnor_indeterminacy(link_mat: np.ndarray) -> int:
    """Modulus in which the triple Milnor invariant mu-bar is defined.

    Milnor's higher linking invariants (the Massey / Borromean numbers) are
    well defined only modulo the gcd of the lower-order linking numbers
    (Milnor 1957).  For a link this returns ``gcd`` of the rounded pairwise
    linking numbers:

      * ``0``  -- every pairwise linking vanishes: the Borromean / Massey
                  regime, where the triple mu-bar is a genuine integer invariant.
      * ``1``  -- the pairwise linkings have gcd 1: every triple (and longer)
                  mu-bar lives in Z/1 = {0} and carries NO integer information --
                  a literal spatial Massey/Borromean invariant is *obstructed*.
      * ``g>1`` -- mu-bar is defined only mod g.

    Parameters
    ----------
    link_mat : np.ndarray
        A pairwise `linking_matrix` (its off-diagonal entries are used, rounded
        to the nearest integer).

    Returns
    -------
    int
        The gcd of the pairwise linking numbers (0 if they all vanish).
    """
    L = np.asarray(link_mat, dtype=float)
    iu = np.triu_indices(L.shape[0], k=1)
    g = 0
    for x in L[iu]:
        g = gcd(g, int(round(x)))
    return g


# ---------------------------------------------------------------------------
# Reference link: Borromean rings
# ---------------------------------------------------------------------------
def borromean_rings(
    *, n_points: int = 400, a: float = 1.7, b: float = 0.7
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The Borromean rings as three eccentric, mutually perpendicular ellipses.

    Every pair is unlinked (Gauss linking 0) yet the three are collectively
    linked -- the canonical higher-order (Massey/Borromean) link, and the
    reference against which a candidate link is tested for Borromean character.

    Parameters
    ----------
    n_points : int
        Samples per ring (no repeated endpoint, matching the other curve
        generators).
    a, b : float
        Semi-major / semi-minor axes of each ellipse; `a != b` (eccentric) is
        what makes the three rings link.

    Returns
    -------
    tuple of np.ndarray
        Three (n_points, 3) curves, one per ring.
    """
    t = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    z = np.zeros_like(t)
    k1 = np.stack([a * np.cos(t), b * np.sin(t), z], axis=1)
    k2 = np.stack([z, a * np.cos(t), b * np.sin(t)], axis=1)
    k3 = np.stack([b * np.sin(t), z, a * np.cos(t)], axis=1)
    return k1, k2, k3


# ---------------------------------------------------------------------------
# Rotational symmetry of a link
# ---------------------------------------------------------------------------
def _rodrigues(axis: np.ndarray, angle: float) -> np.ndarray:
    a = np.asarray(axis, dtype=float)
    a = a / (np.linalg.norm(a) + 1e-12)
    c, s = np.cos(angle), np.sin(angle)
    vx = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + s * vx + (1 - c) * (vx @ vx)


def tetrahedral_rotations() -> list[np.ndarray]:
    """The 12 proper rotations of the regular tetrahedron (the group T = A_4).

    Identity, eight +/-120 deg rotations about the four vertex axes, and three
    180 deg rotations about the edge-midpoint axes (x, y, z).  Useful as the
    candidate symmetry set for a tetrahedrally-arranged link (e.g. the alpha's
    four trefoils).

    Returns
    -------
    list of np.ndarray
        Twelve 3x3 rotation matrices (det +1).
    """
    verts = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], float)
    verts /= np.linalg.norm(verts[0])
    rots = [np.eye(3)]
    for v in verts:
        rots.append(_rodrigues(v, 2 * np.pi / 3))
        rots.append(_rodrigues(v, -2 * np.pi / 3))
    for e in (np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), np.array([0, 0, 1.0])):
        rots.append(_rodrigues(e, np.pi))
    return rots


def _hausdorff(A: np.ndarray, B: np.ndarray) -> float:
    d = np.sqrt(((A[:, None, :] - B[None, :, :]) ** 2).sum(-1))
    return max(d.min(axis=1).max(), d.min(axis=0).max())


def link_symmetry_permutations(
    curves: list[np.ndarray],
    rotations: list[np.ndarray] | None = None,
    *,
    tol: float = 0.25,
    stride: int = 4,
) -> list[tuple[int, ...]]:
    """Which rotations map the link (as a set of curves) to itself.

    For each rotation, every curve is rotated and matched to the nearest
    original curve by symmetric Hausdorff distance; if all curves match within
    `tol` and the matching is a permutation, the rotation is a symmetry and its
    induced component permutation is recorded.

    Parameters
    ----------
    curves : list of np.ndarray
        The link's closed curves.
    rotations : list of np.ndarray, optional
        Candidate rotation matrices; defaults to `tetrahedral_rotations()`.
    tol : float
        Maximum Hausdorff distance for two curves to be considered the same.
    stride : int
        Sub-sample stride applied to each curve before matching (for speed).

    Returns
    -------
    list of tuple
        One permutation tuple per rotation that is a symmetry (perm[i] = j means
        curve i maps onto curve j).
    """
    if rotations is None:
        rotations = tetrahedral_rotations()
    sub = [np.asarray(c, dtype=float)[::stride] for c in curves]
    n = len(sub)
    perms = []
    for R in rotations:
        perm = []
        ok = True
        for ci in sub:
            ri = (R @ ci.T).T
            dists = [_hausdorff(ri, cj) for cj in sub]
            j = int(np.argmin(dists))
            if dists[j] > tol:
                ok = False
                break
            perm.append(j)
        if ok and sorted(perm) == list(range(n)):
            perms.append(tuple(perm))
    return perms


def permutation_parity(perm: tuple[int, ...]) -> int:
    """Sign of a permutation: +1 if even, -1 if odd (via cycle decomposition)."""
    p = list(perm)
    seen = [False] * len(p)
    sign = 1
    for i in range(len(p)):
        if seen[i]:
            continue
        j, length = i, 0
        while not seen[j]:
            seen[j] = True
            j = p[j]
            length += 1
        if length % 2 == 0:
            sign = -sign
    return sign

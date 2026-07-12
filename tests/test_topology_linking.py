"""Tests for jax_solitons.topology.linking_invariants.

Covers the Gauss linking number and pairwise matrix, the Borromean reference
rings and the Borromean-vs-anti-Borromean deletion test, the Milnor
indeterminacy, and rotational link symmetry.  The alpha particle's four-trefoil
tetrahedral cluster (complete K_4 Hopf link, total lk +6, full A_4 symmetry) is
built inline as a fixture.
"""
from __future__ import annotations

import numpy as np
import pytest

from jax_solitons.topology.curves import torus_knot_curve, hopf_link_curves
from jax_solitons.topology import (
    gauss_linking_number,
    linking_matrix,
    link_deletion_test,
    milnor_indeterminacy,
    borromean_rings,
    tetrahedral_rotations,
    link_symmetry_permutations,
    permutation_parity,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------
def _circle(radius=1.0, n=240, center=(0, 0, 0), plane="xy"):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    zero = np.zeros_like(t)
    if plane == "xy":
        pts = np.stack([radius * np.cos(t), radius * np.sin(t), zero], axis=1)
    elif plane == "xz":
        pts = np.stack([radius * np.cos(t), zero, radius * np.sin(t)], axis=1)
    else:
        raise ValueError(plane)
    return pts + np.asarray(center, dtype=float)


def _rot_to_z(axis):
    """Rotation taking +z to unit `axis` (shortest arc)."""
    z = np.array([0, 0, 1.0])
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    v = np.cross(z, a)
    s = np.linalg.norm(v)
    c = z @ a
    if s < 1e-9:
        return np.eye(3) if c > 0 else np.diag([1, -1, -1.0])
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * ((1 - c) / s ** 2)


def alpha_cluster(n=360, d=4.2, R=5.0, r=1.5):
    """Four (2,3) trefoils at the vertices of a regular tetrahedron, generated
    from one ring by the Klein-4 rotations so the six clasps engage coherently.
    Matches the alpha-portrait construction (Paper 20: four trefoils, K_4)."""
    tet = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], float)
    tet /= np.linalg.norm(tet[0])
    group = [np.eye(3), np.diag([1.0, -1, -1]),
             np.diag([-1.0, 1, -1]), np.diag([-1.0, -1, 1])]
    base = torus_knot_curve(2, 3, R=R, r=r, n_points=n)
    ring0 = (_rot_to_z(tet[0]) @ base.T).T + d * tet[0]
    curves = [(g @ ring0.T).T for g in group]
    center = np.vstack(curves).mean(axis=0)
    return [c - center for c in curves]


# ---------------------------------------------------------------------------
# Gauss linking number
# ---------------------------------------------------------------------------
class TestGaussLinkingNumber:
    def test_hopf_link_is_unit(self):
        a, b = hopf_link_curves()
        assert abs(gauss_linking_number(a, b)) == pytest.approx(1.0, abs=0.02)

    def test_unlink_is_zero(self):
        a = _circle(radius=1.0, center=(0, 0, 0))
        b = _circle(radius=1.0, center=(8, 0, 0))      # far apart, unlinked
        assert gauss_linking_number(a, b) == pytest.approx(0.0, abs=1e-3)

    def test_symmetric_in_arguments(self):
        a, b = hopf_link_curves()
        assert gauss_linking_number(a, b) == pytest.approx(
            gauss_linking_number(b, a), abs=1e-9)

    def test_orientation_reversal_flips_sign(self):
        a, b = hopf_link_curves()
        lk = gauss_linking_number(a, b)
        lk_rev = gauss_linking_number(a, b[::-1])
        assert lk_rev == pytest.approx(-lk, abs=1e-6)

    def test_unlinked_concentric_circles(self):
        # two coplanar circles do not link
        a = _circle(radius=1.0, plane="xy")
        b = _circle(radius=3.0, plane="xy")
        assert gauss_linking_number(a, b) == pytest.approx(0.0, abs=1e-3)


# ---------------------------------------------------------------------------
# Linking matrix
# ---------------------------------------------------------------------------
class TestLinkingMatrix:
    def test_shape_symmetry_zero_diagonal(self):
        curves = list(hopf_link_curves())
        L = linking_matrix(curves)
        assert L.shape == (2, 2)
        assert np.allclose(L, L.T)
        assert np.allclose(np.diag(L), 0.0)

    def test_hopf_offdiagonal_unit(self):
        L = linking_matrix(list(hopf_link_curves()))
        assert abs(L[0, 1]) == pytest.approx(1.0, abs=0.02)

    def test_alpha_cluster_is_complete_K4_total_six(self):
        L = linking_matrix(alpha_cluster())
        off = L[np.triu_indices(4, 1)]
        assert np.allclose(off, 1.0, atol=0.05)        # all six edges +1
        assert L.sum() / 2 == pytest.approx(6.0, abs=0.1)


# ---------------------------------------------------------------------------
# Borromean rings + deletion test
# ---------------------------------------------------------------------------
class TestBorromeanRings:
    def test_pairwise_unlinked(self):
        L = linking_matrix(list(borromean_rings()))
        off = L[np.triu_indices(3, 1)]
        assert np.allclose(off, 0.0, atol=1e-2)

    def test_deletion_frees_the_rest(self):
        subs = link_deletion_test(list(borromean_rings()))
        for sub in subs:                               # each remaining pair ~0
            assert np.allclose(sub, 0.0, atol=1e-2)

    def test_milnor_regime_is_well_defined(self):
        L = linking_matrix(list(borromean_rings()))
        assert milnor_indeterminacy(L) == 0            # all pairwise vanish


class TestAntiBorromean:
    def test_alpha_deletion_keeps_links(self):
        # delete any trefoil: the remaining three stay pairwise linked (+1)
        subs = link_deletion_test(alpha_cluster())
        for sub in subs:
            off = sub[np.triu_indices(3, 1)]
            assert np.allclose(off, 1.0, atol=0.05)


# ---------------------------------------------------------------------------
# Milnor indeterminacy
# ---------------------------------------------------------------------------
class TestMilnorIndeterminacy:
    def test_all_unit_links_obstructed(self):
        L = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], float)
        assert milnor_indeterminacy(L) == 1            # mu-bar in Z/1 = {0}

    def test_all_zero_is_well_defined(self):
        L = np.zeros((3, 3))
        assert milnor_indeterminacy(L) == 0

    def test_common_factor(self):
        L = np.array([[0, 2, 2], [2, 0, 4], [2, 4, 0]], float)
        assert milnor_indeterminacy(L) == 2

    def test_alpha_cluster_obstructed(self):
        # the alpha's +6 pairwise links obstruct any spatial Massey invariant
        assert milnor_indeterminacy(linking_matrix(alpha_cluster())) == 1


# ---------------------------------------------------------------------------
# Rotational symmetry
# ---------------------------------------------------------------------------
class TestSymmetry:
    def test_tetrahedral_group_size_and_orthogonality(self):
        rots = tetrahedral_rotations()
        assert len(rots) == 12
        for R in rots:
            assert np.allclose(R @ R.T, np.eye(3), atol=1e-9)
            assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-9)

    def test_alpha_cluster_has_full_A4_symmetry(self):
        perms = link_symmetry_permutations(alpha_cluster())
        assert len(perms) == 12                        # full tetrahedral group
        # every induced component permutation is even (A_4)
        assert all(permutation_parity(p) == 1 for p in perms)

    def test_permutation_parity(self):
        assert permutation_parity((0, 1, 2, 3)) == 1   # identity, even
        assert permutation_parity((1, 0, 2, 3)) == -1  # single swap, odd
        assert permutation_parity((1, 2, 0)) == 1      # 3-cycle, even

    def test_rejects_non_symmetries(self):
        # the Hopf link is not tetrahedrally symmetric: most rotations are
        # rejected (exercises the no-match branch), but the identity survives.
        perms = link_symmetry_permutations(list(hopf_link_curves()))
        assert len(perms) < 12
        assert (0, 1) in perms

    def test_explicit_rotations_argument(self):
        # passing rotations explicitly: identity is always a symmetry
        perms = link_symmetry_permutations(
            alpha_cluster(), rotations=[np.eye(3)])
        assert perms == [(0, 1, 2, 3)]

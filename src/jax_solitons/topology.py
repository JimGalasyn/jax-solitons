"""Topology-preserving lattice invariants: the Berg-Luscher area form.

Since |n| = 1, both d_i n and d_j n are tangent to S^2, so d_i n x d_j n is
parallel to n and F_ij = n . (d_i n x d_j n) is exactly the pullback area
2-form -- the signed spherical area swept per unit coordinate area. The
faithful lattice F_ij is the solid angle of the spherical quadrilateral
spanned by the four corner values of n on the (i,j) plaquette, split into
two triangles (Van Oosterom-Strackee per triangle):

    Omega(a,b,c) = 2 atan2( a.(b x c), 1 + a.b + b.c + c.a )
    F_ij dx^2    = Omega(A,B,C) + Omega(A,C,D),
    A=n(x), B=n(x+e_i), C=n(x+e_i+e_j), D=n(x+e_j).

This is an exact topological density (sums to 2*pi*integer over closed
surfaces), so it carries the unwinding barrier that naive same-index
discretizations of d_i n x d_j n lack at every resolution. The solid-angle
form is the established best-in-class Hopf-index discretization (Phys. Rev.
B 111, 134408 (2025) benchmarks it as the most accurate and fastest-
converging of four methods); the contribution here is exposing it as a
native, differentiable jax.grad-able primitive rather than a post-hoc
diagnostic.

Field convention throughout: n has shape (3, N, N, N) (leading component
axis); a leading batch axis may be added via vmap.
"""

from __future__ import annotations

import jax.numpy as jnp

from jax_solitons.grid import BoxGrid


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross(a, b):
    return jnp.stack([
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ])


def solid_angle(a, b, c):
    """Signed solid angle of the spherical triangle (a, b, c); unit 3-vectors
    stacked as (3, ...)."""
    num = _dot(a, _cross(b, c))
    den = 1.0 + _dot(a, b) + _dot(b, c) + _dot(c, a)
    return 2.0 * jnp.arctan2(num, den)


def area_form_plaquette(n, i, j):
    """Faithful area form on the (i, j) plaquette: per-cell solid angle of the
    n-quadrilateral (= F_ij * dx^2). n is the (3, N, N, N) unit field."""
    A = n
    B = jnp.roll(n, -1, axis=1 + i)   # n(x + e_i); axis 0 is the component
    D = jnp.roll(n, -1, axis=1 + j)   # n(x + e_j)
    C = jnp.roll(B, -1, axis=1 + j)   # n(x + e_i + e_j)
    return solid_angle(A, B, C) + solid_angle(A, C, D)


def hopf_charge(n, grid: BoxGrid):
    """Hopf charge via the area form: B = (F_23, F_31, F_12) as a lattice
    2-form, A from curl A = B solved spectrally, Q_H = (1/16 pi^2) int A.B.

    Differentiable (returns a jnp scalar). Using the geometric F -- rather
    than the naive same-index product -- is what makes Q_H honest on a
    relaxed field.
    """
    dx = grid.dx
    KX, KY, KZ, K2 = grid.k_vectors()
    F23 = area_form_plaquette(n, 1, 2) / dx**2
    F31 = area_form_plaquette(n, 2, 0) / dx**2
    F12 = area_form_plaquette(n, 0, 1) / dx**2
    Bx, By, Bz = F23, F31, F12
    Bxh, Byh, Bzh = jnp.fft.fftn(Bx), jnp.fft.fftn(By), jnp.fft.fftn(Bz)
    inv = jnp.where(K2 > 0, 1.0 / K2, 0.0)
    Axh = 1j * (KY * Bzh - KZ * Byh) * inv   # +1j: curl A = +B (textbook Q_H)
    Ayh = 1j * (KZ * Bxh - KX * Bzh) * inv
    Azh = 1j * (KX * Byh - KY * Bxh) * inv
    Ax = jnp.real(jnp.fft.ifftn(Axh))
    Ay = jnp.real(jnp.fft.ifftn(Ayh))
    Az = jnp.real(jnp.fft.ifftn(Azh))
    H = jnp.sum(Ax * Bx + Ay * By + Az * Bz) * dx**3
    return H / (16.0 * jnp.pi**2)

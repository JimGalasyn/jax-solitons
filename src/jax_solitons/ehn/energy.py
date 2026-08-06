#!/usr/bin/env python3
"""Shared EHN energy core — the primitives common to the relax and quench engines.

Extracted from `ehn_relax.py` (locked-census port, design
`analysis/I1_LOCKED_CENSUS_PORT_DESIGN.md`) so the wrapped-∂a LOCK lives in ONE
place both engines import, rather than being hand-ported (and possibly mis-ported)
into the quench engine. Jim's steer: the relax and quench engines share as much
common code as possible.

The load-bearing shared object is the wrapped-∂a **lock primitive**
`axion_grad(..., agrad="wrapped")` and its consumer `rho_L3` (ρ = B·∂a). The
wrapped phase gradient `angle(φ₂(x+î)·conj(φ₂(x−î)))/2dx` is inherently a
roll-based finite difference (the wrapping only makes sense between neighbouring
lattice sites), so it is ENGINE-AGNOSTIC — identical whether the host engine
otherwise uses finite-difference (relax) or spectral (quench) derivatives.

Faithfulness contract: these functions reproduce `ehn_relax`'s originals
BIT-FOR-BIT (verified by `_selftest_vs_frozen`, which diffs against the tagged
pre-refactor engine `ehn-relax-preshared-refactor`). The only interface change is
that `axion_grad`/`skyrme_e4` take their mode as an explicit ARGUMENT
(`agrad`, and Skyrme is simply called-or-not) instead of module globals — a
sharing requirement, not a physics change.
"""
from __future__ import annotations

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp


# --- discrete ops: real-space central differences (EHN's naive 2nd-order) -----
def _b(f, i):   # backward neighbour f(x−ê_i)
    return jnp.roll(f, 1, axis=i)


def _f(f, i):   # forward neighbour f(x+ê_i)
    return jnp.roll(f, -1, axis=i)


def d_c(f, i, dx):
    """central first derivative ∂_i f."""
    return (_f(f, i) - _b(f, i)) / (2 * dx)


def lap(f, dx):
    """central Laplacian ∂²f."""
    return sum(_f(f, i) - 2 * f + _b(f, i) for i in range(3)) / dx ** 2


def curlA(Ax, Ay, Az, dx):
    """finite-difference curl (∇×A)."""
    return (d_c(Az, 1, dx) - d_c(Ay, 2, dx),
            d_c(Ax, 2, dx) - d_c(Az, 0, dx),
            d_c(Ay, 0, dx) - d_c(Ax, 1, dx))


# --- THE LOCK primitive -------------------------------------------------------
def axion_grad(p2, dx, eps_a, agrad):
    """∂_i a  (a = arg φ₂), three discretizations selected by `agrad`:

    "bilinear" — regularized phase velocity Im(φ₂*∂φ₂)/(|φ₂|²+eps_a). Smooth, but
      modulus-SUPPRESSED: fields drain ∫ρ by modulus rearrangement without any
      unwinding (the "matter sheds ρ while holding Q" escape).
    "wrapped"  — central wrapped phase difference angle(φ₂(x+î)·conj(φ₂(x−î)))/2dx.
      The natural central difference of a COMPACT angle: modulus-BLIND, ∇×∇a is an
      exact integer string delta ⟹ ∫ρ quasi-algebraically locked to N_link (EHN's
      stated protection). THE LOCK.
    "naive"    — per-site a = arctan2(Im φ₂, Re φ₂), then an ordinary central
      difference OF THAT ANGLE. The most literal reading of EHN's "naive spatial
      discretization ... second-order central-difference scheme" applied to a, and
      the third candidate NLINK_LADDER.md names for when bilinear fails to
      reproduce their floor.

      It is expected to be WORSE than bilinear, and the reason is the point of
      running it: arctan2 returns a principal value in (−π, π], so wherever the
      phase crosses the branch cut two adjacent sites differ by ≈2π and the
      difference reports ≈2π/2dx — an O(1/dx) spurious gradient on a whole
      codimension-1 sheet that moves with the field. `wrapped` differs ONLY in
      taking the angle of the ratio rather than the difference of the angles,
      which is exactly what removes the cut. Any ∫ρ this yields is contaminated by
      the sheet, so a bound state under it would be evidence about the artefact,
      not about the physics. That is a result worth having: it is the arm that
      says whether EHN's floor is reproducible from the most literal reading of
      their method.
    """
    if agrad == "wrapped":
        return tuple(jnp.angle(_f(p2, i) * jnp.conj(_b(p2, i))) / (2 * dx)
                     for i in range(3))
    if agrad == "naive":
        # jnp.angle IS arctan2(imag, real) -- same principal branch, same cut.
        a = jnp.angle(p2)
        return tuple(d_c(a, i, dx) for i in range(3))
    if agrad != "bilinear":
        # Was a silent fall-through to bilinear, which is safe with two modes and
        # a trap with three: `agrad` is recorded in the manifest, so a typo would
        # file a bilinear run under whatever name was misspelled and there would be
        # nothing in the artifacts to catch it.
        raise ValueError(f"unknown agrad {agrad!r}; "
                         f"expected 'bilinear', 'wrapped' or 'naive'")
    inv = 1.0 / (jnp.abs(p2) ** 2 + eps_a)
    return tuple(jnp.imag(jnp.conj(p2) * d_c(p2, i, dx)) * inv for i in range(3))


def rho_L3(p2, B, dx, eps_a, agrad):
    """ℒ₃ density ρ = B_i ∂_i a  (a = arg φ₂), with ∂a discretized per `agrad`.
    B is a 3-tuple (Bx,By,Bz) — whatever the host engine supplies (relax: the
    constrained auxiliary field; quench: ∇×A). The lock lives in `axion_grad`."""
    ga = axion_grad(p2, dx, eps_a, agrad)
    return sum(B[i] * ga[i] for i in range(3))


def E_L3_electric(p2, B, s, dx, C, eps_a, agrad):
    """ℒ₃ ELECTRIC energy  ½C Σ ρ·s,  ρ = B·∂a,  s = the A₀/Gauss potential.
    This is exactly `ehn_relax.E_disc`'s `eelec` term. The locked-census port
    (step 2b) derives the quench engine's lock FORCE as −∂(this)/∂φ (autodiff of
    the SAME energy the relax engine descends), so the quench inherits the
    validated lock rather than a re-derived one. The R-C-LC-1 gate showed the
    transverse (A,E) form without this A₀-coupled energy does NOT protect."""
    return 0.5 * C * jnp.sum(rho_L3(p2, B, dx, eps_a, agrad) * s)


def gauss_relax_s(s, p1sq, p2sq, rho, dx, beta, C, q1, q2):
    """One A₀/Gauss relaxation step (EHN Eq.13, `ehn_relax.relax_iter` (13)):
        s ← s + β(Δs − 2(q₁²|φ₁|² + q₂²|φ₂|²)s + Cρ).
    Shared by relax and the quench port — the A₀ field the transverse (A,E) form
    lacked. p1sq=|φ₁|², p2sq=|φ₂|²."""
    return s + beta * (lap(s, dx) - 2.0 * (q1 ** 2 * p1sq + q2 ** 2 * p2sq) * s
                       + C * rho)


# --- Skyrme quartic (single-knot Derrick stabilizer) --------------------------
def skyrme_e4(n4, dx):
    """O(4) Skyrme quartic ℒ₃ on the normalized field n∈S³ built from the 4 real
    scalar components n4=(p1r,p1i,p2r,p2i). E₄ = Σ_{i<j}[(∂ᵢn·∂ᵢn)(∂ⱼn·∂ⱼn) −
    (∂ᵢn·∂ⱼn)²] (standard SU(2)/O(4) Skyrme; ∝1/R Derrick balance)."""
    N = [n4[0], n4[1], n4[2], n4[3]]
    nrm = jnp.sqrt(sum(x ** 2 for x in N) + 1e-12)
    n = [x / nrm for x in N]
    d = [[d_c(n[a], i, dx) for a in range(4)] for i in range(3)]   # d[i][a]=∂ᵢnᵃ
    dot = lambda i, j: sum(d[i][a] * d[j][a] for a in range(4))
    return sum(jnp.sum(dot(i, i) * dot(j, j) - dot(i, j) ** 2)
               for i in range(3) for j in range(i + 1, 3))

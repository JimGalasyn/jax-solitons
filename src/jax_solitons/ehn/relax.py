#!/usr/bin/env python3
"""FAITHFUL Eto-Hamada-Nitta knot-soliton relaxer (arXiv:2407.11731 supplemental).

Implements EHN's EXACT numerical scheme (their Eqs 5,11,12,13), which the spectral
`ehn_knot_batch.py` only approximated (it used one combined dt + a one-shot
const-mass A0). EHN use NAIVE 2nd-order CENTRAL DIFFERENCES (NOT compact U(1)), an
INDEPENDENT auxiliary field B_i softly constrained to ∇×A, and A0 solved by a
SEPARATE Gauss-law iteration with its own step β. The whole system is CO-RELAXED
from a knot IC with the Chern-Simons coupling C ON from the start — which (per the
06-28 supplemental read) is the piece we never ran: the bound knot is a local
minimum WITHIN the linked sector, and co-relaxing from a knot IC stays in that
basin instead of letting B decouple from the φ₁ strings (our build-then-ramp bug).

Energy (EHN Eq.5, v=g=1, q₁=1 q₂=0 → φ₂ global; NO ½ on scalar gradients):
  E = ∫ |D_iφ₁|² + |∂_iφ₂|² + V + ½(∂_iA_j)² + ½CρA₀ + (γ−1)/2(∂_iA_i)²
      + Σ_i [ w_i(B_i−ε∂A) + (U/2)(B_i−ε∂A)² ]
  V = λ(|φ₁|²+|φ₂|²−1)² − κ|φ₁|²|φ₂|² ;  ρ = B_i ∂_i a ;  a = arg φ₂ ;  γ = 1+U.
Relaxation (repeat):
  (12) u ← u − α ∂E/∂u           (fields φ₁,φ₂,A_i,B_i ; α=4e-4 ; via jax.grad)
  (13) s ← s + β[Δs − 2g²|φ₁|²s + Cρ]   (A₀ Gauss-law ; β=2e-3 ; posmass screening)
  (11) w_i ← w_i + U(B_i − ε∂A)  (multiplier ; U=50)
EHN: U=50, γ=51, d=0.8/v, α=4e-4 v⁻², β=2e-3 v⁻², 320³.

STEP-SIZE BOUND — α < 2/H, AND H IS NOT dx-INDEPENDENT BELOW dx≈0.14.
Measured 2026-08-02 by power-iterating the Hessian of E_disc AT THE VACUUM
(λ=1000, U=50, C=400) — which the integrator check below, run from a settled
trefoil, is consistent with, so the core was not separately probed:

      dx      H_max     α_max = 2/H
    1.60     8000.0     2.50e-4      ┐
    0.80     8000.0     2.50e-4      │  potential-dominated plateau:
    0.40     8000.0     2.50e-4      │  H = 8λ exactly, dx-INDEPENDENT
    0.20     8000.0     2.50e-4      ┘
    0.10    15336.4     1.30e-4      <- gradient/gauge terms take over
    0.05    61232.5     3.27e-5      <- H ∝ 1/dx² (ratio 3.99 ≈ 4 over 2x)

Fitting the two sub-plateau rows as a pure c/dx² gives c = 153.4 and 153.1, so

    H(dx) ≈ max(8λ, 153/dx²)        NOT a sum — see below

and the branches cross at sqrt(153.2/8λ) = **dx ≈ 0.14**, not the 0.11 an
analytic 2U/dx² = 8λ predicts. That estimate undershoots because the scalar
gradient contributes alongside the gauge penalty: measured c is 153, not 2U=100.

It has to be max() rather than a sum, and the dx=0.20 row is the proof —
additive would predict 8000 + 153.2/0.04 = 11831, but the table measures exactly
8000.0. The two branches are different eigenmodes, not two terms of one mode.

**If you refine the grid, α MUST come down as 1/dx² once past the plateau.**
The single number most readers want: **the default α=1e-4 is safe down to
dx ≈ 0.088** (where H = 20000). Below that it diverges — at dx=0.05 the limit is
3.3e-5 and 1e-4 is 3x over. Note the plateau ends at 0.14, so dx=0.12 is already
off it (H = 10640, α_max = 1.88e-4) even though nothing breaks at the default.

Recompute rather than trusting this table if λ, U, κ or c4 change: κ and the
--c4 Skyrme quartic (relax.py `skyrme_e4`) are gradient terms, so they enter the
1/dx² branch and move c; λ and U set where the branches cross.

The plateau value 8λ is the JOINT radial mode: V = λ(|φ₁|²+|φ₂|²−1)² with both
scalars at v/√2, so moving them together doubles d(|φ₁|²+|φ₂|²)/dh, quadrupling
V — 2x the curvature per unit norm of the single-field mode (4λ, which measures
4000.0 against analytic 4λ, ratio 1.0000).

Confirmed against the integrator at λ=1000, dx=0.8, 200 steps from a settled
trefoil — the predicted 2/H = 2.50e-4 is where it actually breaks:

    α = 2.00e-4   E=3333.3  Lk=-3.0   stable
    α = 2.05e-4   E=3334.3  Lk=-3.0   stable      (= EHN 4e-4 x dx³)
    α = 2.50e-4   E=3332.1  Lk=-3.0   stable      (marginal: |1-αH| = 1)
    α = 3.00e-4   E=7.2e8             DIVERGING
    α = 4.00e-4   E=nan               NaN

α=2.5e-4 surviving 200 steps is expected: 2/H is where |1-αH| reaches 1, so it
is the marginal case rather than the first unstable one. Anything above it goes.
The practical default should sit below, not on, the bound.

That also means EHN's published α=4e-4 exceeds the bound of this
functional by 1.6x. Their d³-weighted energy (their Eq.1 is
E = ∫d³x ℰ, and the supplemental writes E ≃ d³ Σ ℰ_disc) would give an effective
step of 4e-4 × 0.8³ = 2.05e-4, inside the bound — but so would a plain ½, and
with one dx in the paper nothing here separates 0.512 from 0.5. Recorded as an
open discrepancy, not a resolved one.

  python3 ehn_relax.py            # co-relax a knot, C=400 on from start
"""
import os, sys, time, json
from pathlib import Path
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from functools import partial

from . import knot_batch as EK    # build_ic, skyrmion_number, kvecs, relax_flux_london

PI = np.pi


# Discrete ops + the wrapped-∂a LOCK live in the shared core (ehn_energy),
# imported so the relax and quench engines use ONE copy (locked-census port).
# The extraction reproduces this engine's originals BIT-FOR-BIT vs the tagged
# pre-refactor snapshot (ehn-relax-preshared-refactor); see ehn_energy selftest.
from .energy import (_b, _f, d_c, lap, curlA,               # noqa: E402,F401
                        axion_grad as _shared_axion_grad,
                        rho_L3 as _shared_rho_L3,
                        skyrme_e4 as _shared_skyrme_e4)

AGRAD = "bilinear"   # ∂a discretization; set by run(--agrad) before the first jit trace


def axion_grad(p2, dx, eps_a):
    """∂_i a (a=arg φ₂); mode from module global AGRAD. Impl: ehn_energy (shared)."""
    return _shared_axion_grad(p2, dx, eps_a, AGRAD)


def _rho(u, dx, eps_a):
    return _shared_rho_L3(u[2] + 1j * u[3], (u[7], u[8], u[9]), dx, eps_a, AGRAD)


C4 = 0.0   # Skyrme-quartic (ℒ₃) coeff; set by run(--c4) before the first jit trace.


def skyrme_e4(u, dx):
    """O(4) Skyrme quartic on n∈S³ from (φ₁,φ₂). Impl: ehn_energy (shared)."""
    return _shared_skyrme_e4((u[0], u[1], u[2], u[3]), dx)



def E_disc(u, s, w, dx, lam, kappa, C, U, eps_a, q1, q2):
    """EHN Eq.5 discretised energy DENSITY summed over sites (NO dx³ volume factor —
    EHN's Eq.12 descends ∂E_disc/∂u of the density, so their α=4e-4 is calibrated to
    this; the physical energy E=dx³·ΣE_disc is reported separately). q2≠0 gauges φ₂
    (it becomes a fractional vortex carrying its own U(1) flux — supplemental)."""
    p1r, p1i, p2r, p2i, Ax, Ay, Az, Bx, By, Bz = u
    p1 = p1r + 1j * p1i
    p2 = p2r + 1j * p2i
    A = (Ax, Ay, Az)
    B = (Bx, By, Bz)
    # |D_iφ_{1,2}|² (covariant, charges q1,q2) — NO ½
    eg1 = sum(jnp.sum(jnp.abs(d_c(p1, i, dx) - 1j * q1 * A[i] * p1) ** 2) for i in range(3))
    eg2 = sum(jnp.sum(jnp.abs(d_c(p2, i, dx) - 1j * q2 * A[i] * p2) ** 2) for i in range(3))
    a1 = p1r ** 2 + p1i ** 2
    a2 = p2r ** 2 + p2i ** 2
    V = jnp.sum(lam * (a1 + a2 - 1.0) ** 2 - kappa * a1 * a2)
    # magnetic ½(∂_iA_j)² + gauge fix (γ−1)/2(∂_iA_i)², γ=1+U
    emag = 0.5 * sum(jnp.sum(d_c(A[j], i, dx) ** 2) for i in range(3) for j in range(3))
    divA = sum(d_c(A[i], i, dx) for i in range(3))
    egf = 0.5 * U * jnp.sum(divA ** 2)
    # B-constraint: B_i = (∇×A)_i  (multiplier w + penalty U/2)
    cA = curlA(Ax, Ay, Az, dx)
    econs = sum(jnp.sum(w[i] * (B[i] - cA[i]) + 0.5 * U * (B[i] - cA[i]) ** 2) for i in range(3))
    # electric ½CρA₀, ρ = B_i ∂_i a
    ga = axion_grad(p2, dx, eps_a)
    rho = sum(B[i] * ga[i] for i in range(3))
    eelec = 0.5 * C * jnp.sum(rho * s)
    esky = C4 * skyrme_e4(u, dx) if C4 > 0 else 0.0     # ℒ₃ single-knot stabilizer
    return eg1 + eg2 + V + emag + egf + econs + eelec + esky


_grad_u = jax.jit(jax.grad(E_disc), static_argnums=())


@partial(jax.jit, static_argnums=())
def relax_iter(u, s, w, dx, lam, kappa, C, U, eps_a, alpha, beta, q1, q2):
    # (12) fields
    g = _grad_u(u, s, w, dx, lam, kappa, C, U, eps_a, q1, q2)
    u = tuple(ui - alpha * gi for ui, gi in zip(u, g))
    # (13) A0 Gauss-law (posmass screening 2g²(q1²|φ₁|²+q2²|φ₂|²), g=1)
    a1 = u[0] ** 2 + u[1] ** 2
    a2 = u[2] ** 2 + u[3] ** 2
    rho = _rho(u, dx, eps_a)
    s = s + beta * (lap(s, dx) - 2.0 * (q1 ** 2 * a1 + q2 ** 2 * a2) * s + C * rho)
    # (11) multiplier
    cA = curlA(u[4], u[5], u[6], dx)
    w = tuple(w[i] + U * (u[7 + i] - cA[i]) for i in range(3))
    return u, s, w


def energy_report(u, s, w, dx, lam, kappa, C, U, eps_a, q1=1.0, q2=0.0):
    p1r, p1i, p2r, p2i, Ax, Ay, Az, Bx, By, Bz = u
    p1 = p1r + 1j * p1i; p2 = p2r + 1j * p2i
    A = (Ax, Ay, Az)
    dx3 = dx ** 3
    eg1 = float(sum(jnp.sum(jnp.abs(d_c(p1, i, dx) - 1j * q1 * A[i] * p1) ** 2) for i in range(3)) * dx3)
    eg2 = float(sum(jnp.sum(jnp.abs(d_c(p2, i, dx) - 1j * q2 * A[i] * p2) ** 2) for i in range(3)) * dx3)
    a1 = p1r ** 2 + p1i ** 2; a2 = p2r ** 2 + p2i ** 2
    V = float(jnp.sum(lam * (a1 + a2 - 1.0) ** 2 - kappa * a1 * a2) * dx3)
    cA = curlA(Ax, Ay, Az, dx)
    emag = float(0.5 * sum(jnp.sum(cA[i] ** 2) for i in range(3)) * dx3)
    rho = _rho(u, dx, eps_a)
    eelec = float(0.5 * C * jnp.sum(rho * s) * dx3)
    link = float(jnp.sum(rho) * dx3)
    esky = float(C4 * skyrme_e4(u, dx) * dx3) if C4 > 0 else 0.0
    return {"grad1": eg1, "grad2": eg2, "pot": V, "mag": emag, "elec": eelec, "sky": esky,
            "link": link, "total": eg1 + eg2 + V + emag + eelec + esky}


def build_ic_gpu(N, L, nlink, R, core, n=160):
    """GPU port of EK.build_ic (same linked-knot geometry) — the numpy original is
    O(n·N³) on CPU (~70 min at N=320); here the grid arrays are JAX so each segment
    op runs on the GPU (~1 min at 320³). nlink φ₁ rings threaded by one φ₂ ring."""
    g = np.linspace(-L / 2, L / 2, N, endpoint=False)
    Xn, Yn, Zn = np.meshgrid(g, g, g, indexing="ij")
    X, Y, Z = jnp.asarray(Xn), jnp.asarray(Yn), jnp.asarray(Zn)
    apex = np.array([0.0, 0.0, 0.9 * L])
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    big = np.stack([R * np.cos(t), R * np.sin(t), 0 * t], 1)            # φ₂ big ring
    zhat = np.array([0.0, 0.0, 1.0]); r = 0.5 * R
    smalls = []
    for k in range(nlink):
        thk = 2 * np.pi * k / nlink
        ck = R * np.array([np.cos(thk), np.sin(thk), 0.0])
        rhat = np.array([np.cos(thk), np.sin(thk), 0.0])
        smalls.append(ck[None, :] + r * (np.cos(t)[:, None] * rhat[None, :]
                                         + np.sin(t)[:, None] * zhat[None, :]))

    def dist(curves):
        d = jnp.full(X.shape, jnp.inf)
        for c in curves:
            for p in c:
                d = jnp.minimum(d, jnp.sqrt((X - p[0]) ** 2 + (Y - p[1]) ** 2 + (Z - p[2]) ** 2))
        return d

    def solid_angle(curves):
        th = jnp.zeros_like(X)
        ax, ay, az = apex[0] - X, apex[1] - Y, apex[2] - Z
        na = jnp.sqrt(ax * ax + ay * ay + az * az)
        for c in curves:
            Om = jnp.zeros_like(X)
            M = len(c)
            for i in range(M):
                B, Cc = c[i], c[(i + 1) % M]
                bx, by, bz = B[0] - X, B[1] - Y, B[2] - Z
                cx, cy, cz = Cc[0] - X, Cc[1] - Y, Cc[2] - Z
                nb = jnp.sqrt(bx * bx + by * by + bz * bz); nc = jnp.sqrt(cx * cx + cy * cy + cz * cz)
                crx, cry, crz = by * cz - bz * cy, bz * cx - bx * cz, bx * cy - by * cx
                tri = ax * crx + ay * cry + az * crz
                den = (na * nb * nc + (ax * bx + ay * by + az * bz) * nc
                       + (ax * cx + ay * cy + az * cz) * nb + (bx * cx + by * cy + bz * cz) * na)
                Om = Om + 2 * jnp.arctan2(tri, den)
            th = th + 0.5 * Om
        return th

    prof = lambda d: jnp.tanh(d / core)
    pA = prof(dist(smalls)); pB = prof(dist([big]))
    norm = jnp.sqrt(pA ** 2 + pB ** 2 + 1e-6)
    phi1 = (pA / norm) * jnp.exp(1j * solid_angle(smalls))
    phi2 = (pB / norm) * jnp.exp(1j * solid_angle([big]))
    return phi1.astype(jnp.complex128), phi2.astype(jnp.complex128)


def build_ic_torus(N, L, p, q, R, r, core, n=800, twist=0):
    """φ₁ = T(p,q) torus knot wound on the torus whose CORE CIRCLE is the φ₂ ring
    (radius R, z=0 plane): the φ₁ curve winds p× the long way (around the z-axis)
    and q× around the MERIDIAN (minor radius r) — and it is the meridian winds
    that encircle the φ₂ string, so Lk(φ₁,φ₂) = q. T(2,3) = trefoil (Lk=3, det=3);
    T(3,4) = 8_19 (Lk=4, det=3, inside EHN's observed N_link>=4 stability window);
    T(2,5) = cinquefoil (Lk=5, det=5). EHN Fig.2 panels 2-4 are exactly this class
    ("single φ₁ and φ₂ loops... linking multiple times"). The trefoil-baryon
    discriminator: does the KNOTTED φ₁ string survive locked relaxation (det held,
    one component) — noting EHN did NOT find N_link<4 stable in their box.
    Curve-curve clearance is r everywhere ⟹ keep r ≫ core and r ≫ dx."""
    g = np.linspace(-L / 2, L / 2, N, endpoint=False)
    Xn, Yn, Zn = np.meshgrid(g, g, g, indexing="ij")
    X, Y, Z = jnp.asarray(Xn), jnp.asarray(Yn), jnp.asarray(Zn)
    apex = np.array([0.0, 0.0, 0.9 * L])
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    # sub-grid offset: keep the curves OFF the exact grid planes/corners — an
    # exactly-aligned core (ring in the z=0 grid plane) sits on plaquette
    # boundaries where the winding detector is degenerate ⟹ fragmented skeleton
    # + a biased Gauss Lk (read +25% high before this nudge).
    dxg = L / N
    off = np.array([0.37, 0.29, 0.41]) * dxg
    knot = off + np.stack([(R + r * np.cos(q * t)) * np.cos(p * t),
                           (R + r * np.cos(q * t)) * np.sin(p * t),
                           r * np.sin(q * t)], 1)                    # φ₁ T(p,q)
    tring = np.linspace(0, 2 * np.pi, max(160, n // 4), endpoint=False)
    ring = off + np.stack([R * np.cos(tring), R * np.sin(tring), 0 * tring], 1)  # φ₂ core

    def dist(curves):
        d = jnp.full(X.shape, jnp.inf)
        for c in curves:
            for pt in c:
                d = jnp.minimum(d, jnp.sqrt((X - pt[0]) ** 2 + (Y - pt[1]) ** 2 + (Z - pt[2]) ** 2))
        return d

    def solid_angle(curves):
        th = jnp.zeros_like(X)
        ax_, ay_, az_ = apex[0] - X, apex[1] - Y, apex[2] - Z
        na = jnp.sqrt(ax_ * ax_ + ay_ * ay_ + az_ * az_)
        for c in curves:
            Om = jnp.zeros_like(X)
            M = len(c)
            for i in range(M):
                B, Cc = c[i], c[(i + 1) % M]
                bx, by, bz = B[0] - X, B[1] - Y, B[2] - Z
                cx, cy, cz = Cc[0] - X, Cc[1] - Y, Cc[2] - Z
                nb = jnp.sqrt(bx * bx + by * by + bz * bz); nc = jnp.sqrt(cx * cx + cy * cy + cz * cz)
                crx, cry, crz = by * cz - bz * cy, bz * cx - bx * cz, bx * cy - by * cx
                tri = ax_ * crx + ay_ * cry + az_ * crz
                den = (na * nb * nc + (ax_ * bx + ay_ * by + az_ * bz) * nc
                       + (ax_ * cx + ay_ * cy + az_ * cz) * nb + (bx * cx + by * cy + bz * cz) * na)
                Om = Om + 2 * jnp.arctan2(tri, den)
            th = th + 0.5 * Om
        return th

    prof = lambda d: jnp.tanh(d / core)
    pA = prof(dist([knot])); pB = prof(dist([ring]))
    norm = jnp.sqrt(pA ** 2 + pB ** 2 + 1e-6)
    # FRAMING twist (the neutron-labeled variant): wind arg φ₁ by `twist` extra
    # units of the RING's solid-angle phase — the internal U(1) the string
    # carries (arg of the "other" phase along the tube) advances by 2π·twist·q
    # per knot traversal. Whether the relaxer HOLDS this as a distinct internal
    # label (n/p doublet analog), unwinds it, or drills a compensating zero is
    # the framing-stability experiment; the tracers diagnose which.
    th1 = solid_angle([knot]) + (twist * solid_angle([ring]) if twist else 0.0)
    phi1 = (pA / norm) * jnp.exp(1j * th1)
    phi2 = (pB / norm) * jnp.exp(1j * solid_angle([ring]))
    return phi1.astype(jnp.complex128), phi2.astype(jnp.complex128)


def build_ic_hopf(N, L, R2, az, core, ax_frac=1.2, n=400):
    """DECOUPLED single Hopf link: a SMALL φ₂ ring (radius R2 → short global string,
    small g2) threaded by a LARGE φ₁ loop (a tall ellipse of vertical semi-axis az →
    spread flux, low el/mag). The linking requires the φ₁ loop's horizontal reach to
    cross the φ₂ disk once ⟹ ax = ax_frac·R2 with ax_frac∈(1,2). az is FREE, so the φ₁
    flux size decouples from the φ₂ string length — the test of whether the el/mag∝1/R²
    win came from φ₁-spread (decoupleable → maybe binds) or φ₂-string-length (tied to g2
    → tension is intrinsic). Q=−1."""
    g = np.linspace(-L / 2, L / 2, N, endpoint=False)
    Xn, Yn, Zn = np.meshgrid(g, g, g, indexing="ij")
    X, Y, Z = jnp.asarray(Xn), jnp.asarray(Yn), jnp.asarray(Zn)
    apex = np.array([0.0, 0.0, 0.9 * L])
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ax = ax_frac * R2                                                  # <2R2 ⟹ linked
    phi2_ring = np.stack([R2 * np.cos(t), R2 * np.sin(t), 0 * t], 1)   # small φ₂ ring
    phi1_loop = np.stack([R2 + ax * np.cos(t), 0 * t, az * np.sin(t)], 1)  # tall φ₁ loop

    def dist(curves):
        d = jnp.full(X.shape, jnp.inf)
        for c in curves:
            for p in c:
                d = jnp.minimum(d, jnp.sqrt((X - p[0]) ** 2 + (Y - p[1]) ** 2 + (Z - p[2]) ** 2))
        return d

    def solid_angle(curves):
        th = jnp.zeros_like(X)
        ax_, ay_, az_ = apex[0] - X, apex[1] - Y, apex[2] - Z
        na = jnp.sqrt(ax_ * ax_ + ay_ * ay_ + az_ * az_)
        for c in curves:
            Om = jnp.zeros_like(X)
            M = len(c)
            for i in range(M):
                B, Cc = c[i], c[(i + 1) % M]
                bx, by, bz = B[0] - X, B[1] - Y, B[2] - Z
                cx, cy, cz = Cc[0] - X, Cc[1] - Y, Cc[2] - Z
                nb = jnp.sqrt(bx * bx + by * by + bz * bz); nc = jnp.sqrt(cx * cx + cy * cy + cz * cz)
                crx, cry, crz = by * cz - bz * cy, bz * cx - bx * cz, bx * cy - by * cx
                tri = ax_ * crx + ay_ * cry + az_ * crz
                den = (na * nb * nc + (ax_ * bx + ay_ * by + az_ * bz) * nc
                       + (ax_ * cx + ay_ * cy + az_ * cz) * nb + (bx * cx + by * cy + bz * cz) * na)
                Om = Om + 2 * jnp.arctan2(tri, den)
            th = th + 0.5 * Om
        return th

    prof = lambda d: jnp.tanh(d / core)
    pA = prof(dist([phi1_loop])); pB = prof(dist([phi2_ring]))
    norm = jnp.sqrt(pA ** 2 + pB ** 2 + 1e-6)
    phi1 = (pA / norm) * jnp.exp(1j * solid_angle([phi1_loop]))
    phi2 = (pB / norm) * jnp.exp(1j * solid_angle([phi2_ring]))
    return phi1.astype(jnp.complex128), phi2.astype(jnp.complex128)


def seed_screened_A(phi1, dx, eps_a, q1):
    """Maximally-screened (Meissner) gauge field: A_i = (1/q1)·∂_i(arg φ₁).

    With φ₁=R·e^{iθ}, this gives D_iφ₁ = (∂_iR)·e^{iθ} — the phase winding is FULLY
    screened, so |D_iφ₁|² collapses to just the core-profile gradient (∂_iR)² and the
    quantized 2π flux is confined to the φ₁ strings (B=∇×A → delta-like tubes). This is
    the deep-type-II state EHN's relaxation is supposed to reach; the audit (Eq.7,
    E_elec∝1/R³ for pinned ρ) predicts C=400 should HOLD it. The `relax_flux_london`
    seed, by contrast, leaves g1 un-screened and dominant — the state that expels."""
    ga = axion_grad(phi1, dx, eps_a)          # ∂_i(arg φ₁), core-regularised
    return tuple(g / q1 for g in ga)


def _atomic_write(path, write_fn):
    """Write via <path>.tmp then os.replace, so `path` is NEVER observable
    half-written. os.replace is atomic on POSIX and overwrites in one step.

    Two failures this fixes, both real:

    1. A fleet fetcher polling every 120 s copied field.npz WHILE this function was
       writing it. The copy had a valid PK header and a plausible 86 MB size and
       failed only on open ("not a zip file"), which cost a completed N=192 arm its
       determinant -- the manifest survived, so Lk and nseg did, but the field did
       not. The mitigation already in this file (put the topology in the small
       manifest, "the fetch that truncated before") reduced the damage; this removes
       the cause.
    2. "Single overwriting file" meant a crash mid-write destroyed the PREVIOUS
       checkpoint too, so a host dying during a save lost both states. With a temp
       file, a torn write leaves garbage in .tmp and the last good checkpoint intact
       -- which is the whole point of checkpointing.

    fsync before replace so the bytes are on disk, not just in the page cache,
    before the name starts pointing at them.
    """
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        write_fn(f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _save_field(path, u, s, w, n):
    """Persist full field state (u,s,w) + step index for --resume + post-hoc diagnostics
    (stationarity metric, ⟨Lk⟩ tracer, portraits). float64, atomically replaced.

    np.savez is handed an open FILE OBJECT, not a name: given a name ending in
    ".tmp" it would helpfully append ".npz" and defeat the rename."""
    _atomic_write(Path(str(path)), lambda f: np.savez(
        f, u=np.stack([np.asarray(x) for x in u]),
        s=np.asarray(s), w=np.stack([np.asarray(x) for x in w]), n=n))


def _load_field(path):
    d = np.load(str(path))
    u = tuple(jnp.asarray(d["u"][i]) for i in range(10))
    w = tuple(jnp.asarray(d["w"][i]) for i in range(3))
    return u, jnp.asarray(d["s"]), w, int(d["n"])


def _seed_params(geom, tp, tq, R, rminor, twist, nlink):
    """The seed fields the manifest must carry to identify which IC produced a state.

    Without these, `--geom rings --nlink 3` and `--geom torus --tp 2 --tq 3` write
    BYTE-IDENTICAL params: the torus branch sets nlink = tq, so both record
    nlink=3 and nothing else distinguishes them. Those are different physics --
    rings threads nlink separate phi1 loops on the phi2 ring (EHN's IC), torus
    winds a SINGLE phi1 curve p times round and q times through -- and the held
    T(2,3) trefoil at N_link=3 exists only on the torus branch. A stored state
    whose manifest cannot say which IC produced it cannot be reproduced from its
    manifest, which is the manifest's whole job.

    `rminor` must be the value the IC ACTUALLY used, resolved by the caller --
    not None with the 0.45*R default reapplied here. Two copies of a default
    drift, and a rounded copy is not the number the run used.

    Found 2026-08-02 trying to re-derive the trefoil's seed from field_store: the
    recorded params were consistent with either.
    """
    p = {"geom": geom, "R": float(R), "nlink": int(nlink)}
    if geom == "torus":
        p.update({"tp": int(tp), "tq": int(tq), "twist": int(twist),
                  "rminor": float(rminor)})
    return p


def _seed_from_resumed(resume_path):
    """Carry the seed block FORWARD from the resumed state, never from argv.

    A resumed run builds no IC, so this invocation's --geom/--tp/--tq/--R say
    nothing about the state actually being relaxed. Reading them anyway is worse
    than recording nothing: the manifest becomes CONFIDENTLY WRONG rather than
    merely ambiguous, and silently so.

    That is the dominant path, not a corner case. soliton-playground's
    `_relax_cmd` does `if resume: cmd += ["--resume", ...] else: cmd += geom_args`,
    so every resumed leg in the standard box drops --geom, --tp, --tq and --R.
    B6's headroom probes resume the B2 trefoil (real R = 0.22*153.6 = 33.79) and
    would otherwise file `geom: "rings", R: 14.0` -- the CLI defaults.

    Returns the previous manifest's seed keys, or an explicit unavailable marker.
    Recording nothing is recoverable; recording the wrong thing is not.
    """
    prev = Path(resume_path).parent / "manifest.json"
    out = {"resumed_from": str(resume_path)}
    try:
        params = json.loads(prev.read_text())["params"]
    except Exception:                                          # noqa: BLE001
        return {**out, "geom": None, "seed_provenance": "unavailable"}
    carried = {k: params[k] for k in ("geom", "R", "nlink", "tp", "tq",
                                      "twist", "rminor") if k in params}
    if not carried:
        return {**out, "geom": None, "seed_provenance": "unavailable"}
    return {**out, **carried, "seed_provenance": "carried-forward"}


def run(N=96, L=76.8, nlink=4, R=14.0, core=2.0, lam=1000.0, kappa=0.0008,
        C=400.0, U=50.0, eps_a=0.05, alpha=4e-4, beta=2e-3, q1=1.0, q2=0.0,
        steps=40000, samples=40, n_ic=400, ic="london", cramp=0, agrad="bilinear",
        c4=0.0, out="out_ehn_relax", resume=None, save_every=0,
        geom="rings", tp=2, tq=3, rminor=None, twist=0, topo_every=0,
        det_every=0, det_timeout=180.0):
    global AGRAD, C4
    AGRAD = agrad        # before the first jit trace, so E_disc/relax_iter pick it up
    C4 = c4              # Skyrme quartic ℒ₃ coeff (0 = off)
    dx = L / N
    kv = EK.kvecs(N, L)
    outp = Path(out); outp.mkdir(parents=True, exist_ok=True)
    fld = outp / "field.npz"
    z = jnp.zeros((N, N, N))
    rr = rminor if rminor is not None else 0.45 * R   # ONE copy of the default;
                                                      # _seed_params records this
    if resume:
        u, s, w, n_start = _load_field(resume)
        seed = _seed_from_resumed(resume)
        # nlink drives `floor`, and so the printed link% and the manifest's floor.
        # It used to be left at the ARGUMENT default on this path (4), while
        # _relax_cmd drops --nlink on resume -- so a resumed torus T(2,3) scored
        # its link against a floor 4/3 too large. Carry it forward with the seed.
        if seed.get("nlink") is not None:
            nlink = int(seed["nlink"])
        print(f"RESUMED from {resume} at n={n_start} (skipping IC build); "
              f"seed {seed.get('seed_provenance', 'unavailable')}, nlink={nlink}")
    else:
        # n_ic = curve-segment resolution for the IC solid-angle phase (GPU builder).
        if geom == "torus":
            phi1, phi2 = build_ic_torus(N, L, tp, tq, R, rr, core, n=max(n_ic, 800),
                                        twist=twist)
            nlink = tq        # the q meridian winds encircle φ₂ ⟹ Lk floor = q
        else:
            phi1, phi2 = build_ic_gpu(N, L, nlink, R, core, n=n_ic)
        if ic == "screened":
            Ax, Ay, Az = seed_screened_A(phi1, dx, eps_a, q1)  # Dφ₁≈0, flux confined
        else:
            Ax, Ay, Az = EK.relax_flux_london(phi1, kv, N, niter=400)  # seed linked flux
        Bx, By, Bz = curlA(Ax, Ay, Az, dx)
        u = (jnp.real(phi1), jnp.imag(phi1), jnp.real(phi2), jnp.imag(phi2),
             Ax, Ay, Az, Bx, By, Bz)
        s = z            # A0
        w = (z, z, z)    # multiplier
        n_start = 0
        seed = _seed_params(geom, tp, tq, R, rr, twist, nlink)
    # Built ONCE. Two hand-curated copies at the two manifest writes drifted the
    # last time an argument was added, and the functional parameters below
    # (lam/kappa/eps_a/q1/q2/c4) and IC parameters (core/n_ic) were missing
    # entirely -- so a manifest could name its seed and still not reproduce its
    # state. If you add an argument to run(), add it here.
    params = {"N": N, "L": L, "C": C, "alpha": alpha, "beta": beta, "U": U,
              "ic": ic, "cramp": cramp, "agrad": agrad,
              "lam": lam, "kappa": kappa, "eps_a": eps_a, "q1": q1, "q2": q2,
              "c4": c4, "core": core, "n_ic": n_ic, **seed}
    floor = (2 * PI) ** 2 * nlink
    ehn_ref = {4: 6.0e3, 5: 7.0e3}.get(nlink)
    print(f"EHN FAITHFUL relaxer  N={N} L={L} dx={dx:.3f} nlink={nlink} C={C} "
          f"α={alpha} β={beta} U={U} q1={q1} q2={q2} ic={ic} agrad={agrad}  (EHN E≈{ehn_ref})")
    outp.mkdir(parents=True, exist_ok=True)
    t0 = time.time(); traj = []
    every = max(1, steps // samples)
    for n in range(n_start, steps + 1):
        # adiabatic C-ramp: 0→C over the first `cramp` steps so A₀ tracks the pinned ρ
        # instead of shocking (avoids the dt·C² electric runaway on a floor-level link).
        Cn = C * min(1.0, n / cramp) if cramp > 0 else C
        if n % every == 0:
            p1 = u[0] + 1j * u[1]; p2 = u[2] + 1j * u[3]
            Q = float(EK.skyrmion_number(p1, p2, kv, dx))
            E = energy_report(u, s, w, dx, lam, kappa, Cn, U, eps_a, q1, q2)
            print(f"  n{n:6d}: Q={Q:+.3f} link={E['link']/floor*100:+.0f}% C={Cn:.0f} "
                  f"E={E['total']:8.1f} (g1={E['grad1']:.0f} g2={E['grad2']:.0f} "
                  f"mag={E['mag']:.1f} el={E['elec']:.1f} sky={E['sky']:.0f} pot={E['pot']:.0f})", flush=True)
            entry = {"n": n, "Q": Q, "C": Cn, **E}
            # opt-in per-sample topology series (--topo-every K = every Kth sample):
            # cross-Lk(φ₁,φ₂) + per-species skeleton segment counts, into the manifest
            # (feeds the ECS recorder's P-odd ledger offline; default 0 = off, zero
            # behavior change for existing campaigns).
            want_topo = bool(topo_every) and (n // every) % topo_every == 0
            # A DIVERGED FIELD HAS NO KNOT TYPE, and recording one anyway is worse
            # than recording nothing. Skeletonising a NaN field yields dozens of
            # fragments that all identify as det=1, i.e. exactly the "det 5 -> 1"
            # signature a torus-knot run is looking for, produced for entirely the
            # wrong reason. The loop already breaks on non-finite E ("BLEW UP") --
            # but four lines too late, after the sample was written. So gate on the
            # same condition the break uses, and record null instead.
            finite = bool(np.isfinite(E["total"]))
            want_det = (bool(det_every) and (n // every) % det_every == 0
                        and finite)
            # ONE device->host copy, shared. At 320³ every np.asarray(u[i]) moves
            # 262 MB, so building φ₁ separately for each diagnostic would shift half
            # a gigabyte per sample to compute the same array twice.
            p1n = (np.asarray(u[0]) + 1j * np.asarray(u[1])
                   if (want_topo or want_det) else None)
            if want_topo:
                try:
                    from .cross_linking import cross_linking
                    p2n = np.asarray(u[2]) + 1j * np.asarray(u[3])
                    xlk, ns1, ns2 = cross_linking(p1n, p2n, dx)
                    entry.update({"xlk": (round(float(xlk), 3)
                                          if np.isfinite(xlk) else None),
                                  "nseg1": int(ns1), "nseg2": int(ns2)})
                    print(f"        topo: Lk(φ1,φ2)={entry['xlk']} "
                          f"nseg1={ns1} nseg2={ns2}", flush=True)
                except Exception as e:
                    print(f"        (topo sample skipped: {e})", flush=True)
            # opt-in per-sample SELF-KNOT determinant of the φ₁ string
            # (--det-every K). This is the quantity a torus-knot run at EHN's box
            # size exists to measure, and the three above cannot substitute for it:
            # cross-Lk says the two strings are still linked to each other, nseg
            # says the skeleton changed size, and NEITHER dates a self-reconnection.
            # det 5 → 1 is what does, and it is what the local cinquefoil showed
            # between 33k and 36k steps while its Lk held at −5.0 throughout.
            #
            # It lands in the manifest rather than requiring the field, because on
            # 2026-08-03 a 3.7 GB field.npz truncated to 2.1 GB crossing a vast SSH
            # proxy and the whole deliverable of a $0.42 rental went with it. A 9 KB
            # manifest that already carries the answer cannot truncate that way.
            #
            # TIME-BOXED, and that is not belt-and-braces: identify_knot's own
            # docstring records multi-hour grinds on noisy evolved curves, and
            # knot_determinants catches exceptions per line but an infinite grind is
            # not an exception. A diagnostic must never wedge a paid descent, so on
            # timeout the sample records null and the relaxation carries on.
            if want_det:
                try:
                    from ..knots import with_time_limit
                    from ..vortex_topology import knot_determinants
                    dets = with_time_limit(
                        det_timeout, lambda: knot_determinants(p1n, dx, L), None)
                    entry["det1"] = dets
                    print(f"        det(φ1): "
                          + (f"{dets}" if dets is not None
                             else f"UNIDENTIFIED (>{det_timeout:.0f}s)"), flush=True)
                except Exception as e:
                    print(f"        (det sample skipped: {e})", flush=True)
            elif bool(det_every) and (n // every) % det_every == 0:
                # Explicit null, not an absent key: "we looked and the field was
                # not a field" is a different statement from "we never looked".
                entry["det1"] = None
                print("        det(φ1): SKIPPED — E is not finite", flush=True)
            traj.append(entry)
            if not np.isfinite(E["total"]):
                print("  BLEW UP"); break
            _atomic_write(outp / "manifest.json", lambda f: f.write(json.dumps(
                {"params": params,
                 "floor": floor, "traj": traj,
                 "wall_s": time.time() - t0}, indent=1).encode()))
        if save_every and n % save_every == 0 and n > n_start:
            _save_field(fld, u, s, w, n)      # periodic checkpoint (resume/crash-safe)
        if n < steps:
            u, s, w = relax_iter(u, s, w, dx, lam, kappa, Cn, U, eps_a, alpha, beta, q1, q2)
    if save_every:
        _save_field(fld, u, s, w, steps)       # persist final state (opt-in; ~3.7GB at N=320)
    E = energy_report(u, s, w, dx, lam, kappa, C, U, eps_a, q1, q2)
    # on-box cross-linking diagnostic → into the (small) manifest, so we never have to
    # fetch the 3.7GB field just to read the topology (the fetch that truncated before).
    cross_lk = None
    p1 = None
    try:
        from .cross_linking import cross_linking
        p1 = np.asarray(u[0]) + 1j * np.asarray(u[1])
        p2 = np.asarray(u[2]) + 1j * np.asarray(u[3])
        cross_lk = round(float(cross_linking(p1, p2, dx)[0]), 3)
    except Exception as e:
        print(f"  (cross-linking skipped: {e})")
    # ...and the END-STATE φ₁ self-knot determinant, not gated on --det-every: it is
    # one call on a state that is already in host memory, and it is the headline
    # number a torus-knot run is judged by. `(size, det)` pairs are the form the
    # particle catalog already registers -- cinquefoil_t25 is [[822, 5]] -- so this
    # is directly comparable to a catalog entry rather than needing conversion.
    #
    # It IS gated on the energy being finite. A diverged run still has a φ₁ array,
    # and skeletonising it returns dozens of fragments identifying as det=1 --
    # which is the "det 5 -> 1" signature this measurement exists to detect,
    # manufactured by the divergence rather than observed. None is the honest value.
    #
    # NOTE for consumers: knot_determinants reports a per-line failure as the STRING
    # f"e:{ExceptionName}" in place of that line's integer, so det1 is a list of
    # (int, int|str) and anything diffing it against a catalog entry must cope with
    # that. Kept as a string rather than None because it says WHICH failure.
    det1 = None
    E_finite = bool(np.isfinite(E["total"]))
    if not E_finite:
        print("  determinant: SKIPPED — final E is not finite, so there is no knot "
              "to identify (recording null rather than a manufactured det=1)")
    else:
        try:
            from ..knots import with_time_limit
            from ..vortex_topology import knot_determinants
            if p1 is None:
                p1 = np.asarray(u[0]) + 1j * np.asarray(u[1])
            det1 = with_time_limit(det_timeout,
                                   lambda: knot_determinants(p1, dx, L), None)
        except Exception as e:
            print(f"  (determinant skipped: {e})")
    _atomic_write(outp / "manifest.json", lambda f: f.write(json.dumps(
        {"params": params,
         "floor": floor, "traj": traj, "wall_s": time.time() - t0,
         # det_every/det_timeout are deliberately NOT in `params`: that dict is the
         # set that reproduces the STATE (its own comment says so, and
         # _seed_from_resumed carries it forward), and a diagnostic cadence does not
         # change the descent by one step. Recorded here instead, so the manifest
         # still says what was measured and how often.
         "det_every": det_every, "det_timeout": det_timeout,
         # Disambiguates a null det1: False means the field diverged and there was
         # nothing to identify; True with det1 null means the identification itself
         # timed out. Both are "no answer", for reasons a reader must not conflate.
         "e_finite": E_finite,
         "cross_lk": cross_lk, "det1": det1}, indent=1).encode()))
    print(f"  FINAL E={E['total']:.1f} (EHN≈{ehn_ref}) link={E['link']/floor*100:+.0f}% "
          f"Lk(φ1,φ2)={cross_lk} det(φ1)={det1} "
          f"mag={E['mag']:.1f} el={E['elec']:.1f}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=96)
    ap.add_argument("--L", type=float, default=76.8)
    ap.add_argument("--nlink", type=int, default=4)
    ap.add_argument("--R", type=float, default=14.0)
    ap.add_argument("--core", type=float, default=2.0)
    ap.add_argument("--lam", type=float, default=1000.0)
    ap.add_argument("--kappa", type=float, default=0.0008)
    ap.add_argument("--C", type=float, default=400.0)
    ap.add_argument("--U", type=float, default=50.0)
    ap.add_argument("--eps-a", type=float, default=0.05)
    ap.add_argument("--alpha", type=float, default=1e-4,
                    help="descent step. STABILITY: alpha < 2/H, with "
                         "H ~ max(8*lam, 153/dx^2) -- flat at 8000 (lam=1000) "
                         "down to dx~0.14, then 1/dx^2. THIS DEFAULT IS SAFE "
                         "TO dx~0.088 and diverges below it (3.3e-5 is the "
                         "limit at dx=0.05). Scale alpha as 1/dx^2 when "
                         "refining; recompute if lam/U/kappa/c4 change.")
    ap.add_argument("--beta", type=float, default=2e-3)
    ap.add_argument("--q1", type=float, default=1.0)
    ap.add_argument("--q2", type=float, default=0.0)
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--samples", type=int, default=40)
    ap.add_argument("--n-ic", type=int, default=400)
    ap.add_argument("--ic", choices=["london", "screened"], default="london")
    ap.add_argument("--cramp", type=int, default=0, help="ramp C 0→C over this many steps")
    ap.add_argument("--agrad", choices=["bilinear", "wrapped", "naive"],
                    default="bilinear",
                    help="∂a discretization; wrapped = modulus-blind exact-winding "
                         "lock, naive = central difference of arctan2 (branch cut "
                         "included -- the literal reading of EHN's method)")
    ap.add_argument("--c4", type=float, default=0.0,
                    help="Skyrme quartic ℒ₃ coeff (single-knot stabilizer; 0 = off)")
    ap.add_argument("--out", default="out_ehn_relax")
    ap.add_argument("--resume", default=None,
                    help="path to a field.npz to continue from (skips IC build)")
    ap.add_argument("--save-every", type=int, default=0,
                    help="checkpoint the full field every N steps (0 = final only)")
    ap.add_argument("--topo-every", type=int, default=0,
                    help="opt-in: cross-Lk + skeleton counts every Kth sample into "
                         "the manifest traj (0 = off; feeds the ECS recorder)")
    ap.add_argument("--det-every", type=int, default=0,
                    help="opt-in: phi1 SELF-KNOT determinant every Kth sample into "
                         "the manifest traj (0 = off). Answers what cross-Lk cannot: "
                         "Lk is the link BETWEEN the strings and holds while one "
                         "string unknots ITSELF, so only the determinant dates a "
                         "self-reconnection (det 5 -> 1). The END-STATE determinant "
                         "is always recorded regardless of this flag; K only buys "
                         "the trajectory. Costs a skeleton trace plus a pyknotid "
                         "Alexander per sample -- time-boxed by --det-timeout.")
    ap.add_argument("--det-timeout", type=float, default=180.0,
                    help="wall-clock budget per determinant (s). identify_knot can "
                         "grind for hours on a noisy evolved curve, and a diagnostic "
                         "must never wedge a paid descent: on timeout the sample "
                         "records null and the relaxation continues.")
    ap.add_argument("--geom", choices=["rings", "torus"], default="rings",
                    help="φ₁ geometry: rings = nlink rings on the φ₂ ring (EHN); "
                         "torus = one T(p,q) torus knot around the φ₂ ring (trefoil test)")
    ap.add_argument("--tp", type=int, default=2, help="(--geom torus) longitudinal winding p")
    ap.add_argument("--tq", type=int, default=3, help="(--geom torus) meridional winding q")
    ap.add_argument("--rminor", type=float, default=None,
                    help="(--geom torus) torus minor radius, default 0.45*R")
    ap.add_argument("--twist", type=int, default=0,
                    help="(--geom torus) framing twist: wind arg φ₁ by twist×Θ_ring "
                         "(internal-U(1) label along the string; the n-vs-p DOF candidate)")
    a = ap.parse_args()
    run(N=a.N, L=a.L, nlink=a.nlink, R=a.R, core=a.core, lam=a.lam, kappa=a.kappa,
        C=a.C, U=a.U, eps_a=a.eps_a, alpha=a.alpha, beta=a.beta, q1=a.q1, q2=a.q2,
        steps=a.steps, samples=a.samples, n_ic=a.n_ic, ic=a.ic, cramp=a.cramp,
        agrad=a.agrad, c4=a.c4, out=a.out, resume=a.resume, save_every=a.save_every,
        geom=a.geom, tp=a.tp, tq=a.tq, rminor=a.rminor, twist=a.twist,
        topo_every=a.topo_every, det_every=a.det_every,
        det_timeout=a.det_timeout)

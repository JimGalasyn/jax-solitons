#!/usr/bin/env python3
"""On-box driver: relax an EHN two-scalar KNOT SOLITON and record its fate.

Self-contained (jax + numpy only). Reproduces the Eto-Hamada-Nitta knot soliton
(PRL 135, 091603, 2025): φ₁ GAUGED (flux tube) + φ₂ GLOBAL (axion string), coupled
O(4) potential V=λ(|φ₁|²+|φ₂|²−1)²−κ|φ₁|²|φ₂|² with λ≫g² (so N_link = skyrmion
number π₃(S³)=ℤ, topologically protected), and the Chern-Simons coupling
C·a·F F̃ (a=argφ₂) whose A_0 electric energy stabilises the loop SIZE — handled
with the EHN auxiliary-field (B_i + multiplier w) stabilisation.

IC = N_link φ₁ rings threaded by one φ₂ ring (the EHN knot), S³-constrained
(|φ₁|²+|φ₂|²=1 → clean integer skyrmion charge). We relax and record the skyrmion
charge Q(τ), energy, and loop size: does the knot HOLD (Q→integer, finite size)
or collapse? Writes out/manifest.json.

  python ehn_knot_batch.py --N 160 --L 12 --lam 1000 --C 400 --nlink 4 --out out
"""
import argparse, json, time, warnings
from pathlib import Path
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from functools import partial


def kvecs(N, L):
    k1 = 2 * np.pi * np.fft.fftfreq(N, d=L / N)
    KX, KY, KZ = jnp.meshgrid(jnp.asarray(k1), jnp.asarray(k1), jnp.asarray(k1), indexing="ij")
    return KX, KY, KZ, KX**2 + KY**2 + KZ**2


def coulomb_project(Ax, Ay, Az, kv):
    KX, KY, KZ, K2 = kv
    Axh, Ayh, Azh = jnp.fft.fftn(Ax), jnp.fft.fftn(Ay), jnp.fft.fftn(Az)
    kdotA = KX * Axh + KY * Ayh + KZ * Azh
    inv = jnp.where(K2 > 0, 1.0 / K2, 0.0)
    return (jnp.real(jnp.fft.ifftn(Axh - KX * kdotA * inv)),
            jnp.real(jnp.fft.ifftn(Ayh - KY * kdotA * inv)),
            jnp.real(jnp.fft.ifftn(Azh - KZ * kdotA * inv)))


def curl(Ax, Ay, Az, kv):
    KX, KY, KZ, _ = kv
    Axh, Ayh, Azh = jnp.fft.fftn(Ax), jnp.fft.fftn(Ay), jnp.fft.fftn(Az)
    return (jnp.real(jnp.fft.ifftn(1j * (KY * Azh - KZ * Ayh))),
            jnp.real(jnp.fft.ifftn(1j * (KZ * Axh - KX * Azh))),
            jnp.real(jnp.fft.ifftn(1j * (KX * Ayh - KY * Axh))))


def magnetic_helicity(Ax, Ay, Az, kv, dx):
    bx, by, bz = curl(Ax, Ay, Az, kv)
    return float(jnp.sum(Ax * bx + Ay * by + Az * bz) * dx**3)


def skyrmion_number(phi1, phi2, kv, dx):
    KX, KY, KZ, _ = kv
    Phi = jnp.sqrt(jnp.abs(phi1) ** 2 + jnp.abs(phi2) ** 2 + 1e-12)
    n = [jnp.real(phi1) / Phi, jnp.imag(phi1) / Phi, jnp.real(phi2) / Phi, jnp.imag(phi2) / Phi]
    g = []
    for na in n:
        nah = jnp.fft.fftn(na)
        g.append((jnp.real(jnp.fft.ifftn(1j * KX * nah)),
                  jnp.real(jnp.fft.ifftn(1j * KY * nah)),
                  jnp.real(jnp.fft.ifftn(1j * KZ * nah))))
    def det3(p, q, r):
        px, py, pz = g[p]; qx, qy, qz = g[q]; rx, ry, rz = g[r]
        return px * (qy * rz - qz * ry) + py * (qz * rx - qx * rz) + pz * (qx * ry - qy * rx)
    dens = (n[0] * det3(1, 2, 3) - n[1] * det3(0, 2, 3)
            + n[2] * det3(0, 1, 3) - n[3] * det3(0, 1, 2))
    return float(jnp.sum(dens) * dx**3 / (2.0 * np.pi**2))


# ---- IC: N_link φ₁ rings threaded by one φ₂ ring (the EHN knot) ----
def _ring(center, axis, R, n=400):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    if axis == "z":   loop = np.stack([R * np.cos(t), R * np.sin(t), 0 * t], 1)
    elif axis == "x": loop = np.stack([0 * t, R * np.cos(t), R * np.sin(t)], 1)
    else:             loop = np.stack([R * np.cos(t), 0 * t, R * np.sin(t)], 1)
    return loop + np.asarray(center)


def _solid_angle_phase(curves, apex, X, Y, Z):
    """Σ ½·VOS solid angle (Van Oosterom-Strackee) → 2π winding around each curve."""
    th = np.zeros_like(X)
    ax, ay, az = apex[0] - X, apex[1] - Y, apex[2] - Z
    na = np.sqrt(ax * ax + ay * ay + az * az)
    for c in curves:
        Om = np.zeros_like(X)
        M = len(c)
        for i in range(M):
            B, Cc = c[i], c[(i + 1) % M]
            bx, by, bz = B[0] - X, B[1] - Y, B[2] - Z
            cx, cy, cz = Cc[0] - X, Cc[1] - Y, Cc[2] - Z
            nb = np.sqrt(bx * bx + by * by + bz * bz); nc = np.sqrt(cx * cx + cy * cy + cz * cz)
            crx, cry, crz = by * cz - bz * cy, bz * cx - bx * cz, bx * cy - by * cx
            tri = ax * crx + ay * cry + az * crz
            den = (na * nb * nc + (ax * bx + ay * by + az * bz) * nc
                   + (ax * cx + ay * cy + az * cz) * nb + (bx * cx + by * cy + bz * cz) * na)
            # NOTE `den` cannot be -0.0: its first term na*nb*nc is a product of
            # norms, so the sum is +0.0 or positive. arctan2(0, +0.0) is already
            # 0.0, so there is no sign-of-zero ambiguity here to guard against --
            # measured on the degenerate N=48 geometry, 4 sites hit tri==den==0
            # and 0 had a negative-zero den. (A `both_zero -> 0` guard was tried
            # and was bitwise a no-op.) The real discontinuity in this expression
            # is `den` CROSSING zero with tri small, which no such guard catches.
            Om += 2 * np.arctan2(tri, den)
        th += 0.5 * Om
    return th


def _dist(curves, X, Y, Z):
    d = np.full(X.shape, np.inf)            # low-memory: loop curve points, vectorise grid
    for c in curves:
        for p in c:
            d = np.minimum(d, np.sqrt((X - p[0])**2 + (Y - p[1])**2 + (Z - p[2])**2))
    return d


class LatticeCoincidenceWarning(UserWarning):
    """A phi1 seed curve touches a lattice site: degenerate, observed benign."""


class LatticeCoincidence(ValueError):
    """A seed curve passes exactly through a lattice site.

    `prof(d) = tanh(d/core)` is then EXACTLY 0 there, so |phi| = 0 at a grid point:
    a vortex core pinned to a site, where the phase is undefined and the winding
    cannot be represented. Runs seeded this way diverge -- measured 2026-08-07, an
    N=320 leg NaNed by step 1000 while N=192 (no coincidence) ran 36000 steps
    clean, and a 1-ULP change in L flipped an N=48 run between the two.

    It is not a small perturbation to be tolerated. It is a different, degenerate
    initial condition, and which one you get is decided by whether R/dx lands on an
    integer in binary floating point -- i.e. by luck, silently.
    """


#: agrad modes that DIFFERENCE arg(phi2) directly, and so read the axion phase at
#: every site. For these an exact |phi2| = 0 on a lattice point is fatal. The
#: default `bilinear` instead computes Im(conj(phi2) d phi2)/(|phi2|^2 + eps_a),
#: where eps_a regularises the zero and the same IC runs fine -- measured at N=320
#: (which carries a phi2 coincidence): bilinear ran 36000 steps clean while wrapped
#: NaNed at step 1000, same geometry, same box.
PHASE_DIFFERENCING_AGRAD = ("wrapped", "naive")


def _assert_off_lattice(dA, dB, *, N, L, R, agrad=None):
    """Refuse an IC whose phi2 ring touches a lattice site, WHEN agrad reads the
    phase pointwise; warn otherwise, and warn on phi1 always.

    Checked on the DISTANCE field rather than reconstructed from R/dx, because the
    condition that matters is the one the profile actually sees, and `_dist` takes
    the min over the sampled curve -- which is what tanh is evaluated on.

    `prof(d) = tanh(d/core)` is exactly 0 where a seed sample lands on a site, so
    |phi| = 0 at a grid point: a vortex core pinned to the lattice.

        N     L                    phi2  phi1   outcome (agrad=wrapped)
        48    38.400000000000006     1     2    diverged by step 75
        64    51.2                   0     1    clean to 20000 steps
        96    76.80000000000001      1     2    diverged by step 75
        128   102.4                  0     1    clean
        192   153.6 (stage 1)        0     0    clean to 36000 steps
        320   256.0 (stage 2)        1     1    diverged by step 1000

    EVERY row above is agrad=wrapped, and that qualifier is load-bearing. At N=320
    the SAME coincidence was harmless to bilinear -- it ran the full 36000 steps
    and returned finite Q -- so the guard is conditioned on agrad rather than
    raising for all of them, which would refuse runs that demonstrably work.

    Two honest limits on what the table can support:

      - It cannot separate "phi2 is what matters" from "two or more coincidences
        is what matters": diverged rows total 3, 3, 2 and clean rows 1, 1, 0, so
        both rules give the identical partition. Preference for the phi2 rule
        rests on the bilinear-vs-wrapped comparison above and on the mechanism
        (only phase-differencing modes read arg(phi2) at a site), not on this
        table.
      - phi1-only coincidence is observed benign twice. That is thin, so it warns
        rather than passing silently.
    """
    def _where(d):
        return [tuple(int(v) for v in i) for i in np.argwhere(d == 0.0)[:3]]

    fatal = agrad is None or agrad in PHASE_DIFFERENCING_AGRAD
    n2 = int(np.count_nonzero(dB == 0.0))
    if n2:
        detail = (f"phi2 (big ring): {n2} lattice site(s) lie EXACTLY on the seed "
                  f"curve (N={N}, L={L!r}, R={R!r}, dx={L / N!r}); at "
                  f"{_where(dB)}{' (first 3)' if n2 > 3 else ''}. |phi2| is exactly "
                  f"0 there, so arg(phi2) is undefined at a grid point. Nudge R "
                  f"(or L) off the lattice, e.g. R += dx/2. NOTE this is decided "
                  f"by whether R/dx is an integer in binary floating point: "
                  f"L=38.4 and L=38.400000000000006 differ here.")
        if fatal:
            raise LatticeCoincidence(
                detail + f" agrad={agrad!r} differences the phase pointwise, so "
                f"this diverges (agrad=None is treated as fatal: unknown mode, "
                f"conservative answer).")
        warnings.warn(
            detail + f" agrad={agrad!r} regularises it with eps_a and has been "
            f"measured to survive this, so it is a warning -- but the IC is still "
            f"degenerate and was arrived at by floating-point luck.",
            LatticeCoincidenceWarning, stacklevel=3)
    n1 = int(np.count_nonzero(dA == 0.0))
    if n1:
        warnings.warn(
            f"phi1 (small rings): {n1} lattice site(s) lie exactly on the seed "
            f"curve (N={N}, L={L!r}, R={R!r}); at {_where(dA)}"
            f"{' (first 3)' if n1 > 3 else ''}. |phi1| is exactly 0 there. "
            f"Observed benign twice, and phi1's phase is never differenced the way "
            f"arg(phi2) is -- but see the docstring: the data cannot rule out that "
            f"the count of coincidences is what matters rather than which field.",
            LatticeCoincidenceWarning, stacklevel=3)


def build_ic(N, L, nlink, R, core, n=400, agrad=None):
    g = np.linspace(-L / 2, L / 2, N, endpoint=False)
    X, Y, Z = np.meshgrid(g, g, g, indexing="ij")
    apex = np.array([0.0, 0.0, 0.9 * L])
    # φ₂: one big global ring (z-plane, radius R). φ₁: nlink small rings, each
    # centred ON the big ring at angle θ_k, lying in the plane spanned by the
    # radial r̂_k and ẑ — so the big ring PIERCES each small ring (true linking).
    big = _ring([0, 0, 0], "z", R)
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    zhat = np.array([0.0, 0.0, 1.0]); r = 0.5 * R
    smalls = []
    for k in range(nlink):
        thk = 2 * np.pi * k / nlink
        ck = R * np.array([np.cos(thk), np.sin(thk), 0.0])
        rhat = np.array([np.cos(thk), np.sin(thk), 0.0])
        smalls.append(ck[None, :] + r * (np.cos(t)[:, None] * rhat[None, :] + np.sin(t)[:, None] * zhat[None, :]))
    prof = lambda d: np.tanh(d / core)
    dA, dB = _dist(smalls, X, Y, Z), _dist([big], X, Y, Z)
    _assert_off_lattice(dA, dB, N=N, L=L, R=R, agrad=agrad)
    pA = prof(dA); pB = prof(dB)
    norm = np.sqrt(pA**2 + pB**2 + 1e-6)
    phA = _solid_angle_phase(smalls, apex, X, Y, Z)
    phB = _solid_angle_phase([big], apex, X, Y, Z)
    phi1 = ((pA / norm) * np.exp(1j * phA)).astype(np.complex128)
    phi2 = ((pB / norm) * np.exp(1j * phB)).astype(np.complex128)
    return jnp.asarray(phi1), jnp.asarray(phi2)


@partial(jax.jit, static_argnums=())
def step(phi1, phi2, Ax, Ay, Az, Bx, By, Bz, wx, wy, wz, lp, KX, KY, KZ, K2,
         dt, C, lam, kappa, Mbar2, U, etaw, eps_a, cs):
    # cs = scalar-gradient coefficient of the energy functional being descended:
    #   cs=0.5 → ½|Dφ|² (legacy);  cs=1.0 → |Dφ|² (EHN arXiv:2407.11731 Eq.10).
    # All scalar-sector forces scale with cs so the relaxer descends the matching
    # functional (φ Laplacian exp, gauge cross-terms, and the A-source current 2cs·J).
    a1, a2 = jnp.abs(phi1) ** 2, jnp.abs(phi2) ** 2
    common = a1 + a2 - 1.0
    p1h = jnp.fft.fftn(phi1)
    d1x = jnp.fft.ifftn(1j * KX * p1h); d1y = jnp.fft.ifftn(1j * KY * p1h); d1z = jnp.fft.ifftn(1j * KZ * p1h)
    A2 = Ax**2 + Ay**2 + Az**2
    Adg = Ax * d1x + Ay * d1y + Az * d1z
    divA = jnp.real(jnp.fft.ifftn(1j * (KX * jnp.fft.fftn(Ax) + KY * jnp.fft.fftn(Ay) + KZ * jnp.fft.fftn(Az))))
    N1 = (cs * (-2j * Adg - 1j * divA * phi1 - A2 * phi1) - 2.0 * lam * common * phi1 + kappa * a2 * phi1)
    phi1_n = jnp.fft.ifftn(jnp.exp(-cs * K2 * dt) * (p1h + dt * jnp.fft.fftn(N1)))
    J1x = (2.0 * cs) * (jnp.imag(jnp.conj(phi1) * d1x) - Ax * a1)
    J1y = (2.0 * cs) * (jnp.imag(jnp.conj(phi1) * d1y) - Ay * a1)
    J1z = (2.0 * cs) * (jnp.imag(jnp.conj(phi1) * d1z) - Az * a1)
    p2h = jnp.fft.fftn(phi2)
    d2x = jnp.fft.ifftn(1j * KX * p2h); d2y = jnp.fft.ifftn(1j * KY * p2h); d2z = jnp.fft.ifftn(1j * KZ * p2h)
    N2 = -2.0 * lam * common * phi2 + kappa * a1 * phi2
    phi2_n = jnp.fft.ifftn(jnp.exp(-cs * K2 * dt) * (p2h + dt * jnp.fft.fftn(N2)))
    inv = 1.0 / (a2 + eps_a)
    sm = lambda r: jnp.real(jnp.fft.ifftn(lp * jnp.fft.fftn(r)))
    gax = sm(jnp.imag(jnp.conj(phi2) * d2x) * inv); gay = sm(jnp.imag(jnp.conj(phi2) * d2y) * inv); gaz = sm(jnp.imag(jnp.conj(phi2) * d2z) * inv)
    Axh, Ayh, Azh = jnp.fft.fftn(Ax), jnp.fft.fftn(Ay), jnp.fft.fftn(Az)
    cAx = jnp.real(jnp.fft.ifftn(1j * (KY * Azh - KZ * Ayh)))
    cAy = jnp.real(jnp.fft.ifftn(1j * (KZ * Axh - KX * Azh)))
    cAz = jnp.real(jnp.fft.ifftn(1j * (KX * Ayh - KY * Axh)))
    rho = gax * Bx + gay * By + gaz * Bz
    A0 = jnp.real(jnp.fft.ifftn(C * jnp.fft.fftn(rho) / (K2 + Mbar2)))
    Bx_n = Bx - dt * (C * A0 * gax + wx + U * (Bx - cAx))
    By_n = By - dt * (C * A0 * gay + wy + U * (By - cAy))
    Bz_n = Bz - dt * (C * A0 * gaz + wz + U * (Bz - cAz))
    Gx, Gy, Gz = Bx_n - cAx, By_n - cAy, Bz_n - cAz
    hx, hy, hz = wx + U * Bx_n, wy + U * By_n, wz + U * Bz_n
    hxh, hyh, hzh = jnp.fft.fftn(hx), jnp.fft.fftn(hy), jnp.fft.fftn(hz)
    chx = jnp.real(jnp.fft.ifftn(1j * (KY * hzh - KZ * hyh)))
    chy = jnp.real(jnp.fft.ifftn(1j * (KZ * hxh - KX * hzh)))
    chz = jnp.real(jnp.fft.ifftn(1j * (KX * hyh - KY * hxh)))
    expo_A = jnp.exp(-(1.0 + U) * K2 * dt)
    Ax_n = jnp.real(jnp.fft.ifftn(expo_A * (Axh + dt * jnp.fft.fftn(J1x + chx))))
    Ay_n = jnp.real(jnp.fft.ifftn(expo_A * (Ayh + dt * jnp.fft.fftn(J1y + chy))))
    Az_n = jnp.real(jnp.fft.ifftn(expo_A * (Azh + dt * jnp.fft.fftn(J1z + chz))))
    Ax_n, Ay_n, Az_n = coulomb_project(Ax_n, Ay_n, Az_n, (KX, KY, KZ, K2))
    return (phi1_n, phi2_n, Ax_n, Ay_n, Az_n, Bx_n, By_n, Bz_n,
            wx + etaw * Gx, wy + etaw * Gy, wz + etaw * Gz)


def _axion_grad(phi2, kv, eps_a, lp):
    """∇a (a=arg φ₂), regularised + smoothed — the fixed axion gradient."""
    KX, KY, KZ, K2 = kv
    p2h = jnp.fft.fftn(phi2)
    d2 = [jnp.fft.ifftn(1j * Kc * p2h) for Kc in (KX, KY, KZ)]
    inv = 1.0 / (jnp.abs(phi2) ** 2 + eps_a)
    sm = lambda r: jnp.real(jnp.fft.ifftn(lp * jnp.fft.fftn(r)))
    return (sm(jnp.imag(jnp.conj(phi2) * d2[0]) * inv),
            sm(jnp.imag(jnp.conj(phi2) * d2[1]) * inv),
            sm(jnp.imag(jnp.conj(phi2) * d2[2]) * inv))


@partial(jax.jit, static_argnums=())
def pin_link_flux(Ax, Ay, Az, phi2, KX, KY, KZ, K2, eps_a, lp, floor, dx3):
    """Hard flux-quantisation pin (continuum stand-in for EHN's lattice plaquette
    quantisation): project A so the topological linking flux ∫ρ=∫(∇a)·(∇×A) dV is
    held at the floor (2π)²N_link. The C-electric energy is positive and otherwise
    EXPELS the linked flux (A un-screens φ₁); EHN's compact U(1) forbids that by
    quantising ∮A·dl=2π. We restore it each step by δA = c·∇×(∇a),
    c=(floor−∫ρ dV)/(∫|∇×(∇a)|² dV), which adds magnetic flux exactly through the φ₂
    string. Units: link & denom both carry dx³ (physical) to match the floor."""
    kv = (KX, KY, KZ, K2)
    gax, gay, gaz = _axion_grad(phi2, kv, eps_a, lp)
    Bx, By, Bz = curl(Ax, Ay, Az, kv)
    link = jnp.sum(gax * Bx + gay * By + gaz * Bz) * dx3
    cx, cy, cz = curl(gax, gay, gaz, kv)            # ∇×(∇a): supported on the string
    denom = jnp.sum(cx * cx + cy * cy + cz * cz) * dx3 + 1e-30
    c = (floor - link) / denom
    Ax2, Ay2, Az2 = Ax + c * cx, Ay + c * cy, Az + c * cz
    return coulomb_project(Ax2, Ay2, Az2, kv)


def relax_flux_london(phi1, kv, N, niter=400):
    """Converged static London/Abrikosov flux: minimise ½|Dφ₁|²+½|B|² over A by
    solving (−∇²+|φ₁|²)A = J_super, J_super=Im(φ₁*∇φ₁), in Coulomb gauge via a
    preconditioned fixed point (constant-mass part inverted spectrally, the
    |φ₁|²−m² contrast carried explicitly). Builds the FULL quantised vortex flux
    in a few hundred iters — vs the slow damped step-prebuild that under-builds the
    large-scale (low-k) linking flux. Returns A in Coulomb gauge."""
    KX, KY, KZ, K2 = kv
    p1h = jnp.fft.fftn(phi1)
    d1 = [jnp.fft.ifftn(1j * Kc * p1h) for Kc in (KX, KY, KZ)]
    a1 = jnp.abs(phi1) ** 2
    m2 = jnp.mean(a1)
    Js = [jnp.imag(jnp.conj(phi1) * d1[i]) for i in range(3)]
    Ax = Ay = Az = jnp.zeros((N, N, N))
    denom = K2 + m2
    for _ in range(niter):
        Ax = jnp.real(jnp.fft.ifftn(jnp.fft.fftn(Js[0] + (m2 - a1) * Ax) / denom))
        Ay = jnp.real(jnp.fft.ifftn(jnp.fft.fftn(Js[1] + (m2 - a1) * Ay) / denom))
        Az = jnp.real(jnp.fft.ifftn(jnp.fft.fftn(Js[2] + (m2 - a1) * Az) / denom))
        Ax, Ay, Az = coulomb_project(Ax, Ay, Az, kv)
    return Ax, Ay, Az


def _solve_A0_posmass(rho, a1, kv, C, g2=1.0, niter=120):
    """EHN A₀ Gauss law with POSITION-DEPENDENT screening (arXiv:2407.11731 Eq.12):
        (−∂² + 2g²|φ₁|²) A₀ = C ρ ,   ρ=(∇a)·B.
    The screening mass 2g²|φ₁|² → 0 in the φ₁ cores (where the linked flux lives),
    vs the constant-M̄² approximation the relaxer uses for speed. Preconditioned
    fixed point: constant-mass part inverted spectrally, the contrast carried explicit."""
    KX, KY, KZ, K2 = kv
    mass = 2.0 * g2 * a1
    m2 = jnp.mean(mass)
    A0 = jnp.zeros_like(a1)
    src = C * rho
    for _ in range(niter):
        A0 = jnp.real(jnp.fft.ifftn(jnp.fft.fftn(src + (m2 - mass) * A0) / (K2 + m2)))
    return A0


def two_scalar_energy(phi1, phi2, Ax, Ay, Az, kv, dx, lam, kappa, C, Mbar2, eps_a, lp):
    """EHN-normalised energy (arXiv:2407.11731 Eq.10, v=g=1): NO ½ on the covariant
    scalar gradients |D_iφ₁|²+|∇φ₂|², ½ on the magnetic ½|B|², plus the A₀ Gauss-law
    electric energy ½C·A₀·ρ (ρ=(∇a)·B, A₀=C(−∂²+M̄²)⁻¹ρ — the size-dependent binding
    term). Const-M̄² A₀ to match the relaxer's own dynamics (posmass is ~+10% on the
    small e_elec). Returns total + components in box units (= EHN's v/g at v=g=1)."""
    KX, KY, KZ, K2 = kv
    p1h, p2h = jnp.fft.fftn(phi1), jnp.fft.fftn(phi2)
    d1 = [jnp.fft.ifftn(1j * Kc * p1h) for Kc in (KX, KY, KZ)]
    d2 = [jnp.fft.ifftn(1j * Kc * p2h) for Kc in (KX, KY, KZ)]
    A = (Ax, Ay, Az)
    e_grad1 = sum(jnp.abs(d1[i] - 1j * A[i] * phi1) ** 2 for i in range(3))   # |Dφ₁|²
    e_grad2 = sum(jnp.abs(d2[i]) ** 2 for i in range(3))                     # |∇φ₂|²
    Bx, By, Bz = curl(Ax, Ay, Az, kv)
    e_mag = 0.5 * (Bx**2 + By**2 + Bz**2)                                    # ½|B|²
    a1, a2 = jnp.abs(phi1) ** 2, jnp.abs(phi2) ** 2
    e_pot = lam * (a1 + a2 - 1.0) ** 2 - kappa * a1 * a2
    inv = 1.0 / (a2 + eps_a)
    sm = lambda r: jnp.real(jnp.fft.ifftn(lp * jnp.fft.fftn(r)))
    gax = sm(jnp.imag(jnp.conj(phi2) * d2[0]) * inv)
    gay = sm(jnp.imag(jnp.conj(phi2) * d2[1]) * inv)
    gaz = sm(jnp.imag(jnp.conj(phi2) * d2[2]) * inv)
    rho = gax * Bx + gay * By + gaz * Bz
    A0 = jnp.real(jnp.fft.ifftn(C * jnp.fft.fftn(rho) / (K2 + Mbar2)))
    e_elec = 0.5 * C * A0 * rho
    # EHN-exact posmass A₀ (Eq.12) for the electric energy — measurement cross-check
    A0p = _solve_A0_posmass(rho, a1, kv, C)
    e_elec_pm = 0.5 * C * A0p * rho
    # topological linking flux ∫ρ = ∫(∇a)·B → (2π)²·N_link if fully quantised
    link_flux = float(jnp.sum(rho) * dx**3)
    V = dx**3
    comp = {"grad1": float(jnp.sum(e_grad1) * V), "grad2": float(jnp.sum(e_grad2) * V),
            "mag": float(jnp.sum(e_mag) * V), "pot": float(jnp.sum(e_pot) * V),
            "elec": float(jnp.sum(e_elec) * V),
            "elec_pm": float(jnp.sum(e_elec_pm) * V), "link_flux": link_flux}
    comp["total"] = sum(comp[k] for k in ("grad1", "grad2", "mag", "pot", "elec"))
    return comp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=160)
    ap.add_argument("--L", type=float, default=12.0)
    ap.add_argument("--lam", type=float, default=1000.0)     # λ≫g² (EHN: 1e3)
    ap.add_argument("--kappa", type=float, default=0.0008)
    ap.add_argument("--C", type=float, default=400.0)
    ap.add_argument("--nlink", type=int, default=4)          # EHN min stable
    ap.add_argument("--R", type=float, default=2.5)
    ap.add_argument("--core", type=float, default=0.5)
    ap.add_argument("--dt", type=float, default=2e-4)
    ap.add_argument("--U", type=float, default=50.0)
    ap.add_argument("--etaw", type=float, default=0.2)
    ap.add_argument("--Mbar2", type=float, default=1.0)
    ap.add_argument("--eps-a", type=float, default=0.05)
    ap.add_argument("--prebuild", type=int, default=800)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--samples", type=int, default=20)
    ap.add_argument("--scoeff", type=float, default=1.0)     # scalar-grad coeff: 1.0=EHN, 0.5=legacy
    ap.add_argument("--ic-flux", default="prebuild", choices=["prebuild", "london"])
    ap.add_argument("--flux-scale", type=float, default=1.0)  # post-build A scaling (force flux to floor)
    ap.add_argument("--target-floor", type=float, default=0.0)  # >0: auto-scale A so link_flux = this*(2π)²N_link
    ap.add_argument("--cramp-tau", type=float, default=0.0)   # >0: ramp C 0→target linearly over this tau
    ap.add_argument("--pin-flux", action="store_true")        # hard-pin link_flux to the (2π)²N_link floor each step
    ap.add_argument("--out", default="out")
    a = ap.parse_args()
    t0 = time.time()
    kv = kvecs(a.N, a.L); KX, KY, KZ, K2 = kv; dx = a.L / a.N
    lp = jnp.exp(-0.5 * K2 * (1.5 * dx) ** 2)
    phi1, phi2 = build_ic(a.N, a.L, a.nlink, a.R, a.core)
    z = jnp.zeros((a.N, a.N, a.N)); Ax, Ay, Az = z, z, z
    Bx, By, Bz = z, z, z; wx, wy, wz = z, z, z
    Q0 = skyrmion_number(phi1, phi2, kv, dx)
    # build φ₁ flux: slow damped step-prebuild, or a converged direct London solve
    if a.ic_flux == "london":
        Ax, Ay, Az = relax_flux_london(phi1, kv, a.N, niter=max(400, a.prebuild))
    else:
        for _ in range(a.prebuild):
            _, _, Ax, Ay, Az, *_ = step(phi1, phi2, Ax, Ay, Az, z, z, z, z, z, z, lp,
                                        KX, KY, KZ, K2, a.dt, 0.0, a.lam, a.kappa, a.Mbar2, a.U, a.etaw, a.eps_a, a.scoeff)
    # optionally scale A to seed the topological floor (metastability test)
    floor0 = (2.0 * np.pi) ** 2 * a.nlink
    if a.target_floor > 0.0:
        Ein = two_scalar_energy(phi1, phi2, Ax, Ay, Az, kv, dx, a.lam, a.kappa, a.C, a.Mbar2, a.eps_a, lp)
        cur = Ein["link_flux"]
        if abs(cur) > 1e-9:
            a.flux_scale = a.target_floor * floor0 / cur
    Ax, Ay, Az = a.flux_scale * Ax, a.flux_scale * Ay, a.flux_scale * Az
    Bx, By, Bz = curl(Ax, Ay, Az, kv)
    E_ic = two_scalar_energy(phi1, phi2, Ax, Ay, Az, kv, dx, a.lam, a.kappa, a.C, a.Mbar2, a.eps_a, lp)
    floor_signed = floor0 if E_ic["link_flux"] >= 0 else -floor0   # match the IC handedness
    print(f"  IC[{a.ic_flux}] flux_scale={a.flux_scale:.3f}: E={E_ic['total']:.1f} "
          f"mag={E_ic['mag']:.1f} el={E_ic['elec']:.1f} "
          f"link={E_ic['link_flux']:.1f}/{floor0:.0f}={E_ic['link_flux']/floor0*100:.0f}% "
          f"pin_flux={a.pin_flux}", flush=True)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    traj = []
    every = max(1, a.steps // a.samples)
    p1, p2 = phi1, phi2
    cramp_steps = int(a.cramp_tau / a.dt) if a.cramp_tau > 0 else 0
    for s in range(a.steps + 1):
        # adiabatic C-ramp: hold flux at C=0, then grow C→target over cramp_tau so
        # the electric binding builds without the violent start-up transient.
        C_eff = a.C * min(1.0, s / cramp_steps) if cramp_steps > 0 else a.C
        if s % every == 0:
            Q = skyrmion_number(p1, p2, kv, dx)
            H = magnetic_helicity(Ax, Ay, Az, kv, dx)
            amp = float(jnp.max(jnp.abs(Ax) + jnp.abs(Ay) + jnp.abs(Az)))
            E = two_scalar_energy(p1, p2, Ax, Ay, Az, kv, dx, a.lam, a.kappa, C_eff, a.Mbar2, a.eps_a, lp)
            traj.append({"step": s, "tau": s * a.dt, "C_eff": C_eff, "Q": Q, "AdotB": H, "Aamp": amp,
                         "E": E["total"], "E_comp": E})
            floor = (2.0 * np.pi) ** 2 * a.nlink
            print(f"  tau={s*a.dt:6.3f} C={C_eff:5.0f} Q={Q:+.3f} Aamp={amp:.4f} E={E['total']:9.1f} "
                  f"(g1={E['grad1']:.0f} g2={E['grad2']:.0f} mag={E['mag']:.1f} "
                  f"pot={E['pot']:.0f} el={E['elec']:.1f} el_pm={E['elec_pm']:.1f}) "
                  f"link={E['link_flux']:.1f}/{floor:.0f}={E['link_flux']/floor*100:.0f}%", flush=True)
            if not np.isfinite(Q) or not np.isfinite(amp) or not np.isfinite(E["total"]):
                traj[-1]["BLEW"] = True
                break
            # incremental checkpoint so a long run is monitorable mid-flight
            (out / "manifest.json").write_text(json.dumps(
                {"params": vars(a), "Q0": Q0, "trajectory": traj,
                 "wall_s": time.time() - t0, "dx": dx, "in_progress": True}, indent=1))
        if s < a.steps:
            p1, p2, Ax, Ay, Az, Bx, By, Bz, wx, wy, wz = step(
                p1, p2, Ax, Ay, Az, Bx, By, Bz, wx, wy, wz, lp,
                KX, KY, KZ, K2, a.dt, C_eff, a.lam, a.kappa, a.Mbar2, a.U, a.etaw, a.eps_a, a.scoeff)
            if a.pin_flux:
                Ax, Ay, Az = pin_link_flux(Ax, Ay, Az, p2, KX, KY, KZ, K2, a.eps_a, lp,
                                           floor_signed, dx**3)
    Qf = traj[-1]["Q"]
    Ef = traj[-1].get("E", float("nan"))
    Ef_comp = traj[-1].get("E_comp", {})
    manifest = {"params": vars(a), "Q0": Q0, "Q_final": Qf,
                "E_final": Ef, "E_final_comp": Ef_comp,
                "held": bool(abs(Qf - round(Q0)) < 0.25 and np.isfinite(Qf)),
                "blew": bool(traj[-1].get("BLEW", False)),
                "trajectory": traj, "wall_s": time.time() - t0, "dx": dx}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    ehn_ref = {4: 6.0e3, 5: 7.0e3}.get(a.nlink)
    ref_str = f"  EHN(nl={a.nlink})≈{ehn_ref:.0f}" if ehn_ref else ""
    print(f"Q0={Q0:+.3f} Q_final={Qf:+.3f} held={manifest['held']} blew={manifest['blew']} "
          f"E_final={Ef:.1f}{ref_str} ({manifest['wall_s']:.0f}s)")
    if Ef_comp:
        print("  E components: " + " ".join(f"{k}={v:.1f}" for k, v in Ef_comp.items() if k != "total"))


if __name__ == "__main__":
    main()

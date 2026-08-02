#!/usr/bin/env python3
"""
Classical EM fields from substrate sources — a reusable field solver.

A particle's carrier carries a charge density ρ and a supercurrent density j;
this module solves Maxwell's static equations for the fields they radiate, on a
periodic grid via FFT–Poisson:

    electric:  ∇²φ = −ρ/ε₀,   E = −∇φ              (Coulomb / near field)
    magnetic:  ∇²A = −μ₀ j,   B = ∇×A              (Biot–Savart)

This is intended as a library feature (`nwt_substrate.em`, pending PR): besides
visualising the field around a particle, the same ρ, j → φ, A, E, B pipeline
gives **multipole moments** and **electromagnetic form factors**, which feed
elastic scattering — so it is reusable beyond portraiture.

Conventions: natural units ε₀ = μ₀ = 1.  The k=0 (uniform) Fourier mode is
dropped, so E is the field of (ρ − ⟨ρ⟩) on the periodic cell — correct near the
source (the only region we visualise); j is taken divergence-free (a closed
current), so A is well defined.

Source model for a carrier knot:
  * charge:  total Q smeared as a Gaussian shell on the knot tube;
  * current: I along the local tangent (a current loop following the knot),
             smeared on the same shell — the loop whose B is the magnetic moment.

Demo: `python3 simulations/nwt_em_fields.py [OUTPUT.png] [mode=E|B|both]`
"""
from __future__ import annotations

import sys
import numpy as np
from scipy.interpolate import RegularGridInterpolator

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import nwt_surface_current_portraits as scp  # carrier curves + tube render


# ==========================================================================
# Field solvers (the reusable core)
# ==========================================================================
def _k_grids(N, dx):
    k = 2 * np.pi * np.fft.fftfreq(N, d=dx)
    KX, KY, KZ = np.meshgrid(k, k, k, indexing="ij")
    K2 = KX**2 + KY**2 + KZ**2
    K2[0, 0, 0] = 1.0          # avoid /0; the k=0 mode is zeroed below
    return KX, KY, KZ, K2


def electric_field(rho, dx, eps0=1.0):
    """E from a charge density via ∇²φ = −ρ/ε₀, E = −∇φ.  Returns (Ex,Ey,Ez)."""
    KX, KY, KZ, K2 = _k_grids(rho.shape[0], dx)
    rho_k = np.fft.fftn(rho)
    rho_k[0, 0, 0] = 0.0
    phi_k = rho_k / (eps0 * K2)
    E = [np.real(np.fft.ifftn(-1j * K * phi_k)) for K in (KX, KY, KZ)]
    return E


def magnetic_field(j, dx, mu0=1.0):
    """B from a current density j=(jx,jy,jz) via ∇²A = −μ₀ j, B = ∇×A."""
    KX, KY, KZ, K2 = _k_grids(j[0].shape[0], dx)
    A = []
    for jc in j:
        jc_k = np.fft.fftn(jc); jc_k[0, 0, 0] = 0.0
        A.append(jc_k * mu0 / K2)             # Â = μ₀ ĵ / k²  (component-wise)
    Ax_k, Ay_k, Az_k = A
    Bx = np.real(np.fft.ifftn(1j * (KY * Az_k - KZ * Ay_k)))
    By = np.real(np.fft.ifftn(1j * (KZ * Ax_k - KX * Az_k)))
    Bz = np.real(np.fft.ifftn(1j * (KX * Ay_k - KY * Ax_k)))
    return [Bx, By, Bz]


def _divergence(F, dx):
    KX, KY, KZ, _ = _k_grids(F[0].shape[0], dx)
    return np.real(np.fft.ifftn(1j * (KX * np.fft.fftn(F[0])
                                      + KY * np.fft.fftn(F[1])
                                      + KZ * np.fft.fftn(F[2]))))


def _curl(F, dx):
    KX, KY, KZ, _ = _k_grids(F[0].shape[0], dx)
    Fx, Fy, Fz = (np.fft.fftn(c) for c in F)
    return [np.real(np.fft.ifftn(1j * (KY * Fz - KZ * Fy))),
            np.real(np.fft.ifftn(1j * (KZ * Fx - KX * Fz))),
            np.real(np.fft.ifftn(1j * (KX * Fy - KY * Fx)))]


def maxwell_eh(rho, j, dx, *, eh_xi=0.0, n_iter=6, eps0=1.0, mu0=1.0):
    """Static Maxwell fields with the optional Euler–Heisenberg nonlinear
    vacuum correction.

    The EH effective Lagrangian L_EH = ξ[(E²−B²)² + 7(E·B)²] makes the vacuum a
    nonlinear medium with polarization P = ∂L_EH/∂E and magnetization
    M = −∂L_EH/∂B, so Gauss/Ampère read ∇·(E+P)=ρ, ∇×(B−M)=j.  We solve this by
    fixed-point iteration: linear solve → vacuum P,M from (E,B) → resolve with
    ρ_eff = ρ−∇·P, j_eff = j+∇×M.

    eh_xi : EH coupling in solver units (physical value 2α²/45m⁴ is ~α³ at the
            Compton scale = invisible; boost it to probe the strong-field
            regime, as nwt_euler_heisenberg.py does).  eh_xi=0 → linear Maxwell.

    Returns (E, B), each a list [Fx, Fy, Fz].
    """
    E = electric_field(rho, dx, eps0)
    B = magnetic_field(j, dx, mu0)
    if eh_xi == 0.0:
        return E, B
    for _ in range(n_iter):
        E2 = sum(c * c for c in E); B2 = sum(c * c for c in B)
        EdotB = sum(e * b for e, b in zip(E, B)); inv = E2 - B2
        # P_i = 4ξ(E²−B²)E_i + 14ξ(E·B)B_i ;  M_i = 4ξ(E²−B²)B_i − 14ξ(E·B)E_i
        P = [4 * eh_xi * inv * E[i] + 14 * eh_xi * EdotB * B[i] for i in range(3)]
        M = [4 * eh_xi * inv * B[i] - 14 * eh_xi * EdotB * E[i] for i in range(3)]
        rho_eff = rho - _divergence(P, dx)
        cM = _curl(M, dx); j_eff = [j[i] + cM[i] for i in range(3)]
        E = electric_field(rho_eff, dx, eps0)
        B = magnetic_field(j_eff, dx, mu0)
    return E, B


# ==========================================================================
# Source deposition (charge / current of a carrier knot)
# ==========================================================================
def _grid(N, L):
    g = np.linspace(-L / 2, L / 2, N, endpoint=False)
    X, Y, Z = np.meshgrid(g, g, g, indexing="ij")
    return g, g[1] - g[0], X, Y, Z


def deposit_sources(curves, X, Y, Z, dx, *, Q=0.0, I=1.0, width=0.6):
    """Smear total charge Q and a tangential loop current of strength I onto a
    Gaussian shell around the carrier curve(s).  Returns (rho, (jx,jy,jz))."""
    rho = np.zeros_like(X)
    jx = np.zeros_like(X); jy = np.zeros_like(X); jz = np.zeros_like(X)
    for C in curves:
        T = np.gradient(C, axis=0)
        T /= np.linalg.norm(T, axis=1, keepdims=True) + 1e-12
        for (px, py, pz), (tx, ty, tz) in zip(C, T):
            g = np.exp(-(((X - px)**2 + (Y - py)**2 + (Z - pz)**2)) / (2 * width**2))
            rho += g
            jx += g * tx; jy += g * ty; jz += g * tz
    # normalise total charge to Q (cell integral); current carries I per unit
    tot = rho.sum() * dx**3
    if tot > 0 and Q != 0.0:
        rho *= Q / tot
    else:
        rho *= 0.0
    norm = (np.sqrt(jx**2 + jy**2 + jz**2).max() + 1e-12)
    jx, jy, jz = (I * c / norm for c in (jx, jy, jz))
    return rho, [jx, jy, jz]


# ==========================================================================
# Field-line tracing
# ==========================================================================
def _interp(field, g):
    return [RegularGridInterpolator((g, g, g), F, bounds_error=False, fill_value=0.0)
            for F in field]


def trace_field_lines(field, g, seeds, *, n_steps=400, ds=0.15, both_ways=True):
    """Integrate field lines (RK2) from seed points along the normalised field."""
    fx, fy, fz = _interp(field, g)
    lo, hi = g[0], g[-1]
    lines = []
    for seed in seeds:
        for direction in ((1, -1) if both_ways else (1,)):
            p = np.array(seed, float); pts = [p.copy()]
            for _ in range(n_steps):
                v = np.array([fx([p])[0], fy([p])[0], fz([p])[0]])
                n = np.linalg.norm(v)
                if n < 1e-9:
                    break
                vh = direction * v / n
                pm = p + 0.5 * ds * vh                       # RK2 midpoint
                vm = np.array([fx([pm])[0], fy([pm])[0], fz([pm])[0]])
                nm = np.linalg.norm(vm)
                if nm < 1e-9:
                    break
                p = p + ds * direction * vm / nm
                if not (lo < p[0] < hi and lo < p[1] < hi and lo < p[2] < hi):
                    break
                pts.append(p.copy())
            if len(pts) > 3:
                lines.append(np.array(pts))
    return lines


# ==========================================================================
# Render: carrier tube + E / B / both field lines
# ==========================================================================
def _view_dir(elev, azim):
    el, az = np.radians(elev), np.radians(azim)
    return np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])


def _lines_to_ribbons(lines, view, width, color):
    """Convert polylines to camera-facing ribbon quads (faces, facecolors) so
    they can be merged into the tube's Poly3DCollection and depth-sorted with
    it — the only way matplotlib z-interleaves lines with a surface."""
    quads = []
    for ln in lines:
        if len(ln) < 2:
            continue
        p0, p1 = ln[:-1], ln[1:]
        seg = p1 - p0
        wd = np.cross(seg, view)
        n = np.linalg.norm(wd, axis=1, keepdims=True)
        wd = np.where(n > 1e-9, wd / n, 0.0) * width
        quads.append(np.stack([p0 - wd, p0 + wd, p1 + wd, p1 - wd], axis=1))
    if not quads:
        return np.empty((0, 4, 3)), np.empty((0, 4))
    F = np.concatenate(quads, axis=0)
    return F, np.tile(np.asarray(color, float), (len(F), 1))


def render_particle_field(out, sector="lepton", *, charge=-1.0, mode="both",
                          N=96, L=24.0, m=3, q=1, aspect="idealized", title=None,
                          elev=22, azim=-56):
    geom = scp.ASPECTS[aspect]
    R, r, a = geom["R"], geom["r"], geom["a"]
    g, dx, X, Y, Z = _grid(N, L)
    curves = scp.carrier_curves(sector, R=R, r=r, n=240)
    rho, j = deposit_sources(curves, X, Y, Z, dx, Q=charge, I=1.0, width=0.6)
    from nwt_substrate import em as _em
    view = _view_dir(elev, azim)

    faces, colors = [], []
    # 1) carrier tube faces (surface-current coloured) — same builder as the portrait
    for C in scp.carrier_curves(sector, R=R, r=r, n=480):
        Xt, Yt, Zt, s, pol = scp.tube_surface(C, a=a)
        f, c = scp._tube_quads(Xt, Yt, Zt, 2 * np.pi * m * s + q * pol, "phase")
        faces.append(f); colors.append(c)
    # 2) E field lines (radial) as warm ribbons
    if mode in ("E", "both") and charge != 0.0:
        E = _em.electric_field(rho, dx, bc="open")
        th = np.linspace(0, 2 * np.pi, 13, endpoint=False); ph = np.linspace(0.3, np.pi - 0.3, 5)
        rs = R + r + a + 0.6
        seeds = [(rs*np.sin(p)*np.cos(t), rs*np.sin(p)*np.sin(t), rs*np.cos(p)) for t in th for p in ph]
        ln = trace_field_lines(E, g, seeds, n_steps=240, ds=0.22)
        f, c = _lines_to_ribbons(ln, view, 0.045, (1.0, 0.42, 0.24, 1.0))
        faces.append(f); colors.append(c)
    # 3) B field lines (loops) as cool ribbons — sparse so they don't bury the tube
    if mode in ("B", "both"):
        B = _em.magnetic_field(j, dx, bc="open")
        th = np.linspace(0, 2 * np.pi, 9, endpoint=False)
        seeds = [((R+r+0.8)*np.cos(t), (R+r+0.8)*np.sin(t), 0.0) for t in th]
        ln = trace_field_lines(B, g, seeds, n_steps=360, ds=0.18)
        f, c = _lines_to_ribbons(ln, view, 0.05, (0.22, 0.72, 1.0, 1.0))
        faces.append(f); colors.append(c)

    fig = plt.figure(figsize=(7.5, 7.5))
    ax = fig.add_subplot(111, projection="3d")
    pc = Poly3DCollection(np.concatenate(faces), facecolors=np.concatenate(colors),
                          linewidths=0, shade=False)
    ax.add_collection3d(pc)
    ax.set_xlim(-L/2, L/2); ax.set_ylim(-L/2, L/2); ax.set_zlim(-L/4, L/4)
    ax.set_box_aspect((1, 1, 0.5)); ax.view_init(elev=elev, azim=azim)
    scp._black_3d(ax)
    ax.set_title(title or f"{sector}  field  (mode={mode}, Q={charge:+g})",
                 color="#e8e8e8", fontsize=12)
    fig.patch.set_facecolor("black"); fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor="black", bbox_inches="tight")
    print("wrote", out)


def _precompute_modulated_sources(curves, XYZ, dx, m, width=0.6):
    """Deposit cosine- and sine-weighted current/charge along the carrier so a
    traveling surface-current wave cos(2π m·s + φ) can be assembled per frame as
    a linear combination — and (by Maxwell linearity) so can its field.

    Returns dict of: Jdc/Jc/Js (uniform, cos-, sin-weighted current vectors) and
    Rdc/Rc/Rs (the same for charge density)."""
    X, Y, Z = XYZ
    z = np.zeros_like(X)
    out = {k: ([z.copy(), z.copy(), z.copy()] if k.startswith("J") else z.copy())
           for k in ("Jdc", "Jc", "Js", "Rdc", "Rc", "Rs")}
    for C in curves:
        T = np.gradient(C, axis=0); T /= np.linalg.norm(T, axis=1, keepdims=True) + 1e-12
        M = len(C)
        for i in range(M):
            (px, py, pz), t = C[i], T[i]
            g = np.exp(-((X - px)**2 + (Y - py)**2 + (Z - pz)**2) / (2 * width**2))
            cph, sph = np.cos(2 * np.pi * m * i / M), np.sin(2 * np.pi * m * i / M)
            out["Rdc"] += g; out["Rc"] += g * cph; out["Rs"] += g * sph
            for d in range(3):
                out["Jdc"][d] += g * t[d]
                out["Jc"][d] += g * cph * t[d]
                out["Js"][d] += g * sph * t[d]
    return out


def make_field_animation(out, sector="lepton", *, m=3, q=1, mode="B",
                         n_frames=24, N=80, L=24.0, aspect="idealized",
                         charge=-1.0, I_dc=0.5, I_ac=1.0, rho_amp=0.7,
                         elev=22, azim=-56, duration=0.08, filmstrip=None):
    """Animate one toroidal cycle: the m-harmonic surface current travels around
    the knot and **drags its field with it**.  The source is the modulated
    current/charge cos(2π m·s + φ); φ sweeps 0→2π (seamless loop).  Maxwell
    linearity ⇒ solve the cos/sin field components once, combine per frame."""
    import imageio.v2 as imageio
    from nwt_substrate import em as _em
    geom = scp.ASPECTS[aspect]; R, r, a = geom["R"], geom["r"], geom["a"]
    g, dx, X, Y, Z = _grid(N, L)
    curves = scp.carrier_curves(sector, R=R, r=r, n=240)
    S = _precompute_modulated_sources(curves, (X, Y, Z), dx, m)
    view = _view_dir(elev, azim)

    jmax = np.sqrt(sum(c**2 for c in S["Jdc"])).max() + 1e-12
    Bf = Ef = None
    if mode in ("B", "both"):
        Bf = tuple(_em.magnetic_field([c / jmax for c in S[k]], dx, bc="open")
                   for k in ("Jdc", "Jc", "Js"))
    if mode in ("E", "both"):
        Rq = S["Rdc"] * (charge / (S["Rdc"].sum() * dx**3))
        Ef = (_em.electric_field(Rq, dx, bc="open"),
              _em.electric_field(S["Rc"], dx, bc="open"),
              _em.electric_field(S["Rs"], dx, bc="open"))

    frames = []
    for k in range(n_frames):
        phi = 2 * np.pi * k / n_frames
        cphi, sphi = np.cos(phi), np.sin(phi)
        faces, colors = [], []
        for C in scp.carrier_curves(sector, R=R, r=r, n=480):
            Xt, Yt, Zt, s, pol = scp.tube_surface(C, a=a)
            f, c = scp._tube_quads(Xt, Yt, Zt, 2 * np.pi * m * s + q * pol + phi, "phase")
            faces.append(f); colors.append(c)
        if Bf is not None:
            Bdc, Bc, Bs = Bf
            B = [I_dc * Bdc[d] + I_ac * (cphi * Bc[d] - sphi * Bs[d]) for d in range(3)]
            th = np.linspace(0, 2 * np.pi, 10, endpoint=False)
            seeds = [((R + r + 0.8) * np.cos(t), (R + r + 0.8) * np.sin(t), 0.0) for t in th]
            ln = trace_field_lines(B, g, seeds, n_steps=260, ds=0.18)
            f, c = _lines_to_ribbons(ln, view, 0.05, (0.22, 0.72, 1.0, 1.0))
            faces.append(f); colors.append(c)
        if Ef is not None:
            Eq, Ec, Es = Ef
            E = [Eq[d] + rho_amp * (cphi * Ec[d] - sphi * Es[d]) for d in range(3)]
            th = np.linspace(0, 2 * np.pi, 13, endpoint=False); ph = np.linspace(0.3, np.pi - 0.3, 5)
            rs = R + r + a + 0.6
            seeds = [(rs*np.sin(p)*np.cos(t), rs*np.sin(p)*np.sin(t), rs*np.cos(p)) for t in th for p in ph]
            ln = trace_field_lines(E, g, seeds, n_steps=200, ds=0.22)
            f, c = _lines_to_ribbons(ln, view, 0.045, (1.0, 0.42, 0.24, 1.0))
            faces.append(f); colors.append(c)
        fig = plt.figure(figsize=(5.5, 5.5)); ax = fig.add_subplot(111, projection="3d")
        ax.add_collection3d(Poly3DCollection(np.concatenate(faces),
                            facecolors=np.concatenate(colors), linewidths=0, shade=False))
        ax.set_xlim(-L/2, L/2); ax.set_ylim(-L/2, L/2); ax.set_zlim(-L/4, L/4)
        ax.set_box_aspect((1, 1, 0.5)); ax.view_init(elev=elev, azim=azim); scp._black_3d(ax)
        ax.set_title(f"{sector} — {mode}-field toroidal cycle (m={m})", color="#e8e8e8", fontsize=11)
        fig.patch.set_facecolor("black"); fig.tight_layout(pad=0.2); fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()); plt.close(fig)
        print(f"  frame {k+1}/{n_frames}")
    imageio.mimsave(out, frames, duration=duration, loop=0); print("wrote", out)
    if filmstrip:
        sel = list(range(0, n_frames, max(1, n_frames // 6)))[:6]
        fig, axes = plt.subplots(1, len(sel), figsize=(3 * len(sel), 3.2))
        for ax, i in zip(axes, sel):
            ax.imshow(frames[i]); ax.axis("off")
            ax.set_title(f"φ={i}/{n_frames}", color="#e8e8e8", fontsize=9)
        fig.patch.set_facecolor("black"); fig.tight_layout()
        fig.savefig(filmstrip, dpi=110, facecolor="black", bbox_inches="tight"); print("wrote", filmstrip)


def main():
    out, mode = "nwt_em_field.png", "both"
    for arg in sys.argv[1:]:
        if arg.startswith("mode="):
            mode = arg.split("=", 1)[1]
        else:
            out = arg
    render_particle_field(out, sector="lepton", charge=-1.0, mode=mode, m=3, q=1,
                          title=f"electron — EM field (mode={mode})")


if __name__ == "__main__":
    main()

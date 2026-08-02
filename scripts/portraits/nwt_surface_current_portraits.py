#!/usr/bin/env python3
"""
Surface-current particle portraits — the corpus-correct primitive (configurable).

In NWT the carrier is a *filled tube* whose electromagnetic current/field lives
on the tube SURFACE (Paper 8a solves the Helmholtz equation on the surface of
the vortex tube; Paper 5 shows the (2,1) core is *filled*).  A portrait is a
tube whose surface carries a standing current, NOT a glowing vortex line.

Quantum numbers (Paper-6 compendium (p, q, m, n_q)):
  * n_q  -> carrier TOPOLOGY (drawn knot): 0/1 unknot (leptons), 2 Hopf link
           (mesons), 3 trefoil (baryons), 5 cinquefoil (nucleons), ...
  * m    -> surface-current harmonic (wavelengths along the tube) = the
           "frequency" overtone; climbs with mass within a sector.
  * (p,q)-> torus winding (Pythagorean mass factor); here q = poloidal twist.

Configurable options
--------------------
aspect   : "physical"  -> slender tube at the corpus aspect ratio κ = R/r ≈ 9.84
                          (≈ π²); this κ is the same one Paper 8a turns into
                          α = 1/(√2 κ²), so the portrait's slenderness encodes
                          the fine-structure constant.
           "idealized" -> the crisp, slightly fatter display proportions.
coloring : "phase"     -> cyclic hue of the surface-current phase (traveling).
           "standing"  -> sequential brightness of the standing-wave intensity
                          ½(1+cos θ); nodes go dark, antinodes bright.
kelvin   : None, or (mode, amplitude) -> helical Kelvin-wave deformation of the
           carrier filament (a transverse standing wave with `mode` lobes), the
           geometric excitation of the vortex line.

All component faces go into ONE depth-sorted Poly3DCollection so linked tori
(Hopf) and knot self-crossings interleave correctly; the frame is
rotation-minimizing and closed (no poloidal seam).

Usage:
    python3 simulations/nwt_surface_current_portraits.py [OUTPUT.png] \\
            [aspect=physical|idealized] [coloring=phase|standing]
"""
from __future__ import annotations

import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from nwt_substrate.diagrams import torus_knot_curve

# Aspect presets: (R_major, R_minor, tube_radius).  "physical" sets the tube
# slenderness R_major/tube_radius to the corpus aspect ratio κ ≈ 9.84.
ASPECTS = {
    "idealized": dict(R=4.0, r=1.5, a=0.60),     # crisp display proportions
    "physical":  dict(R=9.84, r=3.0, a=1.00),    # R/a = κ ≈ 9.84  (encodes α)
}
KAPPA_PHYSICAL = 9.84


# --------------------------------------------------------------------------
# Rotation-minimizing, closed frame + tube surface
# --------------------------------------------------------------------------
def _rmf(C):
    S = len(C)
    T = np.gradient(C, axis=0)
    T /= np.linalg.norm(T, axis=1, keepdims=True) + 1e-12
    ref = np.array([0, 0, 1.0])
    if abs(T[0] @ ref) > 0.9:
        ref = np.array([1.0, 0, 0])
    M = [np.cross(T[0], ref) / (np.linalg.norm(np.cross(T[0], ref)) + 1e-12)]
    for i in range(1, S):
        v1 = C[i] - C[i - 1]; c1 = v1 @ v1 + 1e-12
        rL = M[-1] - (2 / c1) * (v1 @ M[-1]) * v1
        tL = T[i - 1] - (2 / c1) * (v1 @ T[i - 1]) * v1
        v2 = T[i] - tL; c2 = v2 @ v2 + 1e-12
        Mi = rL - (2 / c2) * (v2 @ rL) * v2
        M.append(Mi / (np.linalg.norm(Mi) + 1e-12))
    M = np.array(M); B = np.cross(T, M)
    # Close the framing (remove the RMF holonomy -> no poloidal seam).
    axis = T[0]
    delta = np.arctan2(np.dot(np.cross(M[0], M[-1]), axis), np.dot(M[0], M[-1]))
    tw = -delta * np.arange(S) / (S - 1)
    cs, sn = np.cos(tw)[:, None], np.sin(tw)[:, None]
    return T, M * cs + B * sn, -M * sn + B * cs


def _kelvin_deform(C, mode, amp):
    """Helical Kelvin-wave deformation: transverse standing wave, `mode` lobes."""
    T, M, B = _rmf(C)
    s = 2 * np.pi * np.arange(len(C)) / len(C)
    return C + amp * (np.cos(mode * s)[:, None] * M + np.sin(mode * s)[:, None] * B)


def tube_surface(C, a, npol=30):
    T, M, B = _rmf(C)
    phi = np.linspace(0, 2 * np.pi, npol)
    cphi, sphi = np.cos(phi), np.sin(phi)
    X = C[:, 0, None] + a * (M[:, 0, None] * cphi + B[:, 0, None] * sphi)
    Y = C[:, 1, None] + a * (M[:, 1, None] * cphi + B[:, 1, None] * sphi)
    Z = C[:, 2, None] + a * (M[:, 2, None] * cphi + B[:, 2, None] * sphi)
    s = np.linspace(0, 1, len(C))[:, None] + 0 * phi
    pol = phi[None, :] + 0 * s
    return X, Y, Z, s, pol


# --------------------------------------------------------------------------
# Carrier topologies
# --------------------------------------------------------------------------
def carrier_curves(sector, R, r, n=480):
    if sector == "lepton":
        return [torus_knot_curve(2, 1, R=R, r=r, n_points=n)]
    if sector == "meson":
        t = np.linspace(0, 2 * np.pi, n, endpoint=False)
        A = np.stack([R * np.cos(t), R * np.sin(t), 0 * t], 1)
        Bb = np.stack([R * np.cos(t) + R, 0 * t, R * np.sin(t)], 1)
        return [A, Bb]
    if sector == "baryon":
        return [torus_knot_curve(2, 3, R=R, r=r, n_points=n)]
    if sector == "nucleon":
        return [torus_knot_curve(2, 5, R=R, r=r, n_points=n)]
    raise ValueError(sector)


def _black_3d(ax):
    ax.set_facecolor("black")
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color((0, 0, 0, 0)); axis.line.set_color((0, 0, 0, 0))
    ax.grid(False); ax.set_axis_off()


def _facecolors(theta, coloring):
    if coloring == "phase":
        return matplotlib.colormaps["hsv"]((theta % (2 * np.pi)) / (2 * np.pi))
    if coloring == "standing":               # standing-wave intensity, nodes dark
        return matplotlib.colormaps["inferno"](0.5 * (1 + np.cos(theta)))
    raise ValueError(coloring)


def _tube_quads(X, Y, Z, theta, coloring):
    V = np.stack([X, Y, Z], -1); S = V.shape[0]
    ip = (np.arange(S) + 1) % S
    faces = np.stack([V[:, :-1], V[ip][:, :-1], V[ip][:, 1:], V[:, 1:]],
                     axis=2).reshape(-1, 4, 3)
    cols = _facecolors(theta[:, :-1], coloring).reshape(-1, 4)
    return faces, cols


def surface_current_portrait(ax, sector, m, q=1, *, aspect="idealized",
                             coloring="phase", kelvin=None, phase_offset=0.0):
    """Draw a particle's carrier tube with its surface current.

    aspect       : "physical" (κ≈9.84, slender) or "idealized" (crisp).
    coloring     : "phase" (cyclic hue) or "standing" (magnitude, nodes dark).
    kelvin       : None or (mode, amplitude) for a helical filament excitation.
    phase_offset : advance the surface-current phase (animate the toroidal
                   cycle); a full 0→2π sweep is one period and loops seamlessly.
    """
    geom = ASPECTS[aspect]
    R, r, a = geom["R"], geom["r"], geom["a"]
    all_faces, all_cols = [], []
    for C in carrier_curves(sector, R=R, r=r):
        if kelvin is not None:
            C = _kelvin_deform(C, kelvin[0], kelvin[1])
        X, Y, Z, s, pol = tube_surface(C, a=a)
        theta = 2 * np.pi * m * s + q * pol + phase_offset
        f, c = _tube_quads(X, Y, Z, theta, coloring)
        all_faces.append(f); all_cols.append(c)
    pc = Poly3DCollection(np.concatenate(all_faces),
                          facecolors=np.concatenate(all_cols),
                          linewidths=0, shade=False)
    ax.add_collection3d(pc)
    lim = R + r + a
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim * 0.5, lim * 0.5)
    ax.set_box_aspect((1, 1, 0.55)); ax.view_init(elev=34, azim=-56)
    _black_3d(ax)


# --------------------------------------------------------------------------
# Named particles (Paper-6 compendium quantum numbers + predicted mass / MeV).
# --------------------------------------------------------------------------
PARTICLES = {
    "lepton (unknot)": [("e⁻", 2, 1, 3, 0.511), ("μ⁻", 1, 8, 9, 103.6)],
    "meson (Hopf link)": [("π⁺", 3, 5, 5, 143.3), ("K⁺", 2, 5, 8, 514.8),
                          ("ρ", 5, 7, 7, 797.2), ("J/ψ", 2, 7, 7, 3122.9)],
    "baryon (trefoil 3₁)": [("Λ", 3, 4, 12, 1123.4), ("Σ*", 3, 4, 14, 1385.0),
                            ("Ω⁻", 7, 4, 19, 1665.6), ("τ⁻", 3, 4, 17, 1789.8)],
    "nucleon (cinquefoil 5₁)": [("p", 1, 3, 5, 937.2), ("Σ⁺", 1, 3, 6, 1190.0)],
}
SECTOR_KEY = {"lepton (unknot)": "lepton", "meson (Hopf link)": "meson",
              "baryon (trefoil 3₁)": "baryon", "nucleon (cinquefoil 5₁)": "nucleon"}


def render_named_gallery(out, aspect="idealized", coloring="phase", kelvin=None, dpi=120):
    ncol = max(len(v) for v in PARTICLES.values()); nrow = len(PARTICLES)
    fig = plt.figure(figsize=(4.2 * ncol, 4.2 * nrow))
    for r, (label, plist) in enumerate(PARTICLES.items()):
        sec = SECTOR_KEY[label]
        for c in range(ncol):
            ax = fig.add_subplot(nrow, ncol, r * ncol + c + 1, projection="3d")
            if c < len(plist):
                name, p, q, m, mass = plist[c]
                surface_current_portrait(ax, sec, m=m, q=q, aspect=aspect,
                                         coloring=coloring, kelvin=kelvin)
                ax.set_title(f"{name}   (p,q,m)=({p},{q},{m})\n{mass} MeV   [{label}]",
                             color="#e8e8e8", fontsize=9.5, pad=2)
            else:
                _black_3d(ax)
    fig.patch.set_facecolor("black"); fig.tight_layout()
    fig.savefig(out, dpi=dpi, facecolor="black", bbox_inches="tight")
    print("wrote", out)


def make_particle_gif(out, sector, m, q=1, *, n_frames=36, duration=0.06,
                      title=None, **portrait_kwargs):
    """Animate one full toroidal cycle (surface current streaming around the
    loop) as a seamless GIF.  Extra kwargs (aspect/coloring/kelvin) pass through
    to surface_current_portrait."""
    import imageio.v2 as imageio
    frames = []
    for i in range(n_frames):
        fig = plt.figure(figsize=(5, 5)); ax = fig.add_subplot(111, projection="3d")
        surface_current_portrait(ax, sector, m=m, q=q,
                                 phase_offset=2 * np.pi * i / n_frames,
                                 **portrait_kwargs)
        if title:
            ax.set_title(title, color="#e8e8e8", fontsize=11)
        fig.patch.set_facecolor("black"); fig.tight_layout(pad=0.2)
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
        plt.close(fig)
    imageio.mimsave(out, frames, duration=duration, loop=0)
    print("wrote", out)


def render_all_gifs(out_dir, **portrait_kwargs):
    """One seamless toroidal-cycle GIF per named particle (at its own m)."""
    import os
    for label, plist in PARTICLES.items():
        sec = SECTOR_KEY[label]
        for name, p, q, m, mass in plist:
            safe = (name.replace("⁻", "-").replace("⁺", "+").replace("*", "star")
                        .replace("/", "_").replace("ψ", "psi").replace("τ", "tau")
                        .replace("Λ", "Lambda").replace("Σ", "Sigma")
                        .replace("Ω", "Omega").replace("ρ", "rho").replace("μ", "mu")
                        .replace("π", "pi").replace("η", "eta"))
            make_particle_gif(os.path.join(out_dir, f"cycle_{safe}.gif"),
                              sec, m=m, q=q, title=f"{name}  (m={m})",
                              **portrait_kwargs)


def main():
    out = "nwt_surface_current_portraits.png"
    aspect, coloring = "idealized", "phase"
    for arg in sys.argv[1:]:
        if arg.startswith("aspect="): aspect = arg.split("=", 1)[1]
        elif arg.startswith("coloring="): coloring = arg.split("=", 1)[1]
        else: out = arg
    render_named_gallery(out, aspect=aspect, coloring=coloring)


if __name__ == "__main__":
    main()

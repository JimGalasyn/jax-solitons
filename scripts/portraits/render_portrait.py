"""Multi-view field portraits for ehn_relax solitons (field.npz = u[10], s=A₀, w).

Four ways to look at the SAME relaxed field, so no single view can hide a cheat:

  raw    — marching-cubes isosurface of |φ₁| (magenta) + |φ₂| (cyan), NO smoothing.
           The grid facets are LEFT IN on purpose: they make the lattice visible so
           it's clear the knot is a real field on a real grid, not a drawn-in curve.
  twist  — φ₁ core-line (spline-smoothed) swept as a clean rotation-minimizing-frame
           tube, painted by the field phase arg(φ₁). The RMF carries NO twist of its
           own, so any spiral of colour along the loop is the REAL framing twist
           (Călugăreanu Tw). Faint φ₂ ring kept in: ring ⟹ neutron, no ring ⟹ proton.
  efield — E = −∇A₀ : field lines traced from a seed shell, warm ribbons; faint knot.
  bfield — B = ∇×A  : field lines threading the flux loops, cool ribbons; faint knot.

Deps (nwt_em_fields, nwt_surface_current_portraits, gpe_vortex_topology) live in the
parent simulations/ dir; this script lives with the ehn fields in engine_dogfood/.

  python render_portrait.py --field out_trefoil_twist1_n192 --view all
"""
import argparse
import json
import os
import sys

import numpy as np
from skimage import measure
from scipy.interpolate import RegularGridInterpolator, splprep, splev
from scipy.ndimage import gaussian_filter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)                          # the two sibling viz modules
from nwt_em_fields import (_curl, electric_field, trace_field_lines,             # noqa: E402
                           _lines_to_ribbons, _view_dir)
from nwt_surface_current_portraits import tube_surface, _black_3d, _rmf            # noqa: E402

# Was `from gpe_vortex_topology import ...`, a loose module in the source repo's
# simulations/ dir. That tracer is now part of this package and supplies all three
# symbols, so the dependency is a real import rather than a path hack.
from jax_solitons.vortex_topology import (                                       # noqa: E402
    _label_lines, _order_line, vortex_skeleton)


# ---------------------------------------------------------------- load + helpers
def load_field(fdir):
    d = np.load(os.path.join(fdir, "field.npz"))
    u = d["u"]; s = np.asarray(d["s"]); N = u.shape[1]
    meta = json.load(open(os.path.join(fdir, "manifest.json")))
    L = meta["params"]["L"]; dx = L / N
    return dict(p1=u[0] + 1j * u[1], p2=u[2] + 1j * u[3],
                A=[u[4], u[5], u[6]], s=s, N=N, L=L, dx=dx, meta=meta)


def _phys(P, N, dx):
    """cell-index coords → physical coords (box centred on 0)."""
    return (np.asarray(P) - N / 2.0) * dx


def _grid1d(N, L):
    return np.linspace(-L / 2, L / 2, N, endpoint=False)


def _smooth_closed(curve, npts=320):
    """Fit a periodic cubic spline to a noisy closed core-line and resample it
    smoothly, so the swept tube (and its RMF) aren't grid-stair-stepped."""
    c = np.asarray(curve, float)
    keep = np.r_[True, np.abs(np.diff(c, axis=0)).sum(1) > 1e-6]
    c = c[keep]
    if len(c) < 8:
        return curve
    try:
        tck, _ = splprep([c[:, 0], c[:, 1], c[:, 2]], s=len(c) * 0.5, per=1, k=3)
        uu = np.linspace(0, 1, npts, endpoint=False)
        return np.stack(splev(uu, tck), axis=1)
    except Exception:
        return curve


def core_curve(psi, N, dx, min_seg=20, smooth=True):
    """Largest |ψ|=0 vortex core, ordered into a physical-coord polyline."""
    P, T, C = vortex_skeleton(psi)
    if not len(P):
        return None
    lab, big, sizes = _label_lines(C, psi.shape, min_seg)
    if not big:
        return None
    m = lab == big[0]
    curve = _phys(P[m][_order_line(P[m], T[m])], N, dx)
    return _smooth_closed(curve) if smooth else curve


def _iso(scalar, level, N, dx, sigma=0.0):
    sc = gaussian_filter(scalar, sigma) if sigma > 0 else scalar
    if not (sc.min() < level < sc.max()):
        return None, None
    v, f, _, _ = measure.marching_cubes(sc, level=level)
    return _phys(v, N, dx), f


def _shade(faces, rgba, light=(0.35, 0.5, 0.9)):
    """Bake two-sided Lambert shading into per-face colours, so isosurfaces still
    read as 3D when drawn in ONE flat (shade=False) collection — the only way
    matplotlib depth-sorts interpenetrating tubes so a link actually weaves."""
    e1 = faces[:, 1] - faces[:, 0]; e2 = faces[:, 2] - faces[:, 0]
    n = np.cross(e1, e2); n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    lt = np.asarray(light, float); lt /= np.linalg.norm(lt)
    shade = 0.32 + 0.68 * np.abs(n @ lt)
    cols = np.ones((len(faces), 4))
    cols[:, :3] = np.asarray(rgba[:3])[None, :] * shade[:, None]
    cols[:, 3] = rgba[3] if len(rgba) > 3 else 1.0
    return cols


def _iso_parts(scalar, level, N, dx, rgba, sigma=0.0):
    """Isosurface as (list-of-triangles, per-face colours), or None."""
    v, f = _iso(scalar, level, N, dx, sigma)
    if v is None:
        return None
    faces = v[f]
    return list(faces), _shade(faces, rgba)


def _sample_phase(psi, N, L, pts):
    """arg ψ at physical points, interpolating Re/Im (no branch-cut wrap)."""
    g = _grid1d(N, L)
    re = RegularGridInterpolator((g, g, g), psi.real, bounds_error=False, fill_value=0.0)
    im = RegularGridInterpolator((g, g, g), psi.imag, bounds_error=False, fill_value=0.0)
    return np.arctan2(im(pts), re(pts))


def _tube_geom(X, Y, Z):
    """Closed-tube quad faces in (S*npol,) ravel order (i outer, j inner)."""
    V = np.stack([X, Y, Z], -1)
    Vi1 = np.roll(V, -1, 0); Vj1 = np.roll(V, -1, 1); Vi1j1 = np.roll(Vi1, -1, 1)
    return list(np.stack([V, Vi1, Vi1j1, Vj1], axis=2).reshape(-1, 4, 3))


def _quad_phase_colors(phase01, cmap, alpha=1.0):
    """Per-quad colours from a per-vertex cyclic phase (S,npol)∈[0,1); averaged on
    the circle so the 2π branch-cut seam doesn't flicker."""
    z = np.exp(2j * np.pi * phase01)
    zq = 0.25 * (z + np.roll(z, -1, 0) + np.roll(np.roll(z, -1, 0), -1, 1) + np.roll(z, -1, 1))
    cols = cmap((np.angle(zq).reshape(-1) / (2 * np.pi)) % 1.0)
    cols[:, 3] = alpha
    return cols


def _tube_phase01(core, F, rad_cells, npol):
    """Tube geometry + base phase (S,npol)∈[0,1) sampled once from arg(φ₁)."""
    X, Y, Z, _, _ = tube_surface(core, a=rad_cells * F["dx"], npol=npol)
    p01 = (_sample_phase(F["p1"], F["N"], F["L"],
                         np.stack([X.ravel(), Y.ravel(), Z.ravel()], -1)).reshape(X.shape)
           + np.pi) / (2 * np.pi)
    return _tube_geom(X, Y, Z), p01


def _phase_tube_parts(core, F, rad_cells, npol, alpha=1.0, cmap=cm.twilight_shifted):
    """φ₁ core swept as a closed RMF tube, per-quad coloured by arg(φ₁). RMF ⟹ any
    colour spiral is REAL framing twist. Returns (list-of-quads, colours)."""
    faces, p01 = _tube_phase01(core, F, rad_cells, npol)
    return faces, _quad_phase_colors(p01, cmap, alpha)


def _poloidal_seeds(core, dx, roff, n_along=10, n_pol=1):
    """Seed points just OUTSIDE the flux tube, in the local poloidal plane (RMF
    normals) at n_along stations along the core — B loops poloidally around the
    tube, so lines traced from here close into the ring loops of a flux tube."""
    if core is None:
        return []
    _, M, B = _rmf(core)
    idx = np.linspace(0, len(core) - 1, n_along, endpoint=False).astype(int)
    return [core[i] + roff * dx * (np.cos(2 * np.pi * k / n_pol) * M[i]
                                   + np.sin(2 * np.pi * k / n_pol) * B[i])
            for i in idx for k in range(n_pol)]


def _knot_box(core, L, pad=1.3):
    if core is None:
        return np.zeros(3), L / 4
    c = core.mean(0)
    return c, np.abs(core - c).max() * pad


def _fit(ax, center, half):
    for setlim, c in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), center):
        setlim(c - half, c + half)
    ax.set_box_aspect((1, 1, 1))


# ---------------------------------------------------------------- the four views
# Each view returns (parts, caption) where parts = [(polys, colours), ...]; render()
# MERGES them into ONE Poly3DCollection so matplotlib depth-sorts face-by-face and
# interpenetrating tubes actually weave (separate collections can't interleave).
def view_raw(F):
    parts = [p for p in (_iso_parts(np.abs(F["p1"]), 0.5, F["N"], F["dx"], (0.82, 0.13, 0.55)),
                         _iso_parts(np.abs(F["p2"]), 0.5, F["N"], F["dx"], (0.10, 0.74, 0.80)))
             if p is not None]
    return parts, "raw |φ₁|(magenta)+|φ₂|(cyan), merged mesh so the −3 link weaves; grid facets left in (no cheat)"


def view_twist(F, npol=30, rad_cells=2.2):
    core = core_curve(F["p1"], F["N"], F["dx"])
    if core is None:
        return [], "twist: no φ₁ core found"
    parts = []
    ring = _iso_parts(np.abs(F["p2"]), 0.5, F["N"], F["dx"], (0.10, 0.74, 0.80, 0.32))
    if ring is not None:
        parts.append(ring)
    parts.append(_phase_tube_parts(core, F, rad_cells, npol))
    return parts, "twist: φ₁ core tube painted by arg(φ₁) on an RMF frame — colour spiral = real Tw; faint φ₂ ring ⟹ neutron"


def _carrier_parts(F, core):
    return [_phase_tube_parts(core, F, 2.2, 26, alpha=1.0, cmap=cm.hsv)] if core is not None else []


def _field_line_parts(F, kind, core, c, half, elev, azim):
    """Just the E or B field-line ribbons (no carrier) + an n-lines label."""
    N, L, dx = F["N"], F["L"], F["dx"]; g = _grid1d(N, L)
    if kind == "E":
        gs = np.gradient(gaussian_filter(F["s"], 1.0), dx)     # light smooth → clean spokes
        fld = [-gs[0], -gs[1], -gs[2]]; col = (1.0, 0.42, 0.24, 1.0)
        th = np.linspace(0, 2 * np.pi, 18, endpoint=False); ph = np.linspace(0.28, np.pi - 0.28, 6)
        rs = half * 1.05
        seeds = [c + rs * np.array([np.sin(p) * np.cos(t), np.sin(p) * np.sin(t), np.cos(p)])
                 for t in th for p in ph]
        lines = trace_field_lines(fld, g, seeds, n_steps=int(2.6 * half / dx), ds=0.7 * dx,
                                  both_ways=False)
        width = 0.008 * half
    else:
        fld = [gaussian_filter(b, 1.0) for b in _curl(F["A"], dx)]
        col = (0.22, 0.72, 1.0, 1.0)
        seeds = _poloidal_seeds(core, dx, roff=3.2, n_along=12)
        lines = trace_field_lines(fld, g, seeds, n_steps=500, ds=0.45 * dx, both_ways=True)
        width = 0.010 * half
    faces, cols = _lines_to_ribbons(lines, _view_dir(elev, azim), width, col)
    return ([(list(faces), cols)] if len(faces) else []), len(lines)


def view_field(F, kind, elev, azim):
    core = core_curve(F["p1"], F["N"], F["dx"]); c, half = _knot_box(core, F["L"])
    lp, n = _field_line_parts(F, kind, core, c, half, elev, azim)
    tag = "E = −∇A₀ (radial spokes)" if kind == "E" else "B = ∇×A (poloidal loops)"
    note = "" if kind == "E" else "  (φ₂ ring global/ungauged ⟹ no B — loops thread φ₁ only)"
    return _carrier_parts(F, core) + lp, f"{tag} — {n} lines{note}"


def _add_parts(ax, parts):
    """Merge parts into ONE Poly3DCollection so matplotlib depth-sorts face-by-face."""
    if not parts:
        return
    polys = [p for part in parts for p in part[0]]
    cols = np.concatenate([part[1] for part in parts])
    ax.add_collection3d(Poly3DCollection(polys, facecolors=cols, linewidths=0, shade=False))


# ---------------------------------------------------------------- driver
def _hdr(fdir, F):
    nk = F["meta"]["params"].get("nlink"); lk = F["meta"].get("cross_lk")
    return f"{os.path.basename(fdir)}   nlink={nk}  Lk(φ₁,φ₂)={lk}"


def render(fdir, view, out, elev, azim, dpi):
    F = load_field(fdir)
    center, half = _knot_box(core_curve(F["p1"], F["N"], F["dx"]), F["L"])
    build = {"raw": lambda: view_raw(F), "twist": lambda: view_twist(F),
             "efield": lambda: view_field(F, "E", elev, azim),
             "bfield": lambda: view_field(F, "B", elev, azim)}
    parts, sub = build[view]()
    fig = plt.figure(figsize=(7.5, 7.5))
    ax = fig.add_subplot(111, projection="3d")
    _add_parts(ax, parts)
    _fit(ax, center, half); ax.view_init(elev=elev, azim=azim); _black_3d(ax)
    ax.set_title(f"{_hdr(fdir, F)}  —  {view}\n{sub}", color="#e8e8e8", fontsize=10)
    fig.patch.set_facecolor("black"); fig.tight_layout()
    fig.savefig(out, dpi=dpi, facecolor="black", bbox_inches="tight"); plt.close(fig)
    print(f"wrote {out}")


def render_triptych(fdir, out, elev, azim, dpi):
    """E | B | E+B in one figure — the alpha-portrait composition, on the real field."""
    F = load_field(fdir)
    core = core_curve(F["p1"], F["N"], F["dx"]); center, half = _knot_box(core, F["L"])
    carrier = _carrier_parts(F, core)
    Ep, nE = _field_line_parts(F, "E", core, center, half, elev, azim)
    Bp, nB = _field_line_parts(F, "B", core, center, half, elev, azim)
    panels = [(f"E field  (radial spokes ×{nE})", carrier + Ep),
              (f"B field  (poloidal loops ×{nB}, φ₂ ring carries none)", carrier + Bp),
              ("composed  E + B", carrier + Ep + Bp)]
    fig = plt.figure(figsize=(16.5, 6.2))
    for i, (title, parts) in enumerate(panels):
        ax = fig.add_subplot(1, 3, i + 1, projection="3d")
        _add_parts(ax, parts)
        _fit(ax, center, half); ax.view_init(elev=elev, azim=azim); _black_3d(ax)
        ax.set_title(title, color="#e8e8e8", fontsize=11, pad=2)
    fig.suptitle(_hdr(fdir, F) + "   —   E | B | composed", color="#e8e8e8", fontsize=13, y=0.97)
    fig.patch.set_facecolor("black"); fig.tight_layout()
    fig.savefig(out, dpi=dpi, facecolor="black", bbox_inches="tight"); plt.close(fig)
    print(f"wrote {out}")


def render_charge(fdir, out, elev, azim, dpi, zoom=1.3):
    """Charge-structure portrait: WHY the neutron is neutral but the proton isn't.
    Framing-charge model = baryon-core m(=1−|φ₁|²) ± framing-skin t(=|φ₁|²(1−|φ₁|²)),
    normalised so the neutron nets zero. Core/skin as SEPARATE isosurfaces (trefoil
    core always visible): core = red (+), skin translucent — blue (−) neutron / red
    (+) proton. Uses the OPEN-BC Poisson solver so a net charge is represented
    honestly: neutron = dipole (loops CLOSE near the object), proton = monopole
    (lines ESCAPE). zoom>1 pulls the camera + trace out to show the far field."""
    F = load_field(fdir); N, L, dx = F["N"], F["L"], F["dx"]; g = _grid1d(N, L)
    center, half = _knot_box(core_curve(F["p1"], N, dx), L); view = _view_dir(elev, azim)
    try:                                          # open BC keeps the proton's k=0 monopole
        from nwt_substrate import em as _sub_em
        efield = lambda rho: _sub_em.electric_field(rho, dx, bc="open"); bc = "open-BC"
    except Exception:
        efield = lambda rho: electric_field(rho, dx); bc = "periodic-BC"
    p2 = np.abs(F["p1"]) ** 2
    m = np.clip(1.0 - p2, 0, None)               # +baryon core (thin trefoil, |φ₁|→0)
    t = np.clip(p2 * (1.0 - p2), 0, None)        # framing skin shell (|φ₁|²≈½, larger radius)
    mn = m / (m.sum() + 1e-12); tn = t / (t.sum() + 1e-12)
    panels = [("neutron  (m−t):  +core / −skin  →  Q≈0  →  DIPOLE, loops close", -1),
              ("proton  (m+t):  +core & +skin  →  Q>0  →  MONOPOLE, lines escape", +1)]
    span = half * zoom
    fig = plt.figure(figsize=(14.0, 6.8))
    for i, (title, sgn) in enumerate(panels):
        ax = fig.add_subplot(1, 2, i + 1, projection="3d")
        parts = []
        core = _iso_parts(m, 0.7 * float(m.max()), N, dx, (0.93, 0.20, 0.20))
        if core is not None:
            parts.append(core)
        skin_rgba = (0.93, 0.20, 0.20, 0.34) if sgn > 0 else (0.20, 0.44, 0.96, 0.34)
        skin = _iso_parts(t, 0.5 * float(t.max()), N, dx, skin_rgba)
        if skin is not None:
            parts.append(skin)
        E = efield(mn + sgn * tn)
        th = np.linspace(0, 2 * np.pi, 16, endpoint=False); ph = np.linspace(0.3, np.pi - 0.3, 5)
        rs = half * 1.05
        seeds = [center + rs * np.array([np.sin(pp) * np.cos(a), np.sin(pp) * np.sin(a), np.cos(pp)])
                 for a in th for pp in ph]
        lines = trace_field_lines(E, g, seeds, n_steps=int(3.0 * span / dx), ds=0.7 * dx,
                                  both_ways=True)
        ef, ec = _lines_to_ribbons(lines, view, 0.006 * span, (1.0, 0.62, 0.22, 0.95))
        if len(ef):
            parts.append((list(ef), ec))
        _add_parts(ax, parts); _fit(ax, center, span)
        ax.view_init(elev=elev, azim=azim); _black_3d(ax)
        ax.set_title(title, color="#e8e8e8", fontsize=9.5, pad=2)
    fig.suptitle(_hdr(fdir, F) + f"   —   framing charge (m ± t),  {bc},  zoom×{zoom:g}",
                 color="#e8e8e8", fontsize=12, y=0.97)
    fig.patch.set_facecolor("black"); fig.tight_layout()
    fig.savefig(out, dpi=dpi, facecolor="black", bbox_inches="tight"); plt.close(fig)
    print(f"wrote {out}")


def render_cycle(fdir, out, n_frames, n_cycles, elev, azim, dpi):
    """Phase-cycle GIF. The leading time-dependence of a static soliton is the
    internal rotation Φ₁→e^{-iθ}Φ₁; because arg(φ₁) winds SPATIALLY along the loop,
    a uniform θ-sweep slides the colour bands AROUND the loop — the circulation
    you'd see running it forward, imposed cleanly (no dispersion from evolving the
    ungauged φ₁ alone). Tube geometry + base phase sampled once; only recoloured."""
    import imageio.v2 as imageio
    F = load_field(fdir)
    core = core_curve(F["p1"], F["N"], F["dx"]); center, half = _knot_box(core, F["L"])
    faces, base = _tube_phase01(core, F, 2.2, 30)             # sample the N³ phase ONCE
    ring = _iso_parts(np.abs(F["p2"]), 0.5, F["N"], F["dx"], (0.10, 0.74, 0.80, 0.30))
    hdr = _hdr(fdir, F); frames = []
    for k in range(n_frames):
        p01 = (base - n_cycles * k / n_frames) % 1.0         # uniform phase advance
        parts = ([ring] if ring is not None else []) + \
                [(faces, _quad_phase_colors(p01, cm.twilight_shifted))]
        fig = plt.figure(figsize=(6, 6)); ax = fig.add_subplot(111, projection="3d")
        _add_parts(ax, parts); _fit(ax, center, half)
        ax.view_init(elev=elev, azim=azim); _black_3d(ax)
        ax.set_title(f"{hdr}\nphase cycle — {n_cycles}× around the loop  (Φ₁→e^-iωt Φ₁)",
                     color="#e8e8e8", fontsize=8)
        fig.patch.set_facecolor("black"); fig.tight_layout(); fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()); plt.close(fig)
        if (k + 1) % 20 == 0:
            print(f"  frame {k + 1}/{n_frames}", flush=True)
    imageio.mimsave(out, frames, duration=0.05, loop=0)
    print(f"wrote {out}  ({n_frames} frames, {n_cycles} cycles)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--field", required=True, help="dir with field.npz + manifest.json")
    ap.add_argument("--view", default="all",
                    choices=["raw", "twist", "efield", "bfield", "triptych", "charge", "cycle", "all"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--elev", type=float, default=22)
    ap.add_argument("--azim", type=float, default=-56)
    ap.add_argument("--dpi", type=int, default=140)
    ap.add_argument("--frames", type=int, default=90, help="cycle GIF: number of frames")
    ap.add_argument("--cycles", type=int, default=3, help="cycle GIF: phase turns around the loop")
    ap.add_argument("--zoom", type=float, default=1.3, help="charge view: camera+trace span (×knot); >1 = far field")
    a = ap.parse_args()
    for v in (["raw", "twist", "efield", "bfield", "triptych"] if a.view == "all" else [a.view]):
        if v == "cycle":
            render_cycle(a.field, a.out or os.path.join(a.field, "portrait_cycle.gif"),
                         a.frames, a.cycles, a.elev, a.azim, a.dpi)
        elif v == "charge":
            render_charge(a.field, a.out or os.path.join(a.field, "portrait_charge.png"),
                          a.elev, a.azim, a.dpi, zoom=a.zoom)
        elif v == "triptych":
            render_triptych(a.field, a.out or os.path.join(a.field, "portrait_triptych.png"),
                            a.elev, a.azim, a.dpi)
        else:
            render(a.field, v, a.out or os.path.join(a.field, f"portrait_{v}.png"),
                   a.elev, a.azim, a.dpi)


if __name__ == "__main__":
    main()

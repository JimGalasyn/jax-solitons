"""Core-curve knot identification: what KNOT is a soliton's core curve?

The inverse of the carrier-knot ladder: given a (dynamically produced or seeded)
soliton field, recover the KNOT TYPE of its core curve and read off its Alexander
determinant -- which maps onto the Paper-6 carrier ladder (unknot=1 lepton,
trefoil=3 baryon, cinquefoil=5 nucleon, ...). This is the analysis primitive a
campaign's relax-then-ID census uses; it's physics-engine-agnostic (numpy/scipy,
plus pyknotid for the determinant), so it runs on a remote worker just as well as
locally.

Two pieces:

  (1) CORE EXTRACTION (`trace_implicit_curve`) -- a predictor-corrector trace of
      the implicit space curve {g1=0, g2=0}. Tangent is grad(g1) x grad(g2)
      (exact for an implicit codim-2 curve): predict along it, Newton-correct back
      onto the zero set. Front ends:
        * Faddeev n-field : `core_curves_from_n` (preimage of a pole; {n1=0,n2=0})
        * GPE psi-field   : `core_curves_from_psi` (vortex line; {Re,Im psi = 0})
      Returns ORDERED closed polylines (one per component) -- a link if several.

  (2) KNOT ID (`identify_knot`) -- pyknotid's Alexander determinant on the ordered
      polyline. pyknotid is an OPTIONAL dependency, imported lazily only when an
      identification is actually requested (`pip install jax-solitons[knots]`).
      pyknotid 0.5.3 predates numpy's removal of np.float, so the deprecated
      aliases are shimmed before import (see `_knot_class`).

(Moved from the null-worldtube-private dogfood tree, 2026-06-14, so it is a
reusable library primitive and so campaign workers can census remotely.)
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import RegularGridInterpolator as RGI
from scipy.spatial import cKDTree


# -- (2) knot identification via pyknotid (with numpy-alias shim) --------------
def _knot_class():
    """Import pyknotid.spacecurves.Knot, shimming np.float etc. (v0.5.3 compat).

    Lazy so pyknotid stays an OPTIONAL dependency -- importing this module, and
    tracing curves, needs only numpy/scipy; only `identify_knot` needs pyknotid.
    """
    for a, t in {"float": float, "int": int, "bool": bool,
                 "object": object, "complex": complex}.items():
        if not hasattr(np, a):
            setattr(np, a, t)
    try:
        from pyknotid.spacecurves import Knot
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "knot identification needs pyknotid -- `pip install jax-solitons[knots]`"
        ) from e
    return Knot


# carrier determinant -> Paper-6 sector label (T(2,n) ladder)
_DET_CARRIER = {1: "unknot/lepton T(2,1)", 2: "Hopf-link/meson T(2,2)",
                3: "trefoil/baryon T(2,3)", 5: "cinquefoil/nucleon T(2,5)",
                7: "T(2,7)", 9: "T(2,9)"}


def _resample_closed(pts, n):
    """Arc-length resample a closed polyline to n points (knot-type-safe while
    the resampled spacing stays well below the strand-to-strand clearance)."""
    closed = np.vstack([pts, pts[:1]])
    seg = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    si = np.linspace(0.0, s[-1], n, endpoint=False)
    return np.stack([np.interp(si, s, closed[:, k]) for k in range(3)], axis=1)


def identify_knot(points, max_points=600) -> dict:
    """Knot determinant + carrier label for an ordered (M,3) closed polyline.

    Curves are resampled down to max_points before pyknotid: a noisy tracer
    output (thousands of jittery points) inflates the crossing count and the
    pure-Python Alexander computation goes combinatorial (observed: hour-long
    hangs on evolved fields). length/n_points report the ORIGINAL curve.
    """
    pts = np.asarray(points, float)
    if len(pts) < 4:
        return dict(determinant=1, carrier=_DET_CARRIER[1], n_points=len(pts),
                    length=0.0)
    Knot = _knot_class()
    k = Knot(_resample_closed(pts, max_points) if len(pts) > max_points else pts,
             verbose=False)
    det = int(round(abs(k.determinant())))
    length = float(np.sum(np.linalg.norm(np.diff(
        np.vstack([pts, pts[:1]]), axis=0), axis=1)))
    return dict(determinant=det,
                carrier=_DET_CARRIER.get(det, f"det={det} (unrecognised)"),
                n_points=len(pts), length=length)


# -- (1) implicit-curve tracer: trace {g1 = 0, g2 = 0} in 3D -------------------
def trace_implicit_curve(g1, g2, axes, mask=None, h=None, seed_tol=0.30,
                         max_steps=40000, min_loop_pts=10, max_loops=6,
                         lost_tol=0.5):
    """Trace closed components of the space curve {g1=0, g2=0}.

    axes      : (x,y,z) ascending 1D coordinate arrays (grid is meshgrid 'ij').
    mask      : optional bool array; only seed where True (e.g. n3>0 sheet).
    h         : predictor step (default 0.6 dx).
    seed_tol  : a voxel is a seed candidate if sqrt(g1^2+g2^2) < seed_tol there.
    Returns   : list of (M,3) ordered point arrays (closed loops), longest first.
    """
    axes = tuple(np.asarray(a, float) for a in axes)
    d = np.array([a[1] - a[0] for a in axes])
    dx = float(d[0])
    if h is None:
        h = 0.6 * dx
    shape = g1.shape

    G = (RGI(axes, g1, bounds_error=False, fill_value=None),
         RGI(axes, g2, bounds_error=False, fill_value=None))
    grads = (np.gradient(g1, *d), np.gradient(g2, *d))
    DG = [[RGI(axes, c, bounds_error=False, fill_value=None) for c in gr]
          for gr in grads]
    lo = np.array([a[0] for a in axes])
    hi = np.array([a[-1] for a in axes])

    def field_jac(X):
        P = X[None, :]
        g = np.array([float(G[0](P)[0]), float(G[1](P)[0])])
        J = np.array([[float(DG[0][k](P)[0]) for k in range(3)],
                      [float(DG[1][k](P)[0]) for k in range(3)]])
        return g, J

    def correct(X, iters=6):
        for _ in range(iters):
            g, J = field_jac(X)
            A = J @ J.T + 1e-10 * np.eye(2)
            try:
                dX = J.T @ np.linalg.solve(A, -g)
            except np.linalg.LinAlgError:
                break
            X = X + dX
            if not (np.all(X >= lo) and np.all(X <= hi)):
                break
            if np.linalg.norm(dX) < 1e-3 * dx:
                break
        return X

    def tangent(J):
        t = np.cross(J[0], J[1])
        nt = np.linalg.norm(t)
        return t / nt if nt > 1e-12 else None

    s = g1**2 + g2**2
    if mask is not None:
        s = np.where(mask, s, np.inf)
    order = np.argsort(s, axis=None)

    loops = []
    loop_pts = []          # flat list of all accepted loop points (for dedup)
    seed_tree = None       # KD-tree over loop_pts; rebuilt only when a loop lands
    tube = max(1.5 * dx, h)

    for flat in order:
        if len(loops) >= max_loops:
            break
        ijk = np.unravel_index(flat, shape)
        if s[ijk] > seed_tol**2:
            break                                   # no good seeds remain
        # seed coordinate straight from the 1D axes (no full meshgrid allocation)
        X0 = np.array([axes[k][ijk[k]] for k in range(3)])
        X0 = correct(X0)
        if not (np.all(X0 >= lo) and np.all(X0 <= hi)):
            continue
        g, J = field_jac(X0)
        if np.hypot(*g) > seed_tol:
            continue
        if seed_tree is not None and seed_tree.query(X0)[0] < tube:
            continue                                # seed sits on an existing loop
        T = tangent(J)
        if T is None:
            continue

        path = [X0.copy()]
        X, Tprev = X0.copy(), T
        closed = False
        for step in range(max_steps):
            Xc = correct(X + h * Tprev)
            if not (np.all(Xc >= lo) and np.all(Xc <= hi)):
                break
            g, J = field_jac(Xc)
            if np.hypot(*g) > lost_tol:
                break
            T = tangent(J)
            if T is None:
                break
            if np.dot(T, Tprev) < 0:
                T = -T
            X, Tprev = Xc, T
            path.append(X.copy())
            if step > min_loop_pts and np.linalg.norm(X - X0) < 1.6 * h:
                closed = True
                break
        if closed and len(path) >= min_loop_pts:
            P = np.array(path)
            loops.append(P)
            loop_pts.extend(P.tolist())
            seed_tree = cKDTree(np.array(loop_pts))   # rebuild only on acceptance

    loops.sort(key=len, reverse=True)
    return loops


# -- field-specific front ends -------------------------------------------------
def core_curves_from_n(n1, n2, n3, axes, pole="auto", extra_mask=None, **kw):
    """Core curves = preimage of the pole (0,0,pole) of the Faddeev map n:R^3->S^2.

    The core is the ANTI-vacuum pole: the soliton sits at whichever pole the bulk
    vacuum does NOT occupy. Curve: {n1=0, n2=0} on the n3*pole>0 sheet.

    pole="auto" (default) detects it from the field: pole = -sign(mean n3), so a
    +z vacuum (mean n3 > 0) -> trace the -z sheet, and vice-versa. This is
    convention-agnostic: a vacuum at -z gives pole +1 (the historical default),
    while torus_knot_hopfion + arrested_flow leave the vacuum at +z and need
    pole -1. The old hard default pole=+1 silently traced the ENTIRE +z-vacuum
    bulk (~millions of seed points) on the latter -> hour-long tracer hangs.
    Pass pole=+1/-1 to force. extra_mask restricts seeding (e.g. a sphere around
    one daughter's centroid).
    """
    n1, n2, n3 = (np.asarray(x, float) for x in (n1, n2, n3))
    if pole == "auto":
        pole = -1 if float(np.mean(n3)) > 0.0 else +1
        if not np.any(pole * n3 > 0.0):   # degenerate field fills one pole entirely
            pole = -pole
    m = pole * n3 > 0.0
    if extra_mask is not None:
        m = m & extra_mask
    return trace_implicit_curve(n1, n2, axes, mask=m, **kw)


def core_curves_from_psi(psi, axes, **kw):
    """Vortex lines of a GPE field = {Re psi = 0, Im psi = 0}."""
    psi = np.asarray(psi)
    return trace_implicit_curve(psi.real.astype(float), psi.imag.astype(float),
                                axes, **kw)


def with_time_limit(seconds, fn, default):
    """Run fn() under a SIGALRM wall-clock budget; return default on timeout.

    The tracer (pure-python predictor-corrector) and pyknotid (pure-python
    Alexander) can both go pathological on turbulent evolved fields -- observed
    multi-hour grinds. Identification is a diagnostic, never worth wedging a
    run: time it out and report 'unidentified this snapshot'.
    Main-thread only (signal); on platforms without SIGALRM, runs unbounded.
    """
    import signal
    if not hasattr(signal, "SIGALRM"):  # pragma: no cover
        return fn()

    def _raise(signum, frame):
        raise TimeoutError

    old = signal.signal(signal.SIGALRM, _raise)
    signal.alarm(max(1, int(np.ceil(seconds))))   # ceil+clamp: int(0.x)=0 disables it
    try:
        return fn()
    except TimeoutError:  # pragma: no cover  (timing-dependent)
        return default
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


def curve_energy_scores(curves, e_density, axes):
    """Integrated energy density along each curve, ~ the loop integral of e dl.

    Use as identify_core_knot's `scores` so the PHYSICAL core (the
    energy-carrying filament) is ranked dominant, not merely the longest
    curve -- a low-energy spectator preimage filament can rival the core in
    length and flip the reported det with no topology change.
    """
    itp = RGI(axes, np.asarray(e_density, float),
              bounds_error=False, fill_value=0.0)
    scores = []
    for c in curves:
        e = itp(c)
        seg = np.linalg.norm(np.diff(np.vstack([c, c[:1]]), axis=0), axis=1)
        scores.append(float((e * seg).sum()))
    return scores


def identify_core_knot(curves, scores=None, max_points=600) -> dict:
    """Identify the dominant core curve's knot; report components.

    Dominant = argmax(scores) when given (e.g. curve_energy_scores), else the
    longest curve (the tracer's ordering). max_points is forwarded to
    identify_knot (resample cap before pyknotid; lower it -- e.g. 150 -- when the
    cython chelpers are unavailable and a jittery evolved curve makes the
    pure-Python Alexander routine go combinatorial).
    """
    if not curves:
        return dict(determinant=0, carrier="no core found", n_components=0,
                    n_points=0, length=0.0)
    idx = int(np.argmax(scores)) if scores is not None else 0
    info = identify_knot(curves[idx], max_points=max_points)
    info["n_components"] = len(curves)
    info["dominant"] = idx
    return info


def torus_knot(p, q, n=400, R=2.0, r=0.8):
    """Parametric T(p,q) torus knot as an (n,3) closed polyline (for tests/seeds)."""
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pr = R + r * np.cos(q * t)
    return np.stack([pr * np.cos(p * t), pr * np.sin(p * t), r * np.sin(q * t)], 1)

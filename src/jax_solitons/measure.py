"""Implicit-curve tracing on lattice fields (GPU, lax.scan).

Traces closed components of {g1 = 0, g2 = 0} by a predictor-corrector walk:
corrector = Gauss-Newton onto the curve through periodic trilinear
interpolation of (g1, g2) and their gradients; tangent = grad g1 x grad g2
(field-induced orientation); closure detected on the host between scan
chunks. The whole walk compiles to XLA scans (~100x the pure-Python tracer
in the source codebase, ~seconds per curve).

For a hopfion field n, the soliton core is the curve {n1 = 0, n2 = 0}
restricted to n3 > 0 (pass mask=n3 > 0 to trace_curves).

Hard-won robustness lore (each a stuck-run lesson in the source program):
seeds are consumed by EXACT basin assignment (batch-correct every candidate
and test landing on an already-traced loop) -- distance heuristics fail
where two linked tubes graze; loops shorter than min_len are junk; open
traces on tangled multi-tube fields remain a known gap.
"""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import numpy as np

from jax_solitons.grid import BoxGrid


def _trilerp_stack(F, p):
    """F: (M,N,N,N) field stack; p: (3,) point in GRID units. -> (M,)"""
    N = F.shape[1]
    p0 = jnp.floor(p).astype(jnp.int32)
    f = p - p0
    out = jnp.zeros(F.shape[0], dtype=F.dtype)
    for di in (0, 1):
        for dj in (0, 1):
            for dk in (0, 1):
                w = ((f[0] if di else 1 - f[0]) *
                     (f[1] if dj else 1 - f[1]) *
                     (f[2] if dk else 1 - f[2]))
                idx = (p0 + jnp.array([di, dj, dk])) % N
                out = out + w * F[:, idx[0], idx[1], idx[2]]
    return out


def build_tracer(g1, g2, dx, h=None, iters=6, max_steps=60000, lost_tol=0.5):
    """Compile a tracer for the curve {g1=0, g2=0} (periodic fields, grid dx).

    Returns (trace, h, correct_batch): trace(X0_phys) -> (X0_corrected,
    points (k,3) physical); coordinates x = -L/2 + i*dx <-> p = (x+L/2)/dx.
    """
    N = g1.shape[0]
    L = N * dx
    if h is None:
        h = 0.6 * dx
    gr1 = np.gradient(np.asarray(g1), dx)
    gr2 = np.gradient(np.asarray(g2), dx)
    F = jnp.asarray(np.stack([g1, g2, *gr1, *gr2]))   # (8,N,N,N)

    def to_grid(X):
        return (X + L / 2) / dx

    def evalf(X):
        v = _trilerp_stack(F, to_grid(X))
        return v[:2], v[2:].reshape(2, 3)

    def correct(X):
        def body(_, X):
            g, J = evalf(X)
            A = J @ J.T + 1e-10 * jnp.eye(2)
            return X + J.T @ jnp.linalg.solve(A, -g)
        return jax.lax.fori_loop(0, iters, body, X)

    CHUNK = 4000

    @jax.jit
    def start(X0):
        X0 = correct(X0)
        _, J0 = evalf(X0)
        T0 = jnp.cross(J0[0], J0[1])
        return X0, T0 / (jnp.linalg.norm(T0) + 1e-30)

    @jax.jit
    def trace_chunk(X, T, done):
        def step(carry, _):
            X, T, done = carry
            Xn = correct(X + h * T)
            g, J = evalf(Xn)
            Tn = jnp.cross(J[0], J[1])
            Tn = Tn / (jnp.linalg.norm(Tn) + 1e-30)
            lost = jnp.hypot(g[0], g[1]) > lost_tol
            keep = done | lost
            Xn = jnp.where(keep, X, Xn)
            Tn = jnp.where(keep, T, Tn)
            return (Xn, Tn, keep), Xn

        carry, pts = jax.lax.scan(step, (X, T, done), None, length=CHUNK)
        return carry, pts

    def trace(X0):
        """Chunked scan with host-side closure check between chunks."""
        X0, T0 = start(X0)
        X, T, done = X0, T0, jnp.asarray(False)
        chunks = []
        for _ in range(max_steps // CHUNK):
            (X, T, done), pts = trace_chunk(X, T, done)
            chunks.append(np.asarray(pts))
            d = np.linalg.norm(chunks[-1] - np.asarray(X0), axis=1)
            if bool(done) or d.min() < 0.75 * h:
                break
        return X0, np.vstack(chunks)

    correct_batch = jax.jit(jax.vmap(correct))
    return trace, h, correct_batch


def _close_loop(X0, pts, h, min_pts=30):
    """Host: cut the scan output at first return to X0; None if never closes."""
    pts = np.asarray(pts)
    d = np.linalg.norm(pts - np.asarray(X0), axis=1)
    hits = np.where(d[min_pts:] < 0.75 * h)[0]
    if len(hits) == 0:
        return None
    end = min_pts + hits[0]
    return np.vstack([np.asarray(X0)[None, :], pts[:end]])


def trace_curves(g1, g2, grid: BoxGrid, mask=None, seed_tol=0.30, max_loops=8,
                 min_len=3.0, verbose=False):
    """All closed components of {g1=0, g2=0}, longest first (physical coords).

    Seeds = lattice sites with the smallest residual g1^2+g2^2 (optionally
    masked); each seed is GPU-corrected onto the curve, and consumed iff its
    corrected point lands on an already-traced loop (exact basin assignment).
    """
    from scipy.spatial import cKDTree
    g1 = np.asarray(g1, float)
    g2 = np.asarray(g2, float)
    dx = grid.dx
    trace, h, correct_batch = build_tracer(g1, g2, dx)
    s = g1**2 + g2**2
    if mask is not None:
        s = np.where(mask, s, np.inf)
    order = np.argsort(s, axis=None)
    n_cand = int(np.searchsorted(np.sort(s, axis=None), seed_tol**2))
    if n_cand == 0:
        return []
    cand_flat = order[:n_cand]
    ax = np.asarray(grid.axis(), float)
    grids = np.meshgrid(ax, ax, ax, indexing="ij")
    cand = np.stack([g.flat[cand_flat] for g in grids], 1)
    corrected = np.asarray(correct_batch(jnp.asarray(cand)))
    consumed = np.zeros(n_cand, bool)
    curves, trees = [], []
    tried = 0
    while not consumed.all() and len(curves) < max_loops:
        idx = int(np.argmin(consumed))   # first unconsumed (lowest residual)
        tried += 1
        t0 = time.time()
        X0c, pts = trace(jnp.asarray(cand[idx]))
        loop = _close_loop(X0c, pts, h)
        consumed[idx] = True
        if verbose:
            print(f"    seed {tried}: trace {time.time()-t0:.1f}s -> "
                  f"{'closed %d pts' % len(loop) if loop is not None else 'open'}",
                  flush=True)
        if loop is None:
            continue
        tr = cKDTree(loop[::2])
        consumed |= tr.query(corrected)[0] < 0.35
        seg = np.linalg.norm(np.diff(loop, axis=0), axis=1).sum()
        if seg < min_len:
            continue
        if any(np.median(t.query(loop[::5])[0]) < 0.3 for t in trees):
            continue
        curves.append(loop)
        trees.append(tr)
    curves.sort(key=len, reverse=True)
    return curves


def curve_length(pts) -> float:
    """Total arc length of a closed polyline (includes the closing segment)."""
    closed = np.vstack([pts, pts[:1]])
    return float(np.linalg.norm(np.diff(closed, axis=0), axis=1).sum())


def resample_curve(pts, n=500):
    """Arc-length resample a closed polyline to n points (invariant-friendly)."""
    closed = np.vstack([pts, pts[:1]])
    seg = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    si = np.linspace(0.0, s[-1], n, endpoint=False)
    return np.stack([np.interp(si, s, closed[:, k]) for k in range(3)], 1)


def gauss_lk(a, b) -> float:
    """Gauss linking number of two closed polylines (midpoint rule)."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    da = np.roll(a, -1, 0) - a
    db = np.roll(b, -1, 0) - b
    am, bm = a + 0.5 * da, b + 0.5 * db
    r = am[:, None, :] - bm[None, :, :]
    rn = np.linalg.norm(r, axis=2) ** 3
    cr = np.cross(da[:, None, :], db[None, :, :])
    return float(np.sum(np.einsum("ijk,ijk->ij", cr, r) / rn) / (4 * np.pi))

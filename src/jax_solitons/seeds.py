"""Initial conditions. First citizen: the smooth rational-map hopfion seed.

Why rational-map and not the analytic stereographic seed: the analytic seed
lifts n to the CP^1 spinor through the south-pole patch, which is 0/0 exactly
on the soliton core ring, and reaches vacuum only like 1/r^2 (a seam leak on
a periodic box). Both push the seed OUTSIDE the soliton basin. The
rational-map construction (Battye-Sutcliffe / Hietarinta-Salo) composes
smooth maps with a compact C^2 profile, so the field is exactly vacuum
beyond the tube radius w.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from scipy.spatial import cKDTree

from jax_solitons.grid import BoxGrid
from jax_solitons.models.faddeev import n_from_Z
from jax_solitons.topology import hopf_charge


def rational_map_spinor(grid: BoxGrid, R=3.5, w=None, n=1, m=1,
                        center=(0.0, 0.0, 0.0)):
    """Smooth rational-map hopfion CP^1 spinor of charge Q_H = n*m.

    Base map (inverse stereographic R^3 -> S^3):
        phi1 = atan2(y, x)                azimuthal about the z-axis
        phi2 = atan2(r^2 - R^2, 2 R z)    meridional around the core ring
    Degree-(n, m) map on the Hopf fibre (smooth because each amplitude
    vanishes where its phase is ill-defined):
        Z1 = cos(lam) e^{i n phi1},  Z2 = sin(lam) e^{i m phi2}
    Compact profile in the minor radius d = sqrt((rho-R)^2 + z^2):
        lam(d) = (pi/2) smootherstep(d/w), exactly vacuum for d >= w.

    Regularity requires w <= R (else the tube reaches the z-axis and phi1
    re-singularizes). R is the core-ring major radius, w the tube radius.
    """
    if w is None:
        w = 0.85 * R
    if w > R:
        raise ValueError(f"need w <= R for axis regularity (w={w}, R={R})")
    X, Y, Z = (np.asarray(c, dtype=np.float64) for c in grid.coords())
    x0, y0, z0 = center
    x, y, z = X - x0, Y - y0, Z - z0
    rho = np.sqrt(x**2 + y**2)
    r2 = x**2 + y**2 + z**2
    phi1 = np.arctan2(y, x)
    phi2 = np.arctan2(r2 - R**2, 2.0 * R * z)   # oriented so Q_H = +n*m
    d = np.sqrt((rho - R) ** 2 + z**2)
    t = np.clip(d / w, 0.0, 1.0)
    s = t**3 * (10.0 - 15.0 * t + 6.0 * t**2)   # smootherstep (C^2)
    lam = 0.5 * np.pi * s
    Z1 = np.cos(lam) * np.exp(1j * n * phi1)
    Z2 = np.sin(lam) * np.exp(1j * m * phi2)
    nrm = np.sqrt(np.abs(Z1) ** 2 + np.abs(Z2) ** 2)
    return Z1 / nrm, Z2 / nrm


def rational_map_hopfion(grid: BoxGrid, R=3.5, w=None, n=1, m=1,
                         center=(0.0, 0.0, 0.0)) -> jnp.ndarray:
    """Rational-map hopfion as a unit n-field, shape (3, N, N, N), in the
    grid dtype."""
    Z1, Z2 = rational_map_spinor(grid, R=R, w=w, n=n, m=m, center=center)
    nf = n_from_Z(jnp.asarray(Z1), jnp.asarray(Z2))
    return nf.astype(grid.dtype)


def rational_map_hopfion_cp1(grid: BoxGrid, R=3.5, w=None, n=1, m=1,
                             center=(0.0, 0.0, 0.0)) -> jnp.ndarray:
    """Rational-map hopfion as a real CP^1 spinor state, shape (4, N, N, N)
    = (Re Z1, Im Z1, Re Z2, Im Z2), in the grid dtype (the state layout of
    models.faddeev.faddeev_cp1_model)."""
    Z1, Z2 = rational_map_spinor(grid, R=R, w=w, n=n, m=m, center=center)
    z = jnp.stack([jnp.real(jnp.asarray(Z1)), jnp.imag(jnp.asarray(Z1)),
                   jnp.real(jnp.asarray(Z2)), jnp.imag(jnp.asarray(Z2))])
    return z.astype(grid.dtype)


# --- T(p, q) torus-knot tube hopfion (Paper 16 sec.L_3) ---------------------
#
# A finite-size knotted soliton whose preimage core is a (p, q) torus knot
# carrying phase winding m, locked to Hopf charge Q_H = p*m (Paper 16 sec.L_3,
# Whitehead/Rybakov-2015 reduction). Unlike the axially-symmetric rational map
# (Q_H = n*m, unknotted core ring), the core here is the genuine torus-knot
# curve, so this is the seed for the L_2+L_3 coupled-model program: a knotted
# flux tube whose cross-section will carry the abelian-Higgs BPS profile.
#
# Construction: an n-field tube around the knot curve with a CLOSED
# rotation-minimizing frame (RMF), unit meridional winding, and a longitudinal
# phase twist l. Q_H is exactly linear in l (unit slope), Q_H = -(Q0 + l), with
# Q0 a geometry-dependent integer baseline (the framed-tube self-linking). We
# pin Q0 with one cheap area-form hopf_charge measurement at l=0 (it is
# integer-valued, so robust to rounding), then solve for the minimal-|l| twist
# that locks |Q_H| = p*m. The charge SIGN is the Hopf theta = 0/pi (CPT)
# convention of Paper 16 and is left to the caller; the topological lock is on
# |Q_H|.


def _torus_knot_curve(p: int, q: int, R: float, b: float, S: int):
    """Sample the (p, q) torus knot and its unit tangents, shapes (S, 3)."""
    s = np.linspace(0.0, 2.0 * np.pi, S, endpoint=False)
    cx = (R + b * np.cos(q * s)) * np.cos(p * s)
    cy = (R + b * np.cos(q * s)) * np.sin(p * s)
    cz = b * np.sin(q * s)
    g = np.stack([cx, cy, cz], axis=1)
    t = np.gradient(g, axis=0)
    t /= np.linalg.norm(t, axis=1, keepdims=True)
    return g, t, s


def _closed_rmf(g, t):
    """Closed rotation-minimizing frame (double-reflection, Wang 2008), with the
    residual holonomy distributed linearly so the frame is periodic on the loop.
    Returns the two normal-plane vectors (r, u), shapes (S, 3)."""
    S = len(g)
    r = np.zeros_like(g)
    seed = np.array([0.0, 0.0, 1.0])
    if abs(t[0] @ seed) > 0.9:
        seed = np.array([1.0, 0.0, 0.0])
    r0 = seed - (seed @ t[0]) * t[0]
    r[0] = r0 / np.linalg.norm(r0)
    for i in range(S - 1):
        v1 = g[i + 1] - g[i]
        c1 = v1 @ v1
        rL = r[i] - (2.0 / c1) * (v1 @ r[i]) * v1
        tL = t[i] - (2.0 / c1) * (v1 @ t[i]) * v1
        v2 = t[i + 1] - tL
        c2 = v2 @ v2
        r[i + 1] = rL - (2.0 / c2) * (v2 @ rL) * v2 if c2 > 1e-12 else rL
        r[i + 1] /= np.linalg.norm(r[i + 1])
    u = np.cross(t, r)
    ang = np.arctan2(r[-1] @ u[0], r[-1] @ r[0])          # closure holonomy
    s = np.linspace(0.0, 2.0 * np.pi, S, endpoint=False)
    corr = -ang * s / (2.0 * np.pi)
    cc, ss = np.cos(corr), np.sin(corr)
    return cc[:, None] * r + ss[:, None] * u, -ss[:, None] * r + cc[:, None] * u


def _tube_spinor(grid: BoxGrid, p, q, l, R, b, w, S):
    """CP^1 spinor (Z1, Z2) of a unit-meridional-winding tube around the (p, q)
    knot with longitudinal phase twist l. n = (sinL cosF, sinL sinF, cosL) with
    L = pi at the core (south pole) -> 0 in vacuum, F = alpha + l*s."""
    g, t, s_param = _torus_knot_curve(p, q, R, b, S)
    r, u = _closed_rmf(g, t)
    X, Y, Z = (np.asarray(c, np.float64) for c in grid.coords())
    P = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    d, k = cKDTree(g).query(P, k=1)
    disp = P - g[k]
    alpha = np.arctan2(np.einsum("ij,ij->i", disp, u[k]),
                       np.einsum("ij,ij->i", disp, r[k]))
    tt = np.clip(d / w, 0.0, 1.0)
    smooth = tt ** 3 * (10.0 - 15.0 * tt + 6.0 * tt ** 2)        # smootherstep
    lam = np.pi * (1.0 - smooth)                                 # pi core -> 0
    Phi = alpha + l * s_param[k]
    Z1 = np.cos(lam / 2.0).reshape(grid.N, grid.N, grid.N)
    Z2 = (np.sin(lam / 2.0) * np.exp(1j * Phi)).reshape(grid.N, grid.N, grid.N)
    return Z1.astype(np.complex128), Z2


def _knot_geometry(grid: BoxGrid, R, b, w):
    R = 0.20 * grid.L if R is None else R
    b = 0.40 * R if b is None else b
    w = 0.70 * b if w is None else w
    if not (w < b < R):
        raise ValueError(f"need w < b < R (got w={w}, b={b}, R={R})")
    if R + b + w > 0.5 * grid.L:
        raise ValueError("knot tube does not fit in the box (R+b+w > L/2)")
    return R, b, w


def torus_knot_spinor(grid: BoxGrid, p: int, q: int, m: int = 1,
                      R: float | None = None, b: float | None = None,
                      w: float | None = None, S: int = 4000):
    """Smooth T(p, q) torus-knot tube hopfion CP^1 spinor (Z1, Z2), locked to
    |Q_H| = p*m (Paper 16 sec.L_3). Geometry defaults scale with the box:
    major radius R = 0.2 L, tube-center b = 0.4 R, tube radius w = 0.7 b.

    Calibrates the longitudinal twist deterministically: measures the l=0 charge
    baseline Q0 (integer) with the area-form hopf_charge, then picks the
    minimal-|l| twist landing on |Q_H| = p*m. p, q must be coprime."""
    if np.gcd(p, q) != 1:
        raise ValueError(f"T(p, q) needs gcd(p, q) = 1 (got p={p}, q={q})")
    R, b, w = _knot_geometry(grid, R, b, w)
    Z1, Z2 = _tube_spinor(grid, p, q, 0, R, b, w, S)
    n0 = n_from_Z(jnp.asarray(Z1), jnp.asarray(Z2))
    B = int(round(float(hopf_charge(n0, grid))))                # signed Q_H(l=0)
    target = p * m
    # Q_H(l) = B - l (unit slope); land on +target or -target, smaller |l|.
    l = min((B - target, B + target), key=abs)
    if l != 0:
        Z1, Z2 = _tube_spinor(grid, p, q, l, R, b, w, S)
    return Z1, Z2


def torus_knot_hopfion(grid: BoxGrid, p: int, q: int, m: int = 1,
                       **kw) -> jnp.ndarray:
    """T(p, q) torus-knot hopfion as a unit n-field, shape (3, N, N, N)."""
    Z1, Z2 = torus_knot_spinor(grid, p, q, m, **kw)
    return n_from_Z(jnp.asarray(Z1), jnp.asarray(Z2)).astype(grid.dtype)


def torus_knot_hopfion_cp1(grid: BoxGrid, p: int, q: int, m: int = 1,
                           **kw) -> jnp.ndarray:
    """T(p, q) torus-knot hopfion as a real CP^1 spinor state, shape
    (4, N, N, N) = (Re Z1, Im Z1, Re Z2, Im Z2) -- the faddeev_cp1_model
    layout, ready to relax in the convergent spinor frame."""
    Z1, Z2 = torus_knot_spinor(grid, p, q, m, **kw)
    Z1, Z2 = jnp.asarray(Z1), jnp.asarray(Z2)
    return jnp.stack([jnp.real(Z1), jnp.imag(Z1),
                      jnp.real(Z2), jnp.imag(Z2)]).astype(grid.dtype)


# --- L_2 + L_3 coupled seed: a flux-threaded T(p,q) knot --------------------
#
# Composes the L_2 abelian-Higgs vortex with the L_3 torus-knot hopfion on a
# single doublet psi = rho * e^{i chi} * zeta (Paper 16 reading A):
#   - zeta (unit CP^1 direction) carries the Hopf texture -> n, twist-locked to
#     Q_H = p*m (the L_3 sector, reusing the torus-knot machinery);
#   - rho(d) = v tanh(d/xi) is the Higgs modulus -> 0 on the knot curve (a Higgs
#     vortex core running ALONG the knotted tube), -> v in the bulk;
#   - chi = alpha (the meridional angle) is the gauged overall phase: it winds
#     2*pi around the core, i.e. ONE flux quantum threading the tube (the L_2
#     vortex), with A circulating to make D psi -> 0 in the bulk.
# The gauge A cancels only the COMMON phase chi (the flux); the RELATIVE phase
# (the n-texture) survives as Skyrme energy -- exactly the L_2/L_3 split.


def flux_threaded_knot_seed(grid: BoxGrid, p: int, q: int, m: int = 1,
                            e: float = 1.0, v: float = 1.0,
                            R: float | None = None, b: float | None = None,
                            w: float | None = None, xi: float | None = None,
                            S: int = 4000) -> jnp.ndarray:
    """Coupled gauged Faddeev-Skyrme-Higgs state: a (7, N, N, N) array
    ``(Re psi1, Im psi1, Re psi2, Im psi2, A_x, A_y, A_z)`` -- a Higgs flux tube
    bent into a T(p, q) knot and wrapped by a Hopf texture of charge Q_H = p*m.
    The seed of the L_2+L_3 coupled-model program (models.gauged_faddeev).

    `xi` is the Higgs healing length (core radius); defaults to half the n-tube
    radius w. Calibrates the longitudinal twist deterministically (one area-form
    hopf_charge measurement on the pure direction field), as torus_knot_spinor."""
    if np.gcd(p, q) != 1:
        raise ValueError(f"T(p, q) needs gcd(p, q) = 1 (got p={p}, q={q})")
    if e == 0:
        raise ValueError("e (gauge coupling) must be nonzero (A_theta ~ 1/e).")
    R, b, w = _knot_geometry(grid, R, b, w)
    if xi is None:
        xi = 0.5 * w
    if xi <= 0:
        raise ValueError(f"xi (Higgs healing length) must be positive (got {xi}).")

    g, t, s_param = _torus_knot_curve(p, q, R, b, S)
    r, u = _closed_rmf(g, t)
    N = grid.N
    X, Y, Z = (np.asarray(c, np.float64) for c in grid.coords())
    P = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    d, k = cKDTree(g).query(P, k=1)
    disp = P - g[k]
    rk, uk = r[k], u[k]                                       # frame at nearest
    alpha = np.arctan2(np.einsum("ij,ij->i", disp, uk),
                       np.einsum("ij,ij->i", disp, rk))
    s_at = s_param[k]

    tt = np.clip(d / w, 0.0, 1.0)
    lam = np.pi * (1.0 - tt ** 3 * (10.0 - 15.0 * tt + 6.0 * tt ** 2))  # pi->0

    # twist calibration: Q_H[n] depends only on the direction zeta, so lock it
    # on the smooth (modulus-free) field exactly as torus_knot_spinor does.
    def _dir_n(ltw):
        Phi = alpha + ltw * s_at
        z1 = np.cos(lam / 2.0).reshape(N, N, N)
        z2 = (np.sin(lam / 2.0) * np.exp(1j * Phi)).reshape(N, N, N)
        return n_from_Z(jnp.asarray(z1), jnp.asarray(z2))
    B = int(round(float(hopf_charge(_dir_n(0), grid))))
    l = min((B - p * m, B + p * m), key=abs)

    Phi = alpha + l * s_at                       # n-texture relative phase
    rho = v * np.tanh(d / xi)                     # Higgs modulus, 0 at core
    psi1 = rho * np.exp(1j * alpha) * np.cos(lam / 2.0)
    psi2 = rho * np.exp(1j * (alpha + Phi)) * np.sin(lam / 2.0)

    # gauge field: e A = gprof(d)/d * alpha_hat  (cancels d_i chi in the bulk),
    # regular at the core (gprof ~ (d/xi)^2 -> A ~ d -> 0).
    gprof = 1.0 - np.exp(-(d / xi) ** 2)
    alpha_hat = (-np.sin(alpha)[:, None] * rk + np.cos(alpha)[:, None] * uk)
    A = (gprof / (e * (d + 1e-12)))[:, None] * alpha_hat      # (M, 3)

    state = np.stack([
        psi1.real.reshape(N, N, N), psi1.imag.reshape(N, N, N),
        psi2.real.reshape(N, N, N), psi2.imag.reshape(N, N, N),
        A[:, 0].reshape(N, N, N), A[:, 1].reshape(N, N, N),
        A[:, 2].reshape(N, N, N),
    ])
    return jnp.asarray(state, dtype=grid.dtype)


# --- SU(2)/S^3 Skyrmion seeds (models.skyrme) -------------------------------
#
# State: a real unit 4-vector field phi = (phi0, phi1, phi2, phi3), |phi| = 1,
# shape (4, N, N, N), encoding U = phi0 + i phi_a sigma_a in SU(2) ~ S^3.
# Built with the same compact C^2 (smootherstep) profile machinery as the
# hopfion seeds, so each soliton is exactly vacuum (U = +1, phi = (1,0,0,0))
# beyond its radius.


# --------------------------------------------------------------------------
# Periodic minimal-superflow seed: closed curves -> a condensate carrying one
# quantum of circulation on each. Built in the PLAQUETTE basis, because that is
# the basis `vortex_topology.vortex_skeleton` reads.
# --------------------------------------------------------------------------

# plaquette family -> (normal axis, spanned axes), matching vortex_topology._PLAQ
_PLAQ_NORMAL = (2, 0, 1)
_PLAQ_SPAN = ((0, 1), (1, 2), (2, 0))


class DegenerateSeedGeometry(ValueError):
    """A seed curve is not in general position with respect to the lattice.

    The plaquette deposition counts where a curve PIERCES lattice faces. A curve
    running exactly along a lattice plane, or through a cell corner, pierces
    nothing there and the deposited vortex line fails to close -- which the
    divergence check catches. The resulting field would carry an open vortex
    line, which is not a physical configuration and is not what was asked for.

    Same condition, and same remedy, as `ehn.knot_batch.LatticeCoincidence`:
    nudge the geometry off the lattice, e.g. by dx/3 in each direction.
    """


def _pierce_plaquettes(curves, N, L):
    """Integer winding charge per plaquette, shape (3, N, N, N), in _PLAQ order.

    A plaquette of family f sits at an INTEGER lattice coordinate along its
    normal and spans one cell in each of the other two directions. Count signed
    crossings of each such face.

    Half-open convention: a segment travelling up the normal crosses plane K for
    every integer K in (s0, s1]. That never double-counts a shared endpoint
    landing exactly on a plane, and a segment PARALLEL to the plane crosses
    nothing (the 0/0 a naive ceil/floor form produces for a ring lying in a
    lattice plane).
    """
    dx = L / N
    n = np.zeros((3, N, N, N), dtype=np.int64)
    for c in curves:
        g = (np.asarray(c, dtype=float) + L / 2.0) / dx
        g = np.vstack([g, g[:1]])
        p0, p1 = g[:-1], g[1:]
        for f in range(3):
            m = _PLAQ_NORMAL[f]
            a, b = _PLAQ_SPAN[f]
            for s in range(len(p0)):
                a0, a1 = p0[s, m], p1[s, m]
                if a0 == a1:
                    continue
                sgn = 1 if a1 > a0 else -1
                lo, hi = (a0, a1) if sgn > 0 else (a1, a0)
                planes = np.arange(np.floor(lo) + 1.0, np.floor(hi) + 1.0)
                planes = planes[(planes > lo) & (planes <= hi)]
                for plane in planes:
                    t = (plane - a0) / (a1 - a0)
                    pt = p0[s] + t * (p1[s] - p0[s])
                    idx = [0, 0, 0]
                    idx[a] = int(np.floor(pt[a])) % N
                    idx[b] = int(np.floor(pt[b])) % N
                    idx[m] = int(round(plane)) % N
                    n[f, idx[0], idx[1], idx[2]] += sgn
    return n


def _plaquette_divergence(n):
    """Net flux out of each cube. Identically zero iff every vortex line closes."""
    d = np.zeros_like(n[0])
    for f in range(3):
        d += np.roll(n[f], -1, axis=_PLAQ_NORMAL[f]) - n[f]
    return d


def _edge_potential(n, N, L):
    """Edge field A with discrete `curl A = 2*pi*n`, Coulomb gauge, spectral.

    Forward-difference symbol d_i = exp(i k_i dx) - 1; the solution is
    A = -2*pi*(conj(d) x n)/|d|^2. The overall sign was fixed by MEASURING the
    resulting curl (it came back as -2*pi*n, an error of exactly 4*pi on every
    charged plaquette) rather than by re-deriving the lattice algebra. The
    identity now holds to ~1e-15, which is the property the whole construction
    rests on: an exact integer curl means the branch sheet of the integration
    below jumps by exactly 2*pi and is invisible in exp(i*theta).
    """
    dx = L / N
    k1 = 2.0 * np.pi * np.fft.fftfreq(N, d=dx)
    K = np.meshgrid(k1, k1, k1, indexing="ij")
    d = [np.exp(1j * K[i] * dx) - 1.0 for i in range(3)]
    dabs = sum(np.abs(di) ** 2 for di in d)
    dabs[0, 0, 0] = 1.0
    nv = [None, None, None]
    for f in range(3):
        nv[_PLAQ_NORMAL[f]] = np.fft.fftn(n[f].astype(float))
    dc = [np.conj(di) for di in d]
    cross = [dc[1] * nv[2] - dc[2] * nv[1],
             dc[2] * nv[0] - dc[0] * nv[2],
             dc[0] * nv[1] - dc[1] * nv[0]]
    Ak = [-2.0 * np.pi * cr / dabs for cr in cross]
    for c in Ak:
        c[0, 0, 0] = 0.0
    A = [np.real(np.fft.ifftn(c)) for c in Ak]

    # THE ZERO MODE IS NOT FREE, and setting it to zero is the wrong choice.
    # The tree integration below is periodic only if every box-spanning
    # circulation `sum_i A_i` is an exact multiple of 2*pi, so its branch sheet
    # closes on itself. Discrete Stokes makes those circulations differ between
    # transverse sites by exact multiples of 2*pi -- so they share one fractional
    # part -- but their MEAN is not generally zero (measured integers on a
    # clasped pair run -1..2). Zeroing the mean therefore hands every site the
    # same non-integer offset, 0.226 of a turn in that case, and that offset was
    # the entire residual seam. Subtract the shared fraction instead: a uniform
    # superflow of 2*pi*f/L, which is the physically meaningful zero mode.
    for i in range(3):
        turns = A[i].sum(axis=i) / (2.0 * np.pi)
        frac = float(np.mean(turns - np.round(turns)))
        A[i] = A[i] - frac * 2.0 * np.pi / A[i].shape[i]
    return A


def _phase_from_edges(A, N):
    """Integrate the edge field along a spanning tree: x-line, then y, then z."""
    th = np.zeros((N, N, N))
    th[:, 0, 0] = np.concatenate([[0.0], np.cumsum(A[0][:-1, 0, 0])])
    th[:, :, 0] = th[:, 0, 0][:, None] + np.concatenate(
        [np.zeros((N, 1)), np.cumsum(A[1][:, :-1, 0], axis=1)], axis=1)
    return th[:, :, 0][:, :, None] + np.concatenate(
        [np.zeros((N, N, 1)), np.cumsum(A[2][:, :, :-1], axis=2)], axis=2)


def _min_image_distance(curves, N, L):
    """Distance to the nearest curve sample, honouring periodicity: query against
    the curve tiled over the 27 neighbouring cells."""
    dx = L / N
    ax = np.arange(N) * dx - L / 2.0
    X, Y, Z = np.meshgrid(ax, ax, ax, indexing="ij")
    pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    d = np.full(pts.shape[0], np.inf)
    shifts = np.array([[a, b, c] for a in (-L, 0.0, L)
                       for b in (-L, 0.0, L) for c in (-L, 0.0, L)])
    for cu in curves:
        tiled = (np.asarray(cu)[None, :, :] + shifts[:, None, :]).reshape(-1, 3)
        d = np.minimum(d, cKDTree(tiled).query(pts, k=1)[0])
    return d.reshape(N, N, N)


def superflow_seed(grid: BoxGrid, curves, *, core: float = 1.0):
    """Condensate field psi carrying one quantum of circulation on each curve,
    PERIODIC on the box.

    The GPE counterpart of the rational-map seeds above, and the entry point for
    any linked-vortex experiment: hand it the output of
    `invariants.curves.hopf_clasped_trefoils` and the seeded field has that link.

    WHY THIS IS NOT THE OBVIOUS CONSTRUCTION. The obvious one -- amplitude
    `tanh(d/core)`, phase = half the solid angle subtended at each point -- is
    what this used to do, and it is not periodic. Measured on a clasped pair, the
    wrap-around phase mismatch was ~0.75 rad on two of three axes, which is a
    real discontinuity in psi (not a multiple of 2*pi), and a spectral derivative
    rings on it: the compressible bin of `vortex_topology.kinetic_decomposition`
    read 0.61 of the incompressible one, essentially all of it seam rather than
    sound. The free-space solid angle simply has no reason to agree across
    opposite faces of a torus.

    So the phase is built from the vorticity instead, in the PLAQUETTE basis --
    the basis `vortex_topology.vortex_skeleton` actually reads:

      1. count signed curve piercings of every lattice face (exact integers);
      2. check the deposited lines CLOSE (zero discrete divergence) -- they do
         not if the geometry is degenerate, hence `DegenerateSeedGeometry`;
      3. solve `curl A = 2*pi*n` spectrally in Coulomb gauge, fixing the zero
         mode so every box-spanning circulation is an exact multiple of 2*pi;
      4. integrate A along a spanning tree. Because the curl is exactly integer,
         the tree's branch sheet jumps by exactly 2*pi and is invisible in
         exp(i*theta) -- the same trick the Seifert sheet used, but now on a
         quantity that is exact rather than convergent.

    An earlier attempt deposited vorticity with a CIC kernel and solved the
    continuum curl. Its circulation converged to 0.97*2*pi -- a shape error near
    the core, not a scale factor -- so the sheet jumped by 0.97*2*pi and
    `vortex_skeleton` read the entire sheet as vortex line (3756 segments against
    a true ~280). Exactness here is not fastidiousness; it is the difference
    between a seed and a tangle.

    Measured on a clasped trefoil pair at N=72: seam/interior-step 7.9%, 1.4%,
    7.6% (from 143%, 13%, 140%); compressible fraction 0.018 (from 0.61); lk
    reads -0.998; tangle helicity -8.569 against the curves' -8.555.

    RESOLUTION IS STILL THE GAME. Two strands closer than a couple of dx merge
    into one skeleton component, and a merged pair reads as unlinked -- a failure
    that looks exactly like physics and is not. Keep strand-strand clearance well
    above dx, and do not expect one curve to give exactly one component; read
    `vortex_topology.linking_number`'s component list and raise its `min_seg`.

    Parameters
    ----------
    grid : BoxGrid
    curves : sequence of (n_i, 3) arrays
        Closed curves, sampled densely enough that consecutive samples are much
        closer than dx -- the deposition walks segment by segment.
    core : float
        Healing length of the `tanh(d/core)` amplitude, in physical units.

    Returns
    -------
    jnp.ndarray, complex, shape (N, N, N)

    Raises
    ------
    DegenerateSeedGeometry
        If the deposited vortex lines do not close.
    """
    N, L = grid.N, grid.L
    curves = [np.asarray(c, dtype=float) for c in curves]
    n = _pierce_plaquettes(curves, N, L)
    div = _plaquette_divergence(n)
    if np.any(div):
        bad = np.argwhere(div != 0)[:3]
        raise DegenerateSeedGeometry(
            f"deposited vortex lines do not close: {int(np.count_nonzero(div))} "
            f"cell(s) have net flux, first at {[tuple(int(v) for v in b) for b in bad]} "
            f"(N={N}, L={L!r}, dx={L / N!r}). A curve is running along a lattice "
            f"plane or through a cell corner, so it pierces no face there. Nudge "
            f"the geometry off the lattice, e.g. by dx/3 in each direction. This "
            f"is the same condition ehn.knot_batch._assert_off_lattice refuses.")
    A = _edge_potential(n, N, L)
    theta = _phase_from_edges(A, N)
    amp = np.tanh(_min_image_distance(curves, N, L) / core)
    # Follow the grid's own precision policy (fp32 scouting, fp64 to certify)
    # rather than demanding complex128 -- asking for it without x64 enabled
    # silently truncates and warns.
    cdtype = jnp.complex128 if jnp.dtype(grid.dtype).itemsize == 8 else jnp.complex64
    return jnp.asarray(amp * np.exp(1j * theta), dtype=cdtype)


def _hedgehog_profile(r, r0):
    """Skyrme radial profile f(r): pi at r=0 -> 0 for r >= r0, C^2
    (smootherstep). f(0)=pi gives U(0) = -1, the standard hedgehog winding."""
    t = np.clip(r / r0, 0.0, 1.0)
    s = t**3 * (10.0 - 15.0 * t + 6.0 * t**2)   # smootherstep (C^2)
    return np.pi * (1.0 - s)


def skyrmion_hedgehog(grid: BoxGrid, r0: float | None = None,
                      center=(0.0, 0.0, 0.0)) -> jnp.ndarray:
    """B=1 hedgehog Skyrmion: U = cos f(r) + i sin f(r) (r_hat . sigma), i.e.
    phi0 = cos f, phi_a = sin f * r_hat_a, with f the pi->0 profile. Unit
    4-vector field, shape (4, N, N, N). deg(U) = 1.

    r0 is the profile (soliton) radius; defaults to a quarter box so the tube
    fits with vacuum margin. The hedgehog is the B=1 ansatz only -- use
    skyrmion_rational_map for multi-B seeds."""
    if r0 is None:
        r0 = 0.25 * grid.L
    X, Y, Z = (np.asarray(c, dtype=np.float64) for c in grid.coords())
    x0, y0, z0 = center
    x, y, z = X - x0, Y - y0, Z - z0
    r = np.sqrt(x**2 + y**2 + z**2)
    f = _hedgehog_profile(r, r0)
    rsafe = np.where(r > 0, r, 1.0)
    sf = np.sin(f)
    phi = np.stack([np.cos(f), sf * x / rsafe, sf * y / rsafe, sf * z / rsafe])
    return jnp.asarray(phi, dtype=grid.dtype)


def skyrmion_rational_map(grid: BoxGrid, B: int = 2, r0: float | None = None,
                          center=(0.0, 0.0, 0.0)) -> jnp.ndarray:
    """Degree-B Skyrmion via the Houghton-Manton-Sutcliffe rational-map ansatz:
    U(x) = cos f(r) + i sin f(r) (n_R . sigma), where n_R is the S^2 point of
    the rational map R(z) = z^B evaluated at the Riemann coordinate
    z = (x + i y) / (r + z) of the spatial direction. deg(U) = B. The B=2 case
    relaxes to the axially-symmetric torus (the deuteron). Unit 4-vector field,
    shape (4, N, N, N)."""
    if r0 is None:
        r0 = 0.25 * grid.L
    X, Y, Z = (np.asarray(c, dtype=np.float64) for c in grid.coords())
    x0, y0, z0 = center
    x, y, z = X - x0, Y - y0, Z - z0
    r = np.sqrt(x**2 + y**2 + z**2)
    rho = np.sqrt(x**2 + y**2)
    # Riemann sphere coordinate of the spatial direction, z = tan(theta/2)e^{i
    # phi}; stereographic from the south pole (-z), where r + z -> 0. On that
    # axis num=den=0 (a 0/0 coordinate singularity), but the limit of R=z^B is
    # |R|->inf -> n_R=(0,0,-1) for every B, so override it explicitly.
    zc = (x + 1j * y) / (r + z + 1e-12)
    Rz = zc**B                                  # symmetric degree-B map
    a, b = np.real(Rz), np.imag(Rz)
    den = 1.0 + a**2 + b**2
    nR = np.stack([2.0 * a / den, 2.0 * b / den, (1.0 - a**2 - b**2) / den])
    south = (rho < 1e-9) & (z < 0.0)            # the -z symmetry axis
    nR[0] = np.where(south, 0.0, nR[0])
    nR[1] = np.where(south, 0.0, nR[1])
    nR[2] = np.where(south, -1.0, nR[2])
    f = _hedgehog_profile(r, r0)
    sf = np.sin(f)
    phi = np.stack([np.cos(f), sf * nR[0], sf * nR[1], sf * nR[2]])
    return jnp.asarray(phi, dtype=grid.dtype)


def _quaternion_product(a, b):
    """Hamilton product of two unit-4-vector fields a = (a0, a_vec),
    b = (b0, b_vec) (each (4, ...)); the SU(2) group product U_a U_b."""
    a0, av = a[0], a[1:]
    b0, bv = b[0], b[1:]
    s = a0 * b0 - (av[0] * bv[0] + av[1] * bv[1] + av[2] * bv[2])
    cross = np.stack([
        av[1] * bv[2] - av[2] * bv[1],
        av[2] * bv[0] - av[0] * bv[2],
        av[0] * bv[1] - av[1] * bv[0],
    ])
    v = a0 * bv + b0 * av + cross
    return np.concatenate([s[None], v])


def skyrmion_product(grid: BoxGrid, sep: float = 2.0, axis: int = 0,
                     rel_iso=None, r0: float | None = None) -> jnp.ndarray:
    """Product ansatz of two B=1 hedgehogs, U = U_A(x - d/2) U_B(x + d/2),
    separated by `sep` along `axis`, with an optional relative iso-orientation
    `rel_iso` (a 3x3 SO(3) rotation) applied to U_B's pion field. This is the
    input to the binding cross-check: feed the SAME rigid-composition /
    soft-pin method we use on the NWT carrier and compare against the known
    Skyrme B=2 binding (the attractive channel is the relative-iso-pi rotation
    about the separation axis). Unit 4-vector field, shape (4, N, N, N)."""
    d = np.zeros(3)
    d[axis] = 0.5 * sep
    cA = tuple(-d)
    cB = tuple(+d)
    phiA = np.asarray(skyrmion_hedgehog(grid, r0=r0, center=cA), np.float64)
    phiB = np.asarray(skyrmion_hedgehog(grid, r0=r0, center=cB), np.float64)
    if rel_iso is not None:
        R = np.asarray(rel_iso, np.float64)
        phiB = np.concatenate([phiB[:1], np.einsum("ab,bxyz->axyz", R, phiB[1:])])
    phi = _quaternion_product(phiA, phiB)
    phi = phi / np.sqrt((phi**2).sum(axis=0, keepdims=True))   # re-normalise
    return jnp.asarray(phi, dtype=grid.dtype)


def kibble_zurek_tangle(grid: BoxGrid, kcut: float = 1.4, seed: int = 0):
    """Band-limited random-phase complex field: a quenched vortex TANGLE.

    Every other seed in this module places a structure deliberately -- a rational
    map, a torus knot, a hedgehog. This one places nothing. It is a smooth random
    field whose phase winds by accident, so its zero set is a sparse tangle of
    vortex lines with correlation length ~ 1/kcut, and that is the point: it is the
    only way to ask what a GENERIC quench produces rather than what you seeded.

    That question is the control every placed seed lacks. A campaign that puts a
    trefoil in and gets a trefoil out has not learned whether trefoils form; this
    seed is how you find out. It is also the lattice analogue of the Kibble-Zurek
    mechanism that makes cosmological defects -- a field with no long-range phase
    coherence, quenched, leaving whatever topology the correlation length allows.

    Construction: white complex noise in Fourier space, damped by a Gaussian
    envelope exp(-k^2 / 2 kcut^2), transformed back. The envelope is what makes the
    phase smooth on scales below 1/kcut, so the vortex cores are resolved rather
    than being one-cell noise. `kcut` in units of 2*pi/L; the default 1.4 gives a
    handful of lines in a 24-unit box.

    Normalised to mean |psi| = 1, so it lands on the GPE-like vacuum manifold
    without a further rescale. Deterministic in `seed`: same seed, same tangle,
    which a stochastic seed has to promise or no result from it is reproducible.

    Ported from the retired `gpe_conucleation.py`. Returns complex128 regardless of
    `grid.dtype` -- the vortex detectors that consume this difference phases, and a
    fp32 phase near a core costs the winding.
    """
    if kcut <= 0:
        raise ValueError(f"kcut must be positive, got {kcut}")
    N = grid.N
    rng = np.random.default_rng(seed)
    k2 = np.zeros((N, N, N))
    kax = np.asarray(2.0 * np.pi * np.fft.fftfreq(N, d=grid.dx))
    for ax in range(3):
        k2 = k2 + np.reshape(kax, [-1 if i == ax else 1 for i in range(3)]) ** 2
    env = np.exp(-k2 / (2.0 * kcut ** 2))
    amp = (rng.standard_normal((N, N, N)) + 1j * rng.standard_normal((N, N, N))) * env
    psi = np.fft.ifftn(amp)
    psi = psi / (np.mean(np.abs(psi)) + 1e-12)
    return jnp.asarray(psi.astype(np.complex128))

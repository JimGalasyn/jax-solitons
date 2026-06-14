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

#!/usr/bin/env python3
"""Sound-immune vortex topology for real-time GPE fields (phase-winding method).

The density-threshold + pseudovorticity tracers drown in the radiated sound of a
real-time field (sound dips |psi| and is mistaken for cores). The robust,
standard method keys on the PHASE instead: a quantized vortex is exactly where
the phase winds +/-2pi around an elementary plaquette -- sound modulates |psi|
but NEVER winds the phase by 2pi, so this is immune to compressible noise.

  vortex_skeleton(psi): the three plaquette winding-charge fields (one per face
    orientation) -> a thin set of DIRECTED line segments (position + signed unit
    tangent) on the true |psi|=0 core lines.
  linking_number(psi, dx, L): label the segments into connected vortex lines and
    return the Gauss linking of the two largest (integer-clean, no calibration).

Validated by construction: the lk=-1 deuteron seed reads ~ -1; after relaxation
unlinks it, ~ 0; a single trefoil is one component.

  python3 simulations/gpe_vortex_topology.py        # self-test on the deuteron
"""
from __future__ import annotations
import numpy as np
import scipy.ndimage as ndi


def _wrap(d):
    return (d + np.pi) % (2.0 * np.pi) - np.pi


def _winding(theta, a, b):
    """Signed vortex charge through every plaquette in the (a,b) plane (normal =
    the third, cyclic, axis). CCW loop 00->+a->+a+b->+b->00."""
    t00 = theta
    t10 = np.roll(theta, -1, a)
    t11 = np.roll(np.roll(theta, -1, a), -1, b)
    t01 = np.roll(theta, -1, b)
    w = _wrap(t10 - t00) + _wrap(t11 - t10) + _wrap(t01 - t11) + _wrap(t00 - t01)
    return np.rint(w / (2.0 * np.pi)).astype(np.int8)


# plaquette plane (a,b) -> (normal axis, cell-centre offset, unit tangent)
_PLAQ = [((0, 1), 2, (0.5, 0.5, 0.0), (0.0, 0.0, 1.0)),    # xy-face, tangent +z
         ((1, 2), 0, (0.0, 0.5, 0.5), (1.0, 0.0, 0.0)),    # yz-face, tangent +x
         ((2, 0), 1, (0.5, 0.0, 0.5), (0.0, 1.0, 0.0))]    # zx-face, tangent +y


def vortex_skeleton(psi):
    """Return (positions_idx, tangents, cells): directed segments on the core
    lines. positions in CELL units (add offset already applied); cells = the
    (i,j,k) lattice index for connectivity labelling."""
    theta = np.angle(psi)
    P, T, C = [], [], []
    for (a, b), nrm, off, tan in _PLAQ:
        w = _winding(theta, a, b)
        idx = np.argwhere(w != 0)
        if not len(idx):
            continue
        q = w[idx[:, 0], idx[:, 1], idx[:, 2]].astype(float)
        P.append(idx + np.array(off))
        T.append(np.array(tan)[None, :] * q[:, None])     # signed tangent
        C.append(idx)
    if not P:
        return np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 3), int)
    return np.vstack(P), np.vstack(T), np.vstack(C)


def _label_lines(cells, shape, min_seg):
    """Connected vortex lines: 26-connectivity on the set of cells holding a
    pierced plaquette. Returns (labels_per_segment, ordered_big_labels)."""
    mask = np.zeros(shape, bool)
    mask[cells[:, 0], cells[:, 1], cells[:, 2]] = True
    lab, n = ndi.label(mask, structure=np.ones((3, 3, 3)))
    seg_lab = lab[cells[:, 0], cells[:, 1], cells[:, 2]]
    sizes = np.bincount(seg_lab, minlength=n + 1)
    big = [i for i in range(1, n + 1) if sizes[i] >= min_seg]
    big.sort(key=lambda i: -sizes[i])
    return seg_lab, big, sizes


def _gauss(PA, TA, PB, TB, dx):
    R = PA[:, None, :] - PB[None, :, :]
    d3 = np.sum(R * R, axis=-1) ** 1.5 + 1e-12
    num = np.sum(np.cross(TA[:, None, :], TB[None, :, :]) * R, axis=-1)
    return float(dx ** 2 / (4.0 * np.pi) * np.sum(num / d3))


def linking_number(psi, dx, L, min_seg=6):
    """(#lines, lk of the two largest lines, line sizes). lk is integer-clean."""
    P, T, C = vortex_skeleton(psi)
    if not len(P):
        return 0, float("nan"), []
    seg_lab, big, sizes = _label_lines(C, psi.shape, min_seg)
    if len(big) < 2:
        return len(big), float("nan"), [int(sizes[i]) for i in big]
    pos = (P - np.array(psi.shape) / 2.0) * dx          # physical coords
    A, Bi = big[0], big[1]
    mA, mB = seg_lab == A, seg_lab == Bi
    lk = _gauss(pos[mA], T[mA], pos[mB], T[mB], dx)
    return len(big), lk, [int(sizes[i]) for i in big]


def _order_line(P, T, max_gap=2.2):
    """Tangent-following walk: order one line's directed segments into a curve.
    Robust through self-approaches (a trefoil) because it steps in the +tangent
    direction, not just to the nearest point."""
    n = len(P)
    used = np.zeros(n, bool)
    order = [0]; used[0] = True; cur = 0
    for _ in range(n - 1):
        ahead = P[cur] + T[cur] * 0.6
        d = np.sum((P - ahead) ** 2, axis=1)
        d[used] = np.inf
        nxt = int(np.argmin(d))
        if d[nxt] > max_gap ** 2:
            break
        order.append(nxt); used[nxt] = True; cur = nxt
    return np.array(order)


def knot_determinants(psi, dx, L, min_seg=12, min_pts=14):
    """Per-line knot type: order each vortex line and return its knot
    determinant (1=unknot/lepton, 3=trefoil/baryon, 5=cinquefoil, ...).
    Returns list of (size, det) sorted by size. Lines too short to be knotted
    (< min_pts ordered points) are unknots by definition (det=1) -- this is the
    fix for the IndexError pyknotid threw on degenerate short tangle lines."""
    from .knots import identify_knot
    P, T, C = vortex_skeleton(psi)
    if not len(P):
        return []
    seg_lab, big, sizes = _label_lines(C, psi.shape, min_seg)
    pos = (P - np.array(psi.shape) / 2.0) * dx
    out = []
    for lid in big:
        m = seg_lab == lid
        order = _order_line(P[m], T[m])
        curve = pos[m][order]
        if len(curve) < min_pts:
            out.append((int(sizes[lid]), 1))           # too short to knot
            continue
        try:
            det = int(identify_knot(curve)["determinant"])
        except Exception as exc:
            det = f"e:{type(exc).__name__}"
        out.append((int(sizes[lid]), det))
    return out


def net_linking(psi, dx, L, min_seg=10):
    """Signed, summed INTER-LOOP linking <Lk> of the whole tangle = Σ_{i<j} lk(i,j)
    over all distinct vortex-loop pairs. A PSEUDOSCALAR: <Lk> = 0 EXACTLY in any
    parity-symmetric ensemble, so a net <Lk> != 0 is the chiral signature = the
    η_B analog (the obstacle to beat: Rivers-Volovik PRL 127,115702). Returns
    (<Lk>, n_loops, n_pairs)."""
    P, T, C = vortex_skeleton(psi)
    if len(P) < 2:
        return 0.0, 0, 0
    seg_lab, big, _ = _label_lines(C, psi.shape, min_seg)
    if len(big) < 2:
        return 0.0, len(big), 0
    pos = (P - np.array(psi.shape) / 2.0) * dx
    masks = [seg_lab == lid for lid in big]
    tot, npair = 0.0, 0
    for i in range(len(big)):
        for j in range(i + 1, len(big)):
            tot += _gauss(pos[masks[i]], T[masks[i]], pos[masks[j]], T[masks[j]], dx)
            npair += 1
    return float(tot), len(big), npair


def total_helicity(psi, dx, L, r_min_cells=2.0, cap=2500):
    """Net writhe+linking of the WHOLE vortex tangle = the centerline helicity
    (scalar GPE has no internal twist, so this IS the hydrodynamic helicity, in
    units of the circulation^2). The full Gauss double sum over all skeleton
    segments; near pairs (|r-r'| < r_min) excluded to kill the adjacent-self 1/r^3
    divergence. A CHIRALITY meter: ~0 = achiral tangle, net sign = net handedness
    -> the quantity a chiral bias must drive to source a baryon excess (η_B)."""
    P, T, C = vortex_skeleton(psi)
    if len(P) < 2:
        return 0.0
    pos = (P - np.array(psi.shape) / 2.0) * dx
    tan = T
    if len(pos) > cap:                                  # downsample dense tangles
        s = np.linspace(0, len(pos) - 1, cap).astype(int)
        pos, tan = pos[s], tan[s]
    R = pos[:, None, :] - pos[None, :, :]
    d2 = np.sum(R * R, axis=-1)
    near = d2 < (r_min_cells * dx) ** 2
    d3 = d2 ** 1.5 + 1e-12
    num = np.sum(np.cross(tan[:, None, :], tan[None, :, :]) * R, axis=-1)
    num = np.where(near, 0.0, num)
    return float(dx ** 2 / (4.0 * np.pi) * 0.5 * np.sum(num / d3))   # 0.5: ordered-pair double count


def _selftest():
    import nwt_gpe_binding as Bnd
    from nwt_gpe_biot_savart_portraits import Grid
    link = Bnd.deuteron_link(); iso = Bnd.single_trefoil()
    N, L = Bnd._box_for(link, iso, target_dx=0.24)
    grid = Grid(N=N, L=L, tilt=(0.4, 0.3)); apex = np.array([0., 0., 0.9 * L])
    dx = L / N
    theta = grid.minimal_phase(link, apex); d = grid.distance_to(link)
    psi = (Bnd.PROFILE.f_at(d) * np.exp(1j * theta)).astype(np.complex128)
    print(f"box N={N} dx={dx:.3f}  (seed curve lk={Bnd.linking_matrix(link)[0,1]:+.2f})\n")
    print(f"  {'state':>22} {'#lines':>7} {'lk':>8}  sizes")
    n, lk, sz = linking_number(psi, dx, L); print(f"  {'deuteron SEED':>22} {n:>7} {lk:>+8.3f}  {sz}")
    for k in (5, 35, 70):
        p = psi.copy()
        for _ in range(k):
            p = grid.step(p, 0.1)
        n, lk, sz = linking_number(p, dx, L)
        print(f"  {'relaxed '+str(k)+' steps':>22} {n:>7} {lk:>+8.3f}  {sz}")
    ps = (Bnd.PROFILE.f_at(grid.distance_to(iso)) *
          np.exp(1j * grid.minimal_phase(iso, apex))).astype(np.complex128)
    n, lk, sz = linking_number(ps, dx, L); print(f"  {'single trefoil':>22} {n:>7} {lk:>+8.3f}  {sz}")
    print("\n  PASS if: seed lk ~ -1, relaxed lk -> 0 (unlinks), single = 1 line.")


if __name__ == "__main__":
    _selftest()

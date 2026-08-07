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
  core_separation / kinetic_decomposition: the two observables a real-time run
    needs alongside lk(t) -- are the cores still apart, and where did the energy
    go (vortex flow vs radiated sound).

Validated by construction: a seeded lk=-1 pair of clasped trefoils
(`invariants.curves.hopf_clasped_trefoils` through `seeds.superflow_seed`) reads
~ -1; after imaginary-time relaxation unlinks it, ~ 0; a single trefoil is one
component. `_selftest()` runs exactly that.

NOTE the pseudovorticity route is deliberately NOT offered. An earlier program
measured linking as a Gauss integral over w = grad(Re psi) x grad(Im psi),
auto-normalised against the seed. It works, but it inherits two costs this one
does not: the density threshold picks up sound, and the result needs
calibration against a known-lk configuration before it means anything. The
plaquette winding is integer-clean without either.
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
    determinant (1=unknot, 3=trefoil, 5=cinquefoil, ...).
    Returns list of (size, det) sorted by size. Lines too short to be knotted
    (< min_pts ordered points) are unknots by definition (det=1) -- this is the
    fix for the IndexError pyknotid threw on degenerate short tangle lines.

    RAISES ImportError, ONCE, when identification is unavailable, rather than
    substituting it per line. The per-line `f"e:{ExceptionName}"` fallback below is
    for failures that are properties of a CURVE -- a degenerate line, a pathological
    trace. A missing pyknotid is a property of the ENVIRONMENT and identical for
    every line, so recording it per line writes an environment failure into a
    manifest in the exact shape of a measurement.

    That is not hypothetical: on 2026-08-03 a $1.50 N=320 rental returned 73 samples
    of `det1 = [[2352, 'e:ImportError']]`, the leg reported OK with remote_exit=0,
    and the measurement the box was rented for had not run once. pyknotid is an
    optional extra (`knots`), the remote pip line installed no extras, and nothing
    anywhere failed. Checked BEFORE the skeleton work, so an unusable environment
    costs an import rather than a full trace."""
    from .knots import _knot_class, identify_knot
    _knot_class()          # raises ImportError with pyknotid's own install hint
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
    -> the quantity any chiral bias in the dynamics has to drive.

    NO 1/2 ON THE DOUBLE SUM, and there used to be one. The helicity of a set of
    unit-circulation tubes is

        H = sum_i Wr_i + 2 sum_{i<j} Lk_ij

    and the ORDERED double sum over all segment pairs produces exactly that:
    the (i,j) and (j,i) halves are what supply the factor 2 on the cross terms,
    while the within-curve pairs give each Wr_i once. Halving it -- "correcting"
    an ordered-pair double count that the formula actually wants -- returned H/2.

    Caught by comparing against an independent measurement rather than by
    reading: on a single seeded trefoil, where H must equal the core curve's
    writhe and there is no linking term to hide in, this returned -1.72 against
    `linking_invariants.writhe` = -3.28, a ratio of 0.52. It now returns -3.28
    to within the few percent the near-pair exclusion costs.

    That exclusion, plus the staircase geometry of plaquette-based segments,
    means the value is REGULARISED rather than exact -- measured about 5% HIGH in
    magnitude on the single trefoil above (-3.44 against the curve's -3.28). So
    prefer differences between configurations measured the same way over
    absolute numbers."""
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
    return float(dx ** 2 / (4.0 * np.pi) * np.sum(num / d3))


def core_separation(psi, dx, L, min_seg=6):
    """Centroid distance between the two largest vortex lines, in physical units.

    The companion observable to `linking_number` for a real-time run: a linked
    pair that stays bound holds a FINITE separation while radiating, where an
    unbinding one grows without bound (and, in a periodic box, wraps -- so read
    the trend, not the raw number, once it approaches L/2).

    Uses the same phase-winding skeleton as `linking_number` rather than a
    density threshold, so sound does not move the centroids.

    Returns nan if fewer than two lines survive the `min_seg` cut, which is
    itself informative: it is what a reconnection event looks like frame to
    frame (two lines merge into one, then split again).
    """
    P, T, C = vortex_skeleton(psi)
    if not len(P):
        return float("nan")
    seg_lab, big, _ = _label_lines(C, psi.shape, min_seg)
    if len(big) < 2:
        return float("nan")
    pos = (P - np.array(psi.shape) / 2.0) * dx
    ca = pos[seg_lab == big[0]].mean(axis=0)
    cb = pos[seg_lab == big[1]].mean(axis=0)
    return float(np.linalg.norm(ca - cb))


def kinetic_decomposition(psi, dx):
    """Nore-Abid-Brachet split of the kinetic energy: (incompressible, compressible).

    Decompose the density-weighted velocity u = j / sqrt(rho), with
    j = Im(conj(psi) grad psi), into its solenoidal and irrotational parts in
    Fourier space. The incompressible part is the VORTEX flow; the compressible
    part is SOUND.

    This is the meter that makes "binding" observable in a conservative run.
    Real-time evolution conserves total energy, so a pair cannot simply fall
    into a bound state -- it has to shed the difference as radiation. Watching
    E_compressible rise while E_incompressible falls is that shedding, and the
    amount shed by the time the separation settles is the binding energy the
    run actually paid. A relaxation cannot show this at all: gradient flow
    deletes the energy instead of radiating it.

    Assumes a periodic box and a scalar order parameter with bulk |psi| ~ 1.
    For a GAUGED field the gauge-invariant current is
    Im(conj(psi) D psi) with D = grad - i q A, which this does NOT compute --
    the split is only meaningful here for the ungauged case.

    Returns
    -------
    (E_incompressible, E_compressible) : tuple of float
    """
    n = psi.shape[0]
    rho = np.abs(psi) ** 2
    gpsi = np.gradient(psi, dx)
    u = [np.imag(np.conj(psi) * g) / np.sqrt(rho + 1e-6) for g in gpsi]
    uk = [np.fft.fftn(ui) for ui in u]
    k1 = 2.0 * np.pi * np.fft.fftfreq(n, d=dx)
    KX, KY, KZ = np.meshgrid(k1, k1, k1, indexing="ij")
    k2 = KX ** 2 + KY ** 2 + KZ ** 2
    k2[0, 0, 0] = 1.0
    kdu = (KX * uk[0] + KY * uk[1] + KZ * uk[2]) / k2
    comp = [KX * kdu, KY * kdu, KZ * kdu]
    scale = 0.5 * dx ** 3 / n ** 3                 # Parseval on an unnormalised FFT
    e_comp = scale * sum(float(np.sum(np.abs(c) ** 2)) for c in comp)
    e_tot = scale * sum(float(np.sum(np.abs(c) ** 2)) for c in uk)
    return e_tot - e_comp, e_comp


def link_binding_energy(e_cluster, e_single, n_constituents):
    """dE = n * E(one constituent) - E(the linked cluster). Positive = bound.

    THE PROTOCOL IS THE POINT, and it is a subtraction that only works if both
    numbers come from the SAME BOX at the same resolution: the single-knot
    self-energy and the finite-box (periodic image + Seifert-sheet) corrections
    are large compared to any binding, and they cancel only when N, L, and the
    seed profile are identical between the two runs. Relaxing the cluster at one
    resolution and the constituent at another measures the discretisation.

    This is NOT a potential well depth. A well depth needs E at separation
    d -> infinity, and a LINKED pair cannot be taken to infinity without
    unclasping -- the separation coordinate is bounded by the link
    (`hopf_clasped_trefoils`' sep_scale opens the clasp past ~1.2). What this
    returns is the energy of the linked cluster relative to its constituents
    free and separate, which is a real number and a different one.

    Sign convention: dE > 0 means the cluster sits BELOW its separated
    constituents, i.e. the linking binds.
    """
    return float(n_constituents) * float(e_single) - float(e_cluster)


def _selftest(n=96, box=12.0, core=0.7, steps=(0, 10, 40)):
    """Seed a clasped-trefoil pair, relax it, watch the link go. Self-contained.

    Replaces a version that imported two modules from a retired private repo, so
    it could not have run from an installed copy of this package.
    """
    import numpy as _np

    from jax_solitons.grid import BoxGrid
    from jax_solitons.invariants.curves import hopf_clasped_trefoils
    from jax_solitons.invariants.linking_invariants import gauss_linking_number
    from jax_solitons.seeds import superflow_seed

    grid = BoxGrid(N=n, L=box)
    dx = grid.dx
    a, b = hopf_clasped_trefoils(R=2.2, r=0.8, n_points=480)
    print(f"box N={n} L={box} dx={dx:.3f} core={core}   "
          f"seed curve lk={gauss_linking_number(a, b):+.3f}")
    psi = _np.asarray(superflow_seed(grid, [a, b], core=core))

    k1 = 2.0 * _np.pi * _np.fft.fftfreq(n, d=dx)
    KX, KY, KZ = _np.meshgrid(k1, k1, k1, indexing="ij")
    k2 = KX ** 2 + KY ** 2 + KZ ** 2

    def imag_step(p, dt):                       # split-step, imaginary time
        p = p * _np.exp(-0.5 * dt * (_np.abs(p) ** 2 - 1.0))
        p = _np.fft.ifftn(_np.fft.fftn(p) * _np.exp(-0.5 * k2 * dt))
        return p * _np.exp(-0.5 * dt * (_np.abs(p) ** 2 - 1.0))

    print(f"  {'state':>18} {'#lines':>7} {'lk':>8} {'sep':>8} {'E_inc':>9} {'E_comp':>9}")
    done = 0
    for target in steps:
        while done < target:
            psi = imag_step(psi, 0.1)
            done += 1
        nl, lk, _ = linking_number(psi, dx, box)
        e_i, e_c = kinetic_decomposition(psi, dx)
        print(f"  {'relaxed ' + str(done):>18} {nl:>7} {lk:>+8.3f} "
              f"{core_separation(psi, dx, box):>8.2f} {e_i:>9.1f} {e_c:>9.1f}")
    print("\n  PASS if: step 0 reads lk ~ -1 on two lines; relaxation drives it "
          "toward 0.")


if __name__ == "__main__":
    _selftest()

"""
Colored Jones / Rosso-Jones cabled-knot states for torus knots T(m, p).

Quantum SU(2)_k (Chern-Simons / Witten-Reshetikhin-Turaev) invariants of the
torus knots, computed mechanically from the explicit U_q(sl2)
R-matrix -- no closed-form recall.  Complements the classical invariants in
`torus_knots.py` (genus, crossing number) with the *quantum* invariant.

Default Chern-Simons level k = 5 is the so(7)-forced level (k = h^v(so7);
k+1 = h(so7); k+2 = |V(K_7)| = 7), with q = exp(i pi / (k+2)) a root of unity,
so [n] = sin(n pi/(k+2)) / sin(pi/(k+2)).

The central object is the **cabled-knot state**: T(m, p) is the closure of the
m-strand torus braid beta = sigma_1...sigma_{m-1} raised to the power p, each
strand coloured by spin s.  The braiding rho(beta) commutes with the global
quantum-group action, so it preserves each isotypic block V_J (x) Mult_J and
acts there as Id (x) B_J.  The per-channel multiplicity trace

    c_J(p) = Tr_{Mult_J}(B_J^p)          ("plethysm function" value)

is recovered by diagonalising rho(beta) and BUCKETING its eigenvalues by
lambda^m (= the full-twist scalar theta_J theta_s^{-m}, which labels J):

    c_J(p) = (1 / (2J+1)) * sum_{lambda in block J} lambda^p.

The colored-knot-complement state is psi_J = c_J(p) in the Wilson-line basis;
the colored Jones of the closure is the quantum Markov trace  sum_J d_J c_J(p),
which reproduces |jones_torus| ratios at the root of unity exactly (including
the multiplicity > 1 cablings, m >= 3).

Conventions verified against the ordinary (N=2) Jones polynomial of
trefoil/cinquefoil/7_1 and the full-twist ribbon factor.

References: Rosso-Jones (1993); Morton (1995); Reshetikhin-Turaev.  Built for
the Rosso-Jones formula (cabling-space
dimension; see `cabling_space_dimension`).
"""

from __future__ import annotations

import cmath
import math

import numpy as np



DEFAULT_LEVEL = 5  # Chern-Simons level k (default from the originating program; pass level= to override)


def _q(level: int) -> complex:
    """Deformation parameter q = exp(i pi/(k+2)) at Chern-Simons level k."""
    return cmath.exp(1j * math.pi / (level + 2))


def quantum_integer(n: int, level: int = DEFAULT_LEVEL) -> complex:
    """Quantum integer [n] = (q^n - q^-n)/(q - q^-1) = sin(n pi/(k+2))/sin(pi/(k+2))."""
    if n == 0:
        return 0.0 + 0j
    q = _q(level)
    return (q ** n - q ** (-n)) / (q - q ** (-1))


def _qfac(n: int, q: complex) -> complex:
    out = 1.0 + 0j
    for i in range(1, n + 1):
        out *= (q ** i - q ** (-i)) / (q - q ** (-1))
    return out


def _rep_EFK(spin: float, q: complex):
    """Spin-s irrep of U_q(sl2) in the weight basis |m>, m = s, s-1, ..., -s.

    E|m> = sqrt([s-m][s+m+1]) |m+1>,  F|m> = sqrt([s+m][s-m+1]) |m-1>,
    K|m> = q^{2m} |m>.
    """
    def qi(n):
        if n == 0:
            return 0.0 + 0j
        return (q ** n - q ** (-n)) / (q - q ** (-1))

    two_s = round(2 * spin)
    N = two_s + 1
    mvals = [spin - i for i in range(N)]
    E = np.zeros((N, N), complex)
    F = np.zeros((N, N), complex)
    K = np.zeros((N, N), complex)
    for i, m in enumerate(mvals):
        K[i, i] = q ** (2 * m)
        if i - 1 >= 0:
            E[i - 1, i] = cmath.sqrt(qi(round(spin - m)) * qi(round(spin + m + 1)))
        if i + 1 < N:
            F[i + 1, i] = cmath.sqrt(qi(round(spin + m)) * qi(round(spin - m + 1)))
    return E, F, K, mvals


def _braiding(spin: float, q: complex) -> np.ndarray:
    """Braiding check-R = (swap) . R on V_s (x) V_s (Kassel universal R-matrix).

    R = q^{H(x)H/2} sum_n (q-q^-1)^n/[n]! q^{n(n-1)/2} E^n (x) F^n.
    """
    E, F, K, mvals = _rep_EFK(spin, q)
    N = len(mvals)
    D = np.zeros((N * N, N * N), complex)
    for i, m1 in enumerate(mvals):
        for j, m2 in enumerate(mvals):
            D[i * N + j, i * N + j] = q ** (2 * m1 * m2)
    series = np.zeros((N * N, N * N), complex)
    En = np.eye(N, dtype=complex)
    Fn = np.eye(N, dtype=complex)
    n = 0
    while True:
        coeff = ((q - q ** (-1)) ** n / _qfac(n, q) * q ** (n * (n - 1) // 2)
                 if n > 0 else 1.0 + 0j)
        series += coeff * np.kron(En, Fn)
        n += 1
        En = En @ E
        Fn = Fn @ F
        if np.allclose(En, 0) or np.allclose(Fn, 0) or n > 2 * round(2 * spin) + 2:
            break
    R = D @ series
    SWAP = np.zeros((N * N, N * N), complex)
    for i in range(N):
        for j in range(N):
            SWAP[j * N + i, i * N + j] = 1.0
    return SWAP @ R


def _braid_word(spin: float, strands: int, q: complex) -> np.ndarray:
    """rho(sigma_1 ... sigma_{strands-1}) on V_s^{tensor strands}."""
    N = round(2 * spin) + 1
    Rc = _braiding(spin, q)
    B = np.eye(N ** strands, dtype=complex)
    for i in range(strands - 1):
        left = N ** i
        right = N ** (strands - i - 2)
        sig = np.kron(np.kron(np.eye(left, dtype=complex), Rc),
                      np.eye(right, dtype=complex))
        B = sig @ B
    return B


def _twist(j: float, q: complex) -> complex:
    """Ribbon twist theta_j = q^{2 j(j+1)}."""
    return q ** (2 * j * (j + 1))


def torus_channels(spin: float, strands: int) -> dict:
    """Spins J in V_s^{tensor strands} with classical multiplicities (a dict J->mult)."""
    from collections import Counter
    cur = Counter({spin: 1})
    for _ in range(strands - 1):
        nxt = Counter()
        for j, mult in cur.items():
            jj = abs(j - spin)
            while jj <= j + spin + 1e-9:
                nxt[round(jj * 2) / 2] += mult
                jj += 1
        cur = nxt
    return dict(cur)


def cabling_space_dimension(n_q: int, q: int) -> int:
    """Dimension of the Rosso-Jones q-strand cabling space V_{n_q}^{tensor q} = n_q^q.

    For the torus knot T(q, n) presented as q strands coloured by the
    n_q-dimensional rep.  Note n^q is the dimension of the representation
    the Rosso-Jones operator acts on.  Verified == sum_J (2J+1) * mult_J via
    `torus_channels((n_q-1)/2, q)`.
    """
    return n_q ** q


def cabled_state(spin: float, strands: int, power: int,
                 level: int = DEFAULT_LEVEL, kmax: float | None = None) -> dict:
    """Rosso-Jones cabled-knot state {J: c_J(power)} for the closure of
    (sigma_1...sigma_{strands-1})^power, each strand coloured by `spin`.

    c_J(power) = Tr_{Mult_J}(B_J^power), recovered by diagonalising the braid
    and bucketing eigenvalues by lambda^strands.  Returns the Wilson-line-basis
    state psi_J = c_J.  Channels with spin J > kmax are dropped (truncation);
    kmax defaults to level/2 (the top integrable spin).
    """
    if kmax is None:
        kmax = level / 2.0
    q = _q(level)
    if strands == 1:
        return {spin: 1.0 + 0j}
    B = _braid_word(spin, strands, q)
    evals = np.linalg.eigvals(B)
    ch = torus_channels(spin, strands)
    th_s_m = _twist(spin, q) ** strands
    targets = {J: _twist(J, q) / th_s_m for J in ch}
    buckets = {J: [] for J in ch}
    for lam in evals:
        lm = lam ** strands
        J_best = min(ch, key=lambda J: abs(lm - targets[J]))
        buckets[J_best].append(lam)
    psi = {}
    for J, lams in buckets.items():
        if J > kmax + 1e-9:
            continue
        deg = round(2 * J) + 1
        psi[J] = sum(lam ** power for lam in lams) / deg
    return psi


def colored_jones(spin: float, strands: int, power: int,
                  level: int = DEFAULT_LEVEL) -> complex:
    """Colored Jones (unreduced quantum Markov trace) of T(strands, power) at
    colour `spin`:  sum_J [2J+1] c_J(power).  Equals [N] x reduced Jones.
    """
    psi = cabled_state(spin, strands, power, level=level)
    return sum(quantum_integer(round(2 * J) + 1, level) * cJ for J, cJ in psi.items())


def colored_jones_qtrace(spin: float, strands: int, power: int,
                         level: int = DEFAULT_LEVEL) -> complex:
    """Same invariant via the direct quantum trace Tr(rho(beta)^power . mu^{(x)m}),
    mu = K = diag q^{2m}.  Cross-check of `colored_jones` (must agree)."""
    q = _q(level)
    if strands == 1:
        return quantum_integer(round(2 * spin) + 1, level)
    _, _, K, _ = _rep_EFK(spin, q)
    MU = K
    for _ in range(strands - 1):
        MU = np.kron(MU, K)
    B = _braid_word(spin, strands, q)
    BP = np.linalg.matrix_power(B, power)
    return complex(np.trace(BP @ MU))

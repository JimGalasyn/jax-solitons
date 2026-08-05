#!/usr/bin/env python3
"""EHN real-time engine — the quench half of "one energy, two dynamics".

`relax.py` is gradient FLOW: `u ← u − α∇_u E`, first-order and dissipative. It
finds endpoints. It cannot show you a reconnection happening, and by DESIGN.md's
own note (P7, line 61) descent cannot create topology at all.

This module is the same EHN physics run FORWARD IN TIME:

    φ₁:  i∂_tφ₁ = (1−iγ)[ −½D²φ₁ + 2λ·c·φ₁ − κ|φ₂|²φ₁ ],  D = ∇−igA
    φ₂:  i∂_tφ₂ = (1−iγ)[ −½∇²φ₂ + 2λ·c·φ₂ − κ|φ₁|²φ₂ ]        (global)
    A :  ∂_tA = −E
    E :  ∂_tE = ∇×B − gJ₁ − 2μ₅B                  (resistive η, CME bias μ₅)
                                        c = |φ₁|²+|φ₂|²−1,  J₁ = Im(φ₁*∇φ₁) − gA|φ₁|²

Temporal gauge, transverse (A,E). The gauge sector is genuinely dynamical: A
carries its own conjugate momentum E rather than being relaxed to equilibrium,
so this radiates and rings where the relaxer would simply slide downhill.

WHAT IS SHARED, AND WHY IT MATTERS. The lock — ℒ₃'s `ρ = B·∂a` with a = arg φ₂
and ∂a taken WRAPPED — is imported from `ehn.energy`, the module extracted for
exactly this purpose ("the primitives common to the relax and quench engines").
The ℒ₃ force here is `−α_l3·∂E_L3/∂φ₂*` by AUTODIFF of the same energy the relax
engine descends, so the quench inherits the validated lock instead of a
hand-derived transcription of it. `rho_L3`'s own docstring fixes the one
difference between the engines: B is "relax: the constrained auxiliary field;
quench: ∇×A" — here there is no auxiliary B and no multiplier w, so B ≡ ∇×A.

THE A₀ FIELD IS NOT OPTIONAL, and that is a measured result rather than a
preference. The transverse (A,E) form carrying only the spatial axion term
−C(∇a×E) FAILED the R-C-LC-1 gate: without the A₀-coupled energy ½C∫ρ·s the
link is not protected. `step_locked` therefore keeps `s`, relaxed by the shared
`gauss_relax_s`. `step` retains the transverse form because the gate that
rejected it is worth being able to re-run.

CONSERVATIVE BY DEFAULT. γ = η = 0 here, unlike the NWT campaign engine this is
ported from (`simulations/ehn_cme_quench.py`, γ = 0.2–0.5, η = 0.5 — it was
built to CONDENSE a Kibble–Zurek tangle, where damping is the point). Energy
conservation is this module's correctness gate, so the dissipation has to be off
to have a gate at all, and `dH/H` over a fixed interval is what the tests check.

Two honest limits on that word:
  - the ℒ₃ lock is a DESCENT step on φ₂ (`−α_l3·∇E_L3`), so it is dissipative by
    construction. Conservation is a statement about the `C_l3 = 0` sector only,
    and the tests assert it there. This is the validated mechanism, not an
    approximation chosen here.
  - the matter sector is first-order (Schrödinger/GPE-like), not a second-order
    wave equation. That is the model EHN's own quench is written in, and the
    form the R-C-LC-1 gate validated; `models/nlkg.py` is where the relativistic
    second-order substrate lives if that is what you want.

Ported from null-worldtube-private `simulations/ehn_cme_quench.py` (`step`,
`step_locked`), which reaches its shared core through a sys.path insert into
`engine_dogfood/`. That file and `ehn.energy` are the same code, verified by
diff at port time; here it is a package import.
"""
from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from .energy import E_L3_electric, axion_grad, gauss_relax_s, rho_L3
from .knot_batch import (coulomb_project, curl, kvecs, magnetic_helicity,
                         skyrmion_number)

__all__ = ["kvecs", "curl", "coulomb_project", "magnetic_helicity",
           "skyrmion_number", "grad_spectral", "gauss_residual",
           "total_energy", "step", "step_locked", "evolve"]


def grad_spectral(psi, kv):
    """Spectral ∇ψ for a complex field. The quench's derivatives are spectral
    where relax's are finite-difference (P3: spectral is a single-device
    convenience, never load-bearing for scale) -- but `axion_grad` is roll-based
    in BOTH engines, because a wrapped phase difference only means anything
    between neighbouring sites."""
    KX, KY, KZ, _ = kv
    ph = jnp.fft.fftn(psi)
    return (jnp.fft.ifftn(1j * KX * ph),
            jnp.fft.ifftn(1j * KY * ph),
            jnp.fft.ifftn(1j * KZ * ph))


def _matter_kicks(Ax, Ay, Az, kv, g, gamma, lam, kappa):
    """The two matter kick functions, shared by `step` and `step_locked` so the
    bare and locked engines cannot drift apart in the sector they agree on."""
    def kick1(p1, p2):
        d1x, d1y, d1z = grad_spectral(p1, kv)
        A2 = Ax ** 2 + Ay ** 2 + Az ** 2
        Adg = Ax * d1x + Ay * d1y + Az * d1z
        c = jnp.abs(p1) ** 2 + jnp.abs(p2) ** 2 - 1.0
        return ((1.0 - 1j * gamma) * g * Adg
                - (1j + gamma) * (0.5 * g ** 2 * A2 * p1
                                  + 2.0 * lam * c * p1
                                  - kappa * jnp.abs(p2) ** 2 * p1))

    def kick2(p1, p2):
        c = jnp.abs(p1) ** 2 + jnp.abs(p2) ** 2 - 1.0
        return -(1j + gamma) * (2.0 * lam * c * p2
                                - kappa * jnp.abs(p1) ** 2 * p2)
    return kick1, kick2


def _matter_strang(phi1, phi2, kv, dt, gamma, kick1, kick2):
    """Strang split: kin½ · RK2 kick · kin½."""
    _, _, _, K2 = kv
    kin_half = jnp.exp(-(1j + gamma) * 0.25 * K2 * dt)
    phi1 = jnp.fft.ifftn(kin_half * jnp.fft.fftn(phi1))
    phi2 = jnp.fft.ifftn(kin_half * jnp.fft.fftn(phi2))
    p1m = phi1 + 0.5 * dt * kick1(phi1, phi2)
    p2m = phi2 + 0.5 * dt * kick2(phi1, phi2)
    phi1 = phi1 + dt * kick1(p1m, p2m)
    phi2 = phi2 + dt * kick2(p1m, p2m)
    phi1 = jnp.fft.ifftn(kin_half * jnp.fft.fftn(phi1))
    phi2 = jnp.fft.ifftn(kin_half * jnp.fft.fftn(phi2))
    return phi1, phi2


def _maxwell(phi1, Ax, Ay, Az, Ex, Ey, Ez, kv, dt, g, mu5, eta, extra=None):
    """Resistive leapfrog Maxwell + CME on the transverse (A,E) pair. `extra` is
    an optional (x,y,z) addend to ∂_tE -- the spatial axion term for `step`,
    absent for `step_locked`, which routes ℒ₃ through the A₀ energy instead."""
    _, _, _, K2 = kv
    Bx, By, Bz = curl(Ax, Ay, Az, kv)
    cBx, cBy, cBz = curl(Bx, By, Bz, kv)
    d1x, d1y, d1z = grad_spectral(phi1, kv)
    rho_m = jnp.abs(phi1) ** 2
    Jx = jnp.imag(jnp.conj(phi1) * d1x) - g * Ax * rho_m
    Jy = jnp.imag(jnp.conj(phi1) * d1y) - g * Ay * rho_m
    Jz = jnp.imag(jnp.conj(phi1) * d1z) - g * Az * rho_m
    ax, ay, az = extra if extra is not None else (0.0, 0.0, 0.0)
    damp = jnp.exp(-eta * K2 * dt)
    Ex = jnp.real(jnp.fft.ifftn(damp * jnp.fft.fftn(
        Ex + dt * (cBx - g * Jx - 2.0 * mu5 * Bx - ax))))
    Ey = jnp.real(jnp.fft.ifftn(damp * jnp.fft.fftn(
        Ey + dt * (cBy - g * Jy - 2.0 * mu5 * By - ay))))
    Ez = jnp.real(jnp.fft.ifftn(damp * jnp.fft.fftn(
        Ez + dt * (cBz - g * Jz - 2.0 * mu5 * Bz - az))))
    Ex, Ey, Ez = coulomb_project(Ex, Ey, Ez, kv)
    Ax, Ay, Az = coulomb_project(Ax - dt * Ex, Ay - dt * Ey, Az - dt * Ez, kv)
    return Ax, Ay, Az, Ex, Ey, Ez, (Bx, By, Bz)


@partial(jax.jit, static_argnums=(13,))
def step(phi1, phi2, Ax, Ay, Az, Ex, Ey, Ez, kv, dt, g, mu5, eta, agrad,
         gamma=0.0, lam=50.0, kappa=0.5, dx=1.0, C_l3=0.0, eps_a=1e-3):
    """One real-time step, TRANSVERSE form: the ℒ₃ lock (if C_l3 > 0) enters as
    axion electrodynamics, −C(∇a×E) in Ampère plus the reciprocal E·B phase
    back-reaction on φ₂.

    KEPT FOR THE GATE, NOT RECOMMENDED. This is the form the R-C-LC-1 triptych
    REJECTED: without the A₀-coupled ℒ₃ energy the link is not protected. Use
    `step_locked`. It stays because a rejected arm you can no longer run is a
    result you can no longer reproduce -- and with C_l3 = 0 this is exactly the
    bare census engine, which the gate needs as its control.
    """
    kick1, kick2 = _matter_kicks(Ax, Ay, Az, kv, g, gamma, lam, kappa)
    phi1, phi2 = _matter_strang(phi1, phi2, kv, dt, gamma, kick1, kick2)
    gax, gay, gaz = axion_grad(phi2, dx, eps_a, agrad)
    extra = (C_l3 * (gay * Ez - gaz * Ey),
             C_l3 * (gaz * Ex - gax * Ez),
             C_l3 * (gax * Ey - gay * Ex))
    Ax, Ay, Az, Ex, Ey, Ez, (Bx, By, Bz) = _maxwell(
        phi1, Ax, Ay, Az, Ex, Ey, Ez, kv, dt, g, mu5, eta, extra=extra)
    # C_l3 = 0 -> exp(0) = 1 -> φ₂ untouched, so the bare path is bit-for-bit.
    phi2 = phi2 * jnp.exp(1j * dt * C_l3 * (Ex * Bx + Ey * By + Ez * Bz))
    return phi1, phi2, Ax, Ay, Az, Ex, Ey, Ez


#      static: 14 = agrad (a string), 18 = n_s (a Python loop bound). Both are
#      structural rather than numeric -- get the index wrong and n_s arrives as a
#      tracer, which fails loudly at `range(n_s)` rather than silently.
@partial(jax.jit, static_argnums=(14, 18))
def step_locked(phi1, phi2, Ax, Ay, Az, Ex, Ey, Ez, s, kv, dt, g, mu5, eta,
                agrad, gamma=0.0, lam=50.0, kappa=0.5, n_s=1, dx=1.0,
                C_l3=0.0, eps_a=1e-3, beta=2e-3, alpha_l3=4e-4):
    """One real-time step with the ℒ₃/A₀ LOCK — the mechanism R-C-LC-1 requires.

    Bare EHN + CME dynamics, plus: an A₀/Gauss field `s` relaxed by the shared
    `gauss_relax_s`, and the ℒ₃ force on φ₂ taken as the validated relax descent
    `−α_l3·∂E_L3/∂φ₂*` — autodiff of the SAME energy `relax.py` descends, so the
    lock is inherited rather than re-derived.

    Returns the eight fields plus the updated `s`. Note `s` is NOT a dynamical
    field: A₀ is a constraint potential, solved (n_s relaxation sweeps) rather
    than evolved, which is why it has no conjugate momentum here.
    """
    kick1, kick2 = _matter_kicks(Ax, Ay, Az, kv, g, gamma, lam, kappa)
    phi1, phi2 = _matter_strang(phi1, phi2, kv, dt, gamma, kick1, kick2)
    Ax, Ay, Az, Ex, Ey, Ez, _ = _maxwell(
        phi1, Ax, Ay, Az, Ex, Ey, Ez, kv, dt, g, mu5, eta, extra=None)

    # ---- ℒ₃ / A₀ lock -----------------------------------------------------
    B = curl(Ax, Ay, Az, kv)                     # B on the UPDATED A
    p1sq, p2sq = jnp.abs(phi1) ** 2, jnp.abs(phi2) ** 2
    for _ in range(n_s):
        rho = rho_L3(phi2, B, dx, eps_a, agrad)
        s = gauss_relax_s(s, p1sq, p2sq, rho, dx, beta, C_l3, 1.0, 0.0)
    gr, gi = jax.grad(
        lambda a, b: E_L3_electric(a + 1j * b, B, s, dx, C_l3, eps_a, agrad),
        argnums=(0, 1))(jnp.real(phi2), jnp.imag(phi2))
    phi2 = phi2 - alpha_l3 * (gr + 1j * gi)
    return phi1, phi2, Ax, Ay, Az, Ex, Ey, Ez, s


def gauss_residual(phi1, Ax, Ay, Az, Ex, Ey, Ez, kv, g):
    """‖∇·E + gρ‖₂ / ‖gρ‖₂ — Gauss's law as a MONITORED residual.

    In temporal gauge with a transverse projection, ∇·E ≡ 0 by construction, so
    this reports how far the physical charge density is from the constraint the
    projection imposes. It is a diagnostic, not a correction: a residual that
    grows is the signal that the transverse treatment is being asked to carry
    charge dynamics it cannot represent.
    """
    KX, KY, KZ, _ = kv
    divE = jnp.real(jnp.fft.ifftn(1j * (KX * jnp.fft.fftn(Ex)
                                        + KY * jnp.fft.fftn(Ey)
                                        + KZ * jnp.fft.fftn(Ez))))
    src = g * jnp.abs(phi1) ** 2
    src = src - jnp.mean(src)              # only the neutral part is representable
    den = jnp.linalg.norm(src)
    return float(jnp.linalg.norm(divE + src) / jnp.where(den > 0, den, 1.0))


def total_energy(phi1, phi2, Ax, Ay, Az, Ex, Ey, Ez, kv, dx, lam, kappa,
                 g=1.0, components=False):
    """The conserved Hamiltonian OF THIS DYNAMICS:

        H = ∫ ½|D_iφ₁|² + ½|∇φ₂|² + λc² − κ|φ₁|²|φ₂|² + ½|B|² + ½|E|²  dx³

    DO NOT substitute `knot_batch.two_scalar_energy` here, even though it looks
    like the same energy and is the natural thing to reuse. It is EHN-normalised
    (arXiv:2407.11731 Eq.10): "NO ½ on the covariant scalar gradients", matching
    EHN's STATIC functional. This engine integrates `i∂_tφ = −½D²φ + …`, whose
    Hamiltonian carries the ½. Measured with the un-halved functional, a
    perfectly conservative run reports dH/H = −2.9e-2 over T = 0.04 and — the
    part that makes it dangerous — that figure is INDEPENDENT of dt, so it
    survives every convergence check and reads as physical dissipation. With the
    ½ restored the same run gives −8.4e-4 at dt = 2e-3 and falls as O(dt²).

    ½|E|² is included because E is a dynamical variable here; a static
    functional omits it and then appears to lose exactly the energy that the
    gauge sector is carrying.

    `components=True` returns the breakdown dict instead of the scalar.
    """
    d1 = grad_spectral(phi1, kv)
    d2 = grad_spectral(phi2, kv)
    A = (Ax, Ay, Az)
    e_grad1 = sum(jnp.abs(d1[i] - 1j * g * A[i] * phi1) ** 2 for i in range(3))
    e_grad2 = sum(jnp.abs(d2[i]) ** 2 for i in range(3))
    a1, a2 = jnp.abs(phi1) ** 2, jnp.abs(phi2) ** 2
    e_pot = lam * (a1 + a2 - 1.0) ** 2 - kappa * a1 * a2
    Bx, By, Bz = curl(Ax, Ay, Az, kv)
    V = dx ** 3
    comp = {"grad1": float(0.5 * jnp.sum(e_grad1) * V),
            "grad2": float(0.5 * jnp.sum(e_grad2) * V),
            "pot": float(jnp.sum(e_pot) * V),
            "mag": float(0.5 * jnp.sum(Bx ** 2 + By ** 2 + Bz ** 2) * V),
            "elec": float(0.5 * jnp.sum(Ex ** 2 + Ey ** 2 + Ez ** 2) * V)}
    comp["total"] = sum(comp.values())
    return comp if components else comp["total"]


def evolve(phi1, phi2, Ax, Ay, Az, Ex, Ey, Ez, kv, *, steps, dt, dx, g=1.0,
           mu5=0.0, eta=0.0, gamma=0.0, lam=50.0, kappa=0.5, C_l3=0.0,
           eps_a=1e-3, agrad="wrapped", locked=True, n_s=1, beta=2e-3,
           alpha_l3=4e-4, s=None, sample_every=0, observer=None):
    """Drive the engine for `steps`, optionally sampling diagnostics.

    `locked=True` (default) uses `step_locked` — the ℒ₃/A₀ mechanism the gate
    requires. `locked=False` selects the transverse form the gate rejected.

    Returns `(state, samples)` where state is the 8 fields plus `s`, and samples
    is a list of dicts (empty when `sample_every` is 0). Diagnostics are
    computed on the host and appended per P6 — small records, not held fields.
    """
    if s is None:
        s = jnp.zeros_like(jnp.real(phi1))
    samples = []

    def sample(n):
        rec = {"n": n,
               "E_total": total_energy(phi1, phi2, Ax, Ay, Az, Ex, Ey, Ez, kv,
                                       dx, lam, kappa, g=g),
               "Q": float(skyrmion_number(phi1, phi2, kv, dx)),
               "helicity": magnetic_helicity(Ax, Ay, Az, kv, dx),
               "gauss_res": gauss_residual(phi1, Ax, Ay, Az, Ex, Ey, Ez, kv, g)}
        if observer is not None:
            rec.update(observer(phi1, phi2, Ax, Ay, Az, Ex, Ey, Ez, s) or {})
        samples.append(rec)

    if sample_every:
        sample(0)
    for n in range(1, steps + 1):
        if locked:
            phi1, phi2, Ax, Ay, Az, Ex, Ey, Ez, s = step_locked(
                phi1, phi2, Ax, Ay, Az, Ex, Ey, Ez, s, kv, dt, g, mu5, eta,
                agrad, gamma, lam, kappa, n_s, dx, C_l3, eps_a, beta, alpha_l3)
        else:
            phi1, phi2, Ax, Ay, Az, Ex, Ey, Ez = step(
                phi1, phi2, Ax, Ay, Az, Ex, Ey, Ez, kv, dt, g, mu5, eta,
                agrad, gamma, lam, kappa, dx, C_l3, eps_a)
        if sample_every and n % sample_every == 0:
            sample(n)
    return (phi1, phi2, Ax, Ay, Az, Ex, Ey, Ez, s), samples

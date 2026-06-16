"""A 24-harmonic travelling wave on a ring -- "a church organ pipe, but circular".

A second extensibility probe (after examples/kuramoto_ring.py), prompted by Luke
Leighton on the d12rg list: can a ring carry the *full simultaneous harmonics* of
something as a coherent travelling standing wave? Yes -- and it pins down the
physics that makes it possible.

A scalar field phi(x) on a periodic ring, evolved by the 1D wave equation
    d^2 phi / dt^2 = c^2 d^2 phi / dx^2,
whose energy E = integral 1/2 (d_t phi)^2 + 1/2 c^2 (d_x phi)^2 the engine's
`Model` holds as one gradient `EnergyTerm` -- and `jax.grad(E)` IS the force, so
we never write the wave operator by hand. Seed a right-moving pulse built from
harmonics n = 1..24 and it travels once around the ring and **reforms** -- a
coherent travelling standing wave.

WHY it stays coherent (the squeak): the wave equation is **non-dispersive**, so
every harmonic moves at the SAME speed (omega_n = c k_n, proportional to n). All
24 stay phase-locked and the packet recurs with period L/c. Swap in a *dispersive*
law (omega_n ~ n^2, Schrodinger-like) and the harmonics dephase -- the packet
smears and never reforms. So "all harmonics at once, travelling together" is
exactly the non-dispersive ring; the demo shows both.

Extensibility verdict (same as the Kuramoto demo): the `Model` abstraction and the
autodiff force transfer unchanged; `BoxGrid` is used for its ring periodicity +
spacing (its 3D `coords()` ignored); the FFT-preconditioned / dx^3 steppers don't
transfer, so a ~10-line 1D velocity-Verlet brings its own time integration.

Run:  python examples/harmonic_ring.py
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp

from jax_solitons.grid import BoxGrid
from jax_solitons.model import Model

HARMONICS = tuple(range(1, 25))                          # the 1..24 modes, simultaneously


# --------------------------------------------------------------- the model ----
@dataclasses.dataclass(frozen=True)
class WaveGradientEnergy:
    """The gradient (potential) energy of a scalar wave on the ring, 1/2 c^2
    (d_x phi)^2 -- forward difference, `jnp.roll` for the periodic neighbour."""

    c: float = 1.0
    name: str = "wave_gradient"

    def __call__(self, phi, grid: BoxGrid):
        d = (jnp.roll(phi, -1) - phi) / grid.dx
        return 0.5 * self.c**2 * jnp.sum(d**2) * grid.dx


def wave_ring_model(c: float = 1.0) -> Model:
    return Model(name="wave_ring", terms=(WaveGradientEnergy(c=c),),
                 constraint=None, charges=())


def total_energy(phi, v, model: Model, grid: BoxGrid):
    """Kinetic + gradient -- the leapfrog invariant."""
    return 0.5 * jnp.sum(v**2) * grid.dx + model.energy(phi, grid)


# --------------------------------------------------- a 1D velocity-Verlet ------
def make_leapfrog(model: Model, grid: BoxGrid, dt: float):
    """1D velocity-Verlet for d^2 phi/dt^2 = -dE/dphi / dx (no constraint). The
    force is `jax.grad(model.energy)` -- the engine's egrad pattern; only the 1D
    mass element (dx, not dx^3) and the absent constraint differ from the library
    stepper."""
    gradE = jax.grad(lambda p: model.energy(p, grid))
    accel = lambda p: -gradE(p) / grid.dx               # = c^2 * discrete laplacian

    @jax.jit
    def step(carry, _):
        p, v = carry
        a0 = accel(p)
        p = p + dt * v + 0.5 * dt * dt * a0
        v = v + 0.5 * dt * (a0 + accel(p))
        return (p, v), None

    def evolve(p, v, n_steps):
        (p, v), _ = jax.lax.scan(step, (p, v), None, length=n_steps)
        return p, v
    return evolve


# ----------------------------------------------------------------- seeds ------
def harmonic_pulse(grid: BoxGrid, harmonics, *, c: float = 1.0, travelling=True):
    """A pulse carrying every listed harmonic at once: phi = sum_n cos(k_n x)
    (a band-limited, sharply-peaked 'pluck'). With `travelling`, the matching
    d_t phi = sum_n c k_n sin(k_n x) launches it as a right-moving wave."""
    x = grid.axis()
    phi = jnp.zeros(grid.N, dtype=grid.dtype)
    v = jnp.zeros(grid.N, dtype=grid.dtype)
    for n in harmonics:
        k = 2.0 * jnp.pi * n / grid.L
        phi = phi + jnp.cos(k * x)
        if travelling:
            v = v + c * k * jnp.sin(k * x)
    return phi, v


def _mode_energy(phi, grid: BoxGrid, harmonics):
    """Fraction of the spectral power sitting in the seeded harmonics (coherence:
    a linear wave equation creates no new modes, so this stays ~1)."""
    p = jnp.abs(jnp.fft.rfft(phi)) ** 2
    return float(p[jnp.asarray(harmonics)].sum() / p[1:].sum())


def _peak_retention(phi0, *, omega, t):
    """Pure spectral evolution to time t under a dispersion omega(n), returning
    how much of the packet's peak amplitude survives (1 = stays as sharp). A
    coherent (non-dispersive) packet just translates -> ~1; a dispersive one
    spreads -> < 1. (Evaluated at a GENERIC t, away from the ring's exact Talbot
    revivals where even a dispersive packet momentarily reforms.)"""
    N = phi0.shape[0]
    hat = jnp.fft.rfft(phi0)
    n = jnp.arange(hat.shape[0])
    phit = jnp.fft.irfft(hat * jnp.exp(-1j * omega(n) * t), n=N)
    return float(jnp.max(jnp.abs(phit)) / jnp.max(jnp.abs(phi0)))


# ------------------------------------------------------------------ demo ------
def main():
    N, c = 1024, 1.0
    grid = BoxGrid(N=N, L=float(N), dtype=jnp.float32)   # ring: L=N -> dx=1
    model = wave_ring_model(c=c)
    phi0, v0 = harmonic_pulse(grid, HARMONICS, c=c, travelling=True)

    period = grid.L / c                                  # time to travel once around
    dt = 0.05
    n_steps = int(round(period / dt))
    evolve = make_leapfrog(model, grid, dt)

    E0 = float(total_energy(phi0, v0, model, grid))
    peak0 = float(jnp.max(jnp.abs(phi0)))
    phiT, vT = evolve(phi0, v0, n_steps)
    ET = float(total_energy(phiT, vT, model, grid))

    coherT = _mode_energy(phiT, grid, HARMONICS)
    peakT = float(jnp.max(jnp.abs(phiT)))
    recur = float(jnp.linalg.norm(phiT - phi0) / jnp.linalg.norm(phi0))

    print(f"Ring of N={N} points carrying harmonics n=1..{HARMONICS[-1]} at once, c={c}")
    print(f"  leapfrog-evolved one round trip (period L/c={period:.0f}, {n_steps} steps)\n")
    print(f"  spectral coherence (power kept in the 24 modes):  {coherT:.4f}")
    print(f"  peak amplitude retained as it travelled:          {peakT/peak0:.3f}")
    print(f"  energy conservation (leapfrog invariant):         dE/E = {abs(ET-E0)/E0:.1e}")
    print(f"  recurrence after one lap ||phiT-phi0||/||phi0|| =  {recur:.3f}")
    print(f"    (the residual is the DISCRETE scheme's numerical dispersion -- the high")
    print(f"     harmonics creep; it shrinks with N/dt. The continuum packet is exact.)\n")

    # WHY the harmonics stay together: peak retention at a GENERIC travel time
    # (a quarter lap), non-dispersive vs dispersive -- both pure-spectral, exact.
    t = period / 4.0
    k1 = 2.0 * jnp.pi / grid.L
    nd = _peak_retention(phi0, omega=lambda n: c * k1 * n, t=t)          # ~ n
    dp = _peak_retention(phi0, omega=lambda n: c * k1 * n**2, t=t)       # ~ n^2
    print(f"  why they travel together -- peak kept after a quarter lap:")
    print(f"    non-dispersive  omega_n ~ n   : {nd:.3f}   (stays sharp -- just translates)")
    print(f"    dispersive      omega_n ~ n^2 : {dp:.3f}   (spreads -- harmonics dephase)")
    print(f"  So 'all 24 harmonics travelling together' is exactly the non-dispersive ring.")


if __name__ == "__main__":
    main()

"""Kuramoto ring oscillators as topological solitons — an extensibility demo.

Prompted by Luke Leighton on the d12rg list ("can it be put to use on kuramoto
ring oscillators?"). The answer is yes, and it exercises exactly the engine's
core abstractions on a system about as far from a 3D hopfion as you can get:

  - A ring of N phase oscillators theta_i in S^1, nearest-neighbour coupled.
  - The attractive Kuramoto coupling has a Lyapunov (energy) function
        E(theta) = -K * sum_i cos(theta_{i+1} - theta_i),
    whose gradient flow IS the synchronisation dynamics
        d theta_i/dt = K[ sin(theta_{i+1}-theta_i) + sin(theta_{i-1}-theta_i) ].
    We never write that force: `jax.grad(model.energy)` supplies it (the engine's
    egrad pattern), and the library's dimension-agnostic `adam_flow` descends it.
  - A **q-twisted state** theta_i = 2*pi*q*i/N is a topological soliton: its
    integer **winding number** W = (1/2pi) sum_i wrap(theta_{i+1}-theta_i) is a
    1D cousin of the Hopf/Skyrme charge. We **relax-then-ID** it, just like a
    knotted carrier -- and a winding-number-changing "unwinding" event is the
    direct analogue of a soliton decay.

The classic result (Wiley-Strogatz-Girvan 2006): for the ring of identical
oscillators with sinusoidal coupling, the q-twisted state is linearly stable iff
|q| < N/4. This script reproduces that boundary from a noisy seed: stable twists
relax back and KEEP their winding number; unstable ones unwind to a lower |W|.

What this says about extensibility (the actual point):
  * GENERAL, reused unchanged: the `Model` abstraction (energy = sum of composable
    EnergyTerms + charges), the egrad force (`jax.grad(model.energy)`), and the
    `adam_flow` stepper (no shape/dimension assumptions).
  * Reused with a squint: `BoxGrid` -- we use its N / dx / periodicity for the 1D
    ring and ignore its 3D `coords()`.
  * Did NOT transfer: the FFT-preconditioned `arrested_flow` (hard-wired to a 3D
    k-space). A 1D problem brings its own preconditioner, or just uses Adam.

Run:  python examples/kuramoto_ring.py
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp

from jax_solitons.grid import BoxGrid
from jax_solitons.model import Model
from jax_solitons.steppers import adam_flow


# --------------------------------------------------------------- the model ----
@dataclasses.dataclass(frozen=True)
class KuramotoCoupling:
    """Attractive nearest-neighbour coupling on a ring (the Kuramoto Lyapunov
    energy). `jnp.roll` gives the periodic neighbour; `grid.dx` is unused (a
    network of oscillators is intrinsically a lattice, not a continuum field)."""

    K: float = 1.0
    name: str = "kuramoto_coupling"

    def __call__(self, theta, grid: BoxGrid):
        d = jnp.roll(theta, -1) - theta                  # theta_{i+1} - theta_i
        return -self.K * jnp.sum(jnp.cos(d))


def _wrap(d):
    """Principal value of a phase difference, via the complex phase (`jnp.angle`,
    the repo's convention for angles). A difference of EXACTLY +-pi is a genuine
    measure-zero ambiguity -- the two representations are the same point on the
    circle and the winding number is undefined there -- so neither this nor a
    round(d/2pi) wrap can pin the sign at the boundary; it never arises for the
    |q| < N/2 twists here (neighbour differences 2*pi*q/N stay strictly < pi)."""
    return jnp.angle(jnp.exp(1j * d))


def winding_number(theta, grid: BoxGrid):
    """The topological charge: (1/2pi) * sum of wrapped neighbour differences.
    Integer for any configuration whose adjacent phases differ by < pi; the 1D
    analogue of the Hopf/Skyrme charge the engine quantises in 3D."""
    return jnp.sum(_wrap(jnp.roll(theta, -1) - theta)) / (2.0 * jnp.pi)


def order_parameter(theta):
    """Kuramoto global sync order r = |<e^{i theta}>| in [0, 1] (a diagnostic)."""
    return jnp.abs(jnp.mean(jnp.exp(1j * theta)))


def kuramoto_ring_model(K: float = 1.0) -> Model:
    """The Kuramoto ring as a `Model` configuration -- one coupling term, no
    manifold constraint (theta is a real lift so the state can wind), winding
    number as the topological charge."""
    return Model(
        name="kuramoto_ring",
        terms=(KuramotoCoupling(K=K),),
        constraint=None,                                 # real-valued lift; can wind
        charges=(winding_number,),
    )


# ----------------------------------------------------------------- seeds ------
def twisted_state(grid: BoxGrid, q: int, *, noise: float = 0.0, key=None):
    """A q-twisted ring theta_i = 2*pi*q*i/N, optionally perturbed so an unstable
    twist has something to unwind along."""
    i = jnp.arange(grid.N, dtype=grid.dtype)
    theta = 2.0 * jnp.pi * q * i / grid.N
    if noise and key is not None:
        theta = theta + noise * jax.random.normal(key, (grid.N,), dtype=grid.dtype)
    return theta


# ------------------------------------------------------------------ demo ------
def main():
    N = 64
    grid = BoxGrid(N=N, L=float(N), dtype=jnp.float32)   # L=N -> dx=1 (a lattice)
    model = kuramoto_ring_model(K=1.0)
    q_crit = N / 4.0                                     # Wiley-Strogatz-Girvan
    print(f"Kuramoto ring: N={N} oscillators, K=1.0  (stability boundary |q|<{q_crit:.0f})")
    print(f"  relaxing each q-twisted seed (+noise) with the library's adam_flow\n")
    print(f"  {'q':>3} {'W_seed':>7} {'W_relaxed':>10} {'r':>6} {'stable?':>8} "
          f"{'predicted':>10}")

    # Small perturbation -> tests LINEAR stability (the regime the |q|<N/4 result
    # is about). A large kick tests basin/nonlinear stability, a tighter boundary
    # -- the soliton ε*-barrier story, and a separate experiment.
    key = jax.random.PRNGKey(0)
    rows = []
    for q in (0, 2, 6, 10, 14, 16, 18, 22, 28):
        key, sub = jax.random.split(key)
        theta0 = twisted_state(grid, q, noise=0.15, key=sub)
        w_seed = int(jnp.round(winding_number(theta0, grid)))
        relaxed, _ = adam_flow(model, theta0, grid, lr=2e-2, steps=6000)
        w_relaxed = int(jnp.round(winding_number(relaxed, grid)))
        r = float(order_parameter(relaxed))
        held = w_relaxed == w_seed
        predicted = abs(q) < q_crit
        flag = "OK" if held == predicted else "  <-- !!"
        print(f"  {q:>3} {w_seed:>7} {w_relaxed:>10} {r:>6.3f} "
              f"{('hold' if held else 'unwound'):>8} "
              f"{('stable' if predicted else 'unwind'):>10}  {flag}")
        rows.append((q, w_seed, w_relaxed, held, predicted))

    agree = sum(1 for *_, held, predicted in rows if held == predicted)
    print(f"\n  {agree}/{len(rows)} seeds match the |q|<N/4 stability boundary.")
    print("  Relax-then-ID on a 1D phase field: the winding number is the charge,")
    print("  and unwinding is a topological-charge-changing event -- same engine,")
    print("  same pattern, a system with no 3D field in sight.")


if __name__ == "__main__":
    main()

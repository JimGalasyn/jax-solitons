#!/usr/bin/env python3
"""Relativistic NLKG vortex dynamics: ring self-propulsion, ring-ring collision
(reconnection + Kelvin waves), and a multi-ring tangle.

This is the dynamical-substrate counterpart of the static relaxers: a genuine
WAVE equation (steppers.verlet), so vortices propagate, radiate sound, reconnect
and untie freely. It demonstrates the models.nlkg drop-in -- the model is three
composable EnergyTerms, the evolution is the stock velocity-Verlet stepper.

    python examples/nlkg_vortex_tangle.py ring      # one ring self-propels
    python examples/nlkg_vortex_tangle.py collide    # two rings -> reconnection
    python examples/nlkg_vortex_tangle.py tangle      # multi-ring tangle
    python examples/nlkg_vortex_tangle.py ring --outdir /tmp/nlkg --steps 4000

--sponge turns on an absorbing boundary shell (a -gamma d_t Phi damping in the
outer cells) so outgoing sound does not wrap around the periodic box; without it
the run is purely conservative (energy drift < 0.1%).
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from jax_solitons.grid import BoxGrid  # noqa: E402
from jax_solitons.models import nlkg  # noqa: E402
from jax_solitons.steppers.verlet import kinetic_energy, make_verlet_step  # noqa: E402

MODES = {
    "ring": dict(
        rings=[dict(R=6.0, xi=1.0, center=(0, 0, -6), axis="z", sign=1)],
    ),
    "collide": dict(  # two coaxial rings fired toward each other
        rings=[
            dict(R=6.0, xi=1.0, center=(0, 0, -8), axis="z", sign=1),
            dict(R=6.0, xi=1.0, center=(0, 0, +8), axis="z", sign=-1),
        ],
    ),
    "tangle": dict(  # several rings at assorted positions/orientations
        rings=[
            dict(R=5.0, xi=1.0, center=c, axis=ax, sign=s)
            for c, ax, s in [
                ((-6, -6, 0), "z", 1),
                ((6, 6, 2), "x", 1),
                ((0, -7, 6), "y", -1),
                ((7, -2, -6), "z", -1),
                ((-5, 6, -4), "x", 1),
            ]
        ],
    ),
}


def make_sponge(grid: BoxGrid, width: int = 12, gmax: float = 2.0):
    """Absorbing boundary shell: gamma = 0 in the interior, ramps (cos^2) to gmax
    in the outer `width` cells. A velocity damping v *= exp(-gamma*dt) per step
    soaks up radiation that reaches the faces instead of letting the periodic BC
    wrap it back in -- an effectively open/radiative boundary."""
    i = np.arange(grid.N)
    d = np.minimum(i, grid.N - 1 - i)
    r = np.clip((width - d) / width, 0, 1) ** 2
    rx, ry, rz = np.meshgrid(r, r, r, indexing="ij")
    shell = 1.0 - (1 - rx) * (1 - ry) * (1 - rz)
    return jnp.asarray(gmax * shell, dtype=grid.dtype)


def settle(model, grid, s, steps=400, dtau_frac=0.1):
    """Short imaginary-time gradient flow: settle |Phi| to the equilibrium vortex
    profile (drop the IC sound burst) without moving the vortices. dtau is kept
    below the explicit-diffusion limit dx^2/6."""
    gE = jax.jit(jax.grad(lambda x: model.energy(x, grid)))
    dtau = dtau_frac * grid.dx**2
    for _ in range(steps):
        s = s - dtau * gE(s) / grid.dx**3
    return s


def save_silhouette(state, grid, path, axis=2):
    """Vortex-line silhouette: min |Phi| projected along `axis` -- dark curves
    trace the cores (|Phi| -> 0), a 2D shadow of the 3D tangle."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    phi = nlkg.to_complex(state)
    proj = np.asarray(jnp.min(jnp.abs(phi), axis=axis))
    plt.figure(figsize=(5, 5))
    plt.imshow(proj.T, origin="lower", cmap="magma", vmin=0, vmax=1)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(path, dpi=90, bbox_inches="tight")
    plt.close()


def core_cells(state, f0sq=1.0, frac=0.25):
    """Crude vortex-length proxy: # cells with |Phi|^2 < frac*F0^2."""
    phi = nlkg.to_complex(state)
    return int(jnp.sum(jnp.abs(phi) ** 2 < frac * f0sq))


def run(mode, N=128, L=32.0, lam=1.0, f0sq=1.0, m0=0.0, dt_frac=0.4,
        steps=4000, every=200, settle_steps=400, sponge=False, outdir=None):
    grid = BoxGrid(N=N, L=L, dtype=jnp.float64)
    model = nlkg.nlkg_model(lam=lam, f0sq=f0sq, m0=m0)
    s = nlkg.vortex_seed(grid, f0=float(np.sqrt(f0sq)), **MODES[mode])
    s = settle(model, grid, s, steps=settle_steps)
    v = jnp.zeros_like(s)                       # at rest; rings self-propel

    dt = dt_frac * grid.dx / np.sqrt(3.0)       # CFL: dt < dx/sqrt(3)
    step = make_verlet_step(model, grid, dt)
    damp = jnp.exp(-make_sponge(grid) * dt) if sponge else None

    def total_E():
        return float(model.energy(s, grid) + kinetic_energy(v, grid))

    print(f"NLKG {mode}  N={N} L={L} dx={grid.dx:.3f} dt={dt:.4f} lam={lam} "
          f"steps={steps} sponge={sponge} (CFL dx/sqrt3={grid.dx/np.sqrt(3):.3f})")
    E0 = total_E()
    if outdir:
        Path(outdir).mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    frame = 0
    for n in range(steps + 1):
        if n % every == 0:
            E = total_E()
            print(f"  s{n:5d} t={n*dt:6.2f}: E={E:9.3f} "
                  f"(dE/E0={(E - E0) / abs(E0) * 100:+.3f}%) "
                  f"core_cells={core_cells(s, f0sq):6d}", flush=True)
            if outdir:
                save_silhouette(s, grid, f"{outdir}/{mode}_{frame:03d}.png")
                frame += 1
            if not np.isfinite(E):
                print("  BLEW UP")
                break
        if n < steps:
            s, v = step(s, v)
            if damp is not None:           # absorbing-shell velocity damping
                v = v * damp
    print(f"  ({time.time() - t0:.0f}s)")
    return grid, s, v


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=list(MODES), nargs="?", default="ring")
    p.add_argument("--N", type=int, default=128)
    p.add_argument("--L", type=float, default=32.0)
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--every", type=int, default=200)
    p.add_argument("--sponge", action="store_true")
    p.add_argument("--outdir", default=None)
    a = p.parse_args(argv)
    run(a.mode, N=a.N, L=a.L, steps=a.steps, every=a.every,
        sponge=a.sponge, outdir=a.outdir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

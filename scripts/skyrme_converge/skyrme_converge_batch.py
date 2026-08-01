#!/usr/bin/env python3
"""On-box driver: Skyrme B=2 binding convergence (resolution ladder, dx -> 0).

Imports the REAL jax_solitons.models.skyrme (installed on-box from the
skyrme-module branch by vast/onstart_skyrme.sh) -- no inlined port to drift,
unlike soft_link_batch.py. One driver run relaxes BOTH the B=1 hedgehog and the
B=2 rational-map torus at a given N in the SAME box (so the two legs share the
resolution) and writes scalars.

THE RECIPE (validated locally to recover the known binding to ~0.1%): the
massless Skyrme B=2 has a published classical binding ~4.3% (per-baryon energy
1.232 -> 1.179 in Bogomolny-bound units). Two things must be right or the
number is junk:
  1. The BOX MUST FIT THE SOLITON. Size ~ sqrt(c4/c2); at c2=c4=1 the B=1
     natural radius r0* ~ 2.0, so L=16 (r0* ~ L/8) fits B=1 and the larger B=2
     torus with margin. (c4=4 / L=8 made the soliton BIGGER than the box ->
     boundary-frustrated pseudo-collapse, E sinking below the Bogomolny bound.)
  2. A MONOTONE-SETTLING relaxer. Fixed-lr Adam glides through the minimum into
     a lattice spike (E < bound) or creeps UP in the wrong frame; arrested_flow
     (preconditioned backtracking, guaranteed monotone) settles AT the minimum.
At c4=1/L=16/N=80 this gives E1/bnd=1.220, E2/bnd=1.168, binding=+4.21% (vs the
published ~4.3%). This ladder confirms it tightens to 1.232/1.179/4.3% as dx->0.

The Bogomolny bound E >= 12 pi^2 sqrt(c2 c4)|B| is the physical floor (any
settled E < bound = under-resolved); baryon number is the exact on-box charge.

  python skyrme_converge_batch.py --N 96 --L 16 --c4 1 --out out_sk

Collate the ladder LOCALLY with skyrme_converge_analyze.py.
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from jax_solitons.grid import BoxGrid
from jax_solitons.models.skyrme import (
    skyrme_model, baryon_charge, skyrme_bound, E2O4Term, E4SkyrmeTerm)
from jax_solitons.seeds import skyrmion_hedgehog, skyrmion_rational_map
from jax_solitons.steppers import arrested_flow


def settle(model, seed, grid, *, c2, c4, bound, B_target, dt, block, max_steps,
           dt_floor, label):
    """Monotone arrested_flow in blocks until it stalls (dt floored = settled
    at the minimum) or max_steps. Returns the endpoint record -- since the flow
    is monotone, the endpoint IS the minimum (no capture heuristic needed)."""
    state = seed
    done = 0
    while done < max_steps:
        state, h = arrested_flow(model, state, grid, dt=dt, steps=block)
        done += block
        dt = h[-1][2]
        E = float(model.energy(state, grid))
        e2 = float(E2O4Term(c2=c2)(state, grid))
        e4 = float(E4SkyrmeTerm(c4=c4)(state, grid))
        print(f"    [{label}] step={done} E={E:10.2f} /bnd={E/bound:.3f} "
              f"ratio={e2/e4:.3f} dt={dt:.1e}", flush=True)
        if dt < dt_floor:
            break
    B = float(baryon_charge(state, grid))
    E = float(model.energy(state, grid))
    e2 = float(E2O4Term(c2=c2)(state, grid)); e4 = float(E4SkyrmeTerm(c4=c4)(state, grid))
    return {"E": E, "E_over_bound": E / bound, "ratio": e2 / e4, "B": B,
            "B_target": B_target, "physical": bool(E >= bound and round(B) == B_target),
            "sub_bound_seen": bool(E < bound), "steps": done}, state


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=96)
    ap.add_argument("--L", type=float, default=16.0)
    ap.add_argument("--c2", type=float, default=1.0)
    ap.add_argument("--c4", type=float, default=1.0)
    ap.add_argument("--dt", type=float, default=1e-3, help="arrested_flow start dt")
    ap.add_argument("--block", type=int, default=600, help="steps per arrested block")
    ap.add_argument("--max-steps", type=int, default=12000)
    ap.add_argument("--dt-floor", type=float, default=1e-9,
                    help="stop when backtracking dt floors below this (settled)")
    ap.add_argument("--r0-1", type=float, default=2.0, help="B=1 seed radius (~r0*)")
    ap.add_argument("--r0-2", type=float, default=2.6, help="B=2 torus seed radius")
    ap.add_argument("--only", choices=["b1", "b2", "both"], default="both")
    ap.add_argument("--out", default="out_sk")
    args = ap.parse_args()

    grid = BoxGrid(N=args.N, L=args.L, dtype=jnp.float64)
    c2, c4 = args.c2, args.c4
    model = skyrme_model(c2=c2, c4=c4)
    dx = args.L / args.N
    b1, b2 = skyrme_bound(1, c2, c4), skyrme_bound(2, c2, c4)
    print(f"N={args.N} L={args.L} dx={dx:.4f} c2={c2} c4={c4} dt={args.dt}  "
          f"bound1={b1:.2f} bound2={b2:.2f}", flush=True)

    summary = {"N": args.N, "L": args.L, "dx": dx, "c2": c2, "c4": c4,
               "dt": args.dt, "bound1": b1, "bound2": b2, "results": {}}
    sk = dict(c2=c2, c4=c4, dt=args.dt, block=args.block, max_steps=args.max_steps,
              dt_floor=args.dt_floor)

    if args.only in ("b1", "both"):
        print("B=1 hedgehog:", flush=True)
        t0 = time.time()
        rec, _ = settle(model, skyrmion_hedgehog(grid, r0=args.r0_1), grid,
                        bound=b1, B_target=1, label="b1", **sk)
        rec["secs"] = time.time() - t0
        summary["results"]["b1"] = rec
        print(f"  [b1] E={rec['E']:.2f} /bound={rec['E_over_bound']:.3f} "
              f"B={rec['B']:+.0f} physical={rec['physical']}", flush=True)

    if args.only in ("b2", "both"):
        print("B=2 rational-map torus:", flush=True)
        t0 = time.time()
        rec, _ = settle(model, skyrmion_rational_map(grid, B=2, r0=args.r0_2),
                        grid, bound=b2, B_target=2, label="b2", **sk)
        rec["secs"] = time.time() - t0
        summary["results"]["b2"] = rec
        print(f"  [b2] E={rec['E']:.2f} /bound={rec['E_over_bound']:.3f} "
              f"B={rec['B']:+.0f} physical={rec['physical']}", flush=True)

    if "b1" in summary["results"] and "b2" in summary["results"]:
        E1 = summary["results"]["b1"]["E"]
        E2 = summary["results"]["b2"]["E"]
        summary["E1"] = E1
        summary["E2"] = E2
        summary["binding_frac"] = 1.0 - E2 / (2.0 * E1)
        summary["binds"] = bool(E2 < 2.0 * E1)
        print(f"\n  E(B=2)/(2 E(B=1)) = {E2/(2*E1):.4f}  "
              f"binding = {100*(1-E2/(2*E1)):+.2f}%  "
              f"({'BINDS' if E2 < 2*E1 else 'no bind'})  ideal ~4.3%", flush=True)

    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
    with open(outdir / "manifest.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"wrote {outdir}/manifest.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

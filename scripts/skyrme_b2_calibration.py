"""Skyrme B=2 binding calibration (the reason the module exists).

Relax the B=1 hedgehog and the B=2 rational-map (torus) seed to their minima
and check the published Battye-Sutcliffe binding inequality

    E(B=2) < 2 E(B=1)      (classical massless binding ~4%).

This is the known-answer yardstick: if our descent reproduces the known Skyrme
B=2 binding, the method is trustworthy on the NWT deuteron; if it does not,
we have found the method's blind spot on a system where we know the answer.

Run:  .venv/bin/python scripts/skyrme_b2_calibration.py [--N 40] [--steps 8000]
"""

from __future__ import annotations

import argparse

import jax.numpy as jnp

from jax_solitons import BoxGrid
from jax_solitons.models.skyrme import (
    E2O4Term,
    E4SkyrmeTerm,
    baryon_charge,
    skyrme_bound,
    skyrme_model,
)
from jax_solitons.seeds import skyrmion_hedgehog, skyrmion_rational_map
from jax_solitons.steppers import adam_flow


def virial(phi, grid, c2, c4):
    """(c2 E2)/(c4 E4) -> 1 at the Derrick (virial) point of the massless
    model (the weighted terms balance at the soliton size)."""
    e2 = float(E2O4Term(c2=c2)(phi, grid))
    e4 = float(E4SkyrmeTerm(c4=c4)(phi, grid))
    return e2, e4, e2 / e4


def relax(model, seed, grid, *, lr, steps, log_every, c2, c4, bound, B_target):
    """Adam descent, but RETURN the virial-point config (min |c2.E2/c4.E4 - 1|)
    among PHYSICAL states (E >= bound, B held) rather than the endpoint: Adam
    glides through the physical minimum (weighted terms balance, E ~ 1.23 x
    bound) and on into lattice Derrick collapse (ratio -> inf, E sinks BELOW the
    Bogomolny bound -- a sub-grid spike the forward-diff energy can't see). The
    virial crossing is the physical soliton; capture it, rejecting collapsed
    (sub-bound) or topology-lost samples. If NO sampled state is physical, the
    run is under-resolved at this dx -- flagged, endpoint returned.
    (The Faddeev 'return-best' lesson, keyed on the virial residual since the
    collapsed state has spuriously LOW E; the Bogomolny bound is the clean
    convergence floor the NWT deuteron system lacked.)"""
    state = seed
    best = None              # (|ratio-1|, E, state, B) among physical samples
    sub_bound_seen = False
    seg = max(1, steps // max(1, log_every))
    done = 0
    while done < steps:
        n = min(seg, steps - done)
        state, _ = adam_flow(model, state, grid, lr=lr, steps=n)
        done += n
        E = float(model.energy(state, grid))
        e2, e4, vr = virial(state, grid, c2, c4)
        B = baryon_charge(state, grid)
        physical = (E >= bound) and (round(B) == B_target)
        sub_bound_seen = sub_bound_seen or (E < bound)
        resid = abs(vr - 1.0)
        if physical and (best is None or resid < best[0]):
            best = (resid, E, state, B)
        tag = "" if physical else "  (sub-bound/collapsed)"
        bE = best[1] if best else float("nan")
        print(f"    step {done:6d}  E={E:10.2f}  c2E2/c4E4={vr:6.3f}  B={B:+.0f}"
              f"   [virial-best E={bE:.2f}]{tag}")
    if best is None:
        print("    !! UNDER-RESOLVED: no physical (E>=bound, B held) sample; "
              "returning endpoint")
        return state, float(model.energy(state, grid)), baryon_charge(state, grid)
    if sub_bound_seen:
        print("    .. note: trajectory dipped below bound (collapse channel "
              "reachable at this dx)")
    return best[2], best[1], best[3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=40)
    ap.add_argument("--L", type=float, default=8.0)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--c2", type=float, default=1.0)
    ap.add_argument("--c4", type=float, default=4.0)
    ap.add_argument("--r0_1", type=float, default=None, help="B=1 seed radius")
    ap.add_argument("--r0_2", type=float, default=None, help="B=2 seed radius")
    args = ap.parse_args()

    grid = BoxGrid(N=args.N, L=args.L)
    c2, c4 = args.c2, args.c4
    model = skyrme_model(c2=c2, c4=c4)
    print(f"grid N={args.N} L={args.L} dx={grid.dx:.3f}  c2={c2} c4={c4}  "
          f"steps={args.steps} lr={args.lr}")

    b1, b2 = skyrme_bound(1, c2, c4), skyrme_bound(2, c2, c4)

    print("B=1 hedgehog:")
    s1, E1, B1 = relax(model, skyrmion_hedgehog(grid, r0=args.r0_1), grid,
                       lr=args.lr, steps=args.steps, log_every=24, c2=c2, c4=c4,
                       bound=b1, B_target=1)

    print("B=2 rational-map torus:")
    s2, E2, B2 = relax(model, skyrmion_rational_map(grid, B=2, r0=args.r0_2),
                       grid, lr=args.lr, steps=args.steps, log_every=24,
                       c2=c2, c4=c4, bound=b2, B_target=2)
    print("\n=== RESULT (virial-point energies) ===")
    print(f"E(B=1) = {E1:10.2f}   /bound {E1/b1:5.3f}   B={B1:+.0f}")
    print(f"E(B=2) = {E2:10.2f}   /bound {E2/b2:5.3f}   B={B2:+.0f}")
    print(f"2 E(B=1) = {2*E1:10.2f}")
    print(f"E(B=2)/(2 E(B=1)) = {E2/(2*E1):.4f}   "
          f"binding = {100*(1 - E2/(2*E1)):+.2f}%")
    print("BINDS" if E2 < 2 * E1 else "DOES NOT BIND (E2 >= 2 E1)")


if __name__ == "__main__":
    main()

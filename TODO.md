# TODO

## Deep-convergence relaxation protocol (virial E2/E4 -> ~1)

At N=96, L=18, c4=4 the VK spectrum gate passes (E(Q=2)/E(Q=1) = 1.6402,
published 1.623; source engine 1.604) with honest charges, but neither
relaxer reaches the Derrick point:

| pipeline | E(Q=1) | E2/E4 |
|---|---|---|
| arrested_flow(1500) + adam_flow(40k, lr=2e-3) | 1181.4 | 0.655 |
| adam_flow(40k, lr=2e-3) from seed | 1212.0 | 1.340 |

The two bracket the virial point from opposite sides; the source research
engine reached E2/E4 ~ 0.91 (its lattice-normal base at this resolution).
Notable: **fp32 and x64 endpoints are IDENTICAL to 4 significant figures**,
so precision is not the blocker — flow depth/schedule is.

Likely fix: staged protocol (arrested scout -> Adam with lr decay ->
optional L-BFGS polish), exposed as a stepper composition. The GPU-tier
gate currently asserts a sanity band [0.5, 1.5] with measured values
documented; tighten to [0.8, 1.2] when this lands.

## Other

- nwt-substrate knot-id integration (flips the trefoil-determinant gate)
- VOS minimal-superflow seeds + multi-soliton composition
- gauged abelian-Higgs model terms + Coulomb gauge constraint
- L-BFGS and ETDRK steppers
- jaxDecomp sharding layer (NamedSharding default + shard_map halo islands)
- orbax checkpoint backend when sharded arrays land
- vmap sweep driver over RunConfig batches (the farm front-end)

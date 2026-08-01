# TODO

## RESOLVED: deep convergence is a coordinate-frame problem, not a schedule

The old hypothesis (staged lr decay) was refuted by measurement; the fix
was the OPTIMIZER FRAME. Projected Adam on the n-field freezes the soft
Derrick scaling mode: at N=96, L=18, c4=4, lr=2e-3 and lr=1e-2 both crawl
+0.002 per 1k steps from an E2/E4 ~ 0.65-0.68 plateau with E creeping UP
(noise-floor orbit, never converges). The same Adam on the CP^1 spinor
state (`faddeev_cp1_model`, the source engine's frame) glides straight in:

| pipeline (Q=1) | E | E2/E4 |
|---|---|---|
| n-frame: arrested(1500) + adam(40k, lr=2e-3) | 1181.4 | 0.655 |
| n-frame: adam(40k, lr=2e-3) from seed | 1212.0 | 1.340 |
| n-frame: arrested(1500) + adam(12k, lr=1e-2) | 1171.9 | 0.680 |
| **cp1-frame: adam(40k, lr=2e-3) from seed** | **1108.0** | **0.929** |

The spinor run hits the virial plateau (0.907) by 2k steps, then a second
descent phase runs ~12k-22k and settles at 0.929 — the source lesson
"15k = scouting, 40k = converged" holds in this engine too. The GPU-tier
virial gate is now a depth gate at [0.8, 1.2]. (Experiment lives in the
source research codebase.)

## Gallery (docs/gallery, GitHub Pages later)

Image gallery of the prettier results + validation runs, regenerable
from committed scripts (each image = one `examples/` script + RunConfig,
so the gallery doubles as living documentation):

- relaxed hopfions Q=1..4 (energy iso-surfaces + core curves; the
  (n,m)-factorisation family side by side)
- a CP^1 deep relaxation as an animation: seed -> virial plateau ->
  second descent (the gate-5 story in one GIF)
- GPE: vortex ring with phase coloring; dark-soliton kink pair; the
  Bogoliubov dispersion measured-vs-analytic curve (validation figure)
- VK spectrum plot: E(Q) vs Q^(3/4) with the published values overlaid
- area-form quantization: flux/(4*pi) histogram on random maps snapping
  to integers (the Tier-1 exactness test as a picture)
- a kicked-soliton "fireball" frame with the relax-then-ID core curve
  overlaid — the instrument-honesty figure (bath vs basin)

Render style: the source repo's hopfion/turntable renderers port over;
keep images neutral (engine physics, no downstream interpretation).

## Collider-campaign scaling (the pedal-to-the-metal tier)

Collision sweeps (species x boost x impact parameter x orientation) are
10^4-10^6 runs at N>=256 — ~$700/10^4 on marketplace 3090s at measured
rates. Order of attack:
1. adaptive sampling around outcome boundaries (10-100x over grids)
2. trigger-style ID: cheap classifiers (Q ledger, blob count) on all
   events, knot-ID only on flagged ones; cythonize/parallelize the
   tracer (ID, not GPU, is the measured cost center)
3. event records not raw fields (~1MB/run: config + ledgers + census +
   core polylines; full fields only when triggered)
4. fleet orchestration over the R4 manifest (work queue, auto-rent,
   bad-host probe auto-cycle)
5. asymmetric BoxGrid (Nx,Ny,Nz) for collision geometry; absorbing
   far field to stop periodic radiation re-entry
6. jaxDecomp sharding only when single runs outgrow one GPU (N>=512)

## Other

- knot identification (core-curve Alexander determinant) integration
  (flips the trefoil-determinant gate)
- VOS minimal-superflow seeds + multi-soliton composition
- gauged abelian-Higgs: model SHIPPED (compact-link, exact gauge invariance,
  flux charge, vortex seed — models/abelian_higgs.py). Remaining: Coulomb-gauge
  retraction (optional); BPS line-tension validation (2πv²|n|) on a periodic
  vortex-antivortex config; reconcile the Paper 11 (λ=2e²) / Paper 16 + oracle
  (λ=e²/2) convention split. Unblocks the null-worldtube gauged_higgs cleanup.
- L-BFGS and ETDRK steppers
- jaxDecomp sharding layer (NamedSharding default + shard_map halo islands)
- vmap sweep driver over RunConfig batches (the farm front-end)

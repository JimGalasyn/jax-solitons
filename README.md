# jax-solitons

[![CI](https://github.com/JimGalasyn/jax-solitons/actions/workflows/ci.yml/badge.svg)](https://github.com/JimGalasyn/jax-solitons/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/JimGalasyn/jax-solitons/branch/main/graph/badge.svg)](https://codecov.io/gh/JimGalasyn/jax-solitons)
[![release](https://img.shields.io/github/v/release/JimGalasyn/jax-solitons?include_prereleases&label=release)](https://github.com/JimGalasyn/jax-solitons/releases)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20774254.svg)](https://doi.org/10.5281/zenodo.20774254)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://github.com/JimGalasyn/jax-solitons)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Status: pre-alpha (0.0.x).** The API is being designed in the open and
> **will** change without notice until 0.1. Nothing here is stable yet.

A general JAX engine for classical field-theory solitons — Faddeev-Skyrme
hopfions, gauged abelian-Higgs vortices, Gross-Pitaevskii vortex knots —
designed for GPU farms.

## Why another soliton code?

The field's first public GPU soliton codes (cuSkyrmion, soliton_solver, 2026)
are CUDA C / Numba: single-GPU, forward-only (no autodiff), and either
single-theory (cuSkyrmion: Skyrme variants) or 2D (soliton_solver: a
composable 8-theory core, but planar). The mature neighbours are similar in
shape — mumax3/mumax+ (CUDA micromagnetics, forward-only) and the GPE/BEC
split-step solvers (GPUE, TorchGPE, PyGPE) are each single-physics and
single-GPU; only PyTorch's magnum.np is genuinely differentiable, and only
for finite-difference micromagnetics. `jax-solitons` aims to be the
differentiable, composable, 3D, farm-scale counterpart that occupies the
intersection none of them do, built on four design commitments:

1. **One `Model` abstraction.** A model is a field with a manifold constraint,
   an energy that is a sum of composable local terms (E₂, Skyrme E₄, |B|²,
   potential, GPE nonlinearity), and topological diagnostics. Forces come from
   `jax.grad` of the energy — you only ever write the energy. Hybrid theories
   are configurations, not new code.
2. **Batch-first state.** Solver state is a PyTree with an optional leading
   batch axis: `vmap` gives ensemble parameter sweeps for free; sharding
   scales them across devices.
3. **Topology-preserving, stencil-local numerics.** The topological charge is
   discretized by the Berg-Lüscher plaquette solid-angle (area form) — which
   measures honestly *and* presents a real unwinding barrier, where naive
   discretizations admit none at any resolution. The geometric area-form
   construction is a principled, established route to a faithful
   topological-charge discretization, and the numerics of the 3D Hopf index
   are an active question (cf. Phys. Rev. B **111**, 134408 (2025)); what is
   new here is carrying it as a *native, differentiable* primitive of a
   composable, farm-scale engine rather than a post-hoc diagnostic. All
   canonical operators are stencil-local, so they JIT, vectorize,
   and shard without all-to-all communication; spectral paths ride
   [jaxDecomp](https://github.com/DifferentiableUniverseInitiative/jaxDecomp)
   when distributed FFTs are needed.
4. **Restartable, registered runs over a fleet.** Orchestration is a
   physics-agnostic campaign contract (the only soliton-specific thing crossing
   it is an injected `RunFn`): a config-hashed registry + full-state checkpoints
   (field + velocity/optimizer state + RNG key) for bit-identical restart;
   streaming event-records, with full fields kept only on triggered events;
   probe-or-bail host admission; all over a pluggable executor (local now;
   SkyPilot for spot fleets, stubbed). A literature sweep found no library
   covers this combination — see [CAMPAIGN.md](CAMPAIGN.md).

Design principles (the scale-first contract every PR is reviewed
against) are in [DESIGN.md](DESIGN.md).

## Scope: what physics fits

The engine covers anything expressible as **an energy functional of
classical fields on a periodic lattice with local (or spectrally
solvable) terms**, evolved by energy descent, Hamiltonian dynamics, or
split-step — and run as campaigns: relaxed, evolved, topology-counted,
10⁵ times, with receipts.

**Native fits** (a `Model` configuration, no new infrastructure):
the topological-soliton zoo — Faddeev-Skyrme hopfions, nuclear Skyrme
(SU(2) ≅ the unit-quaternion constraint already shipped for CP¹), CP^N
sigma models, φ⁴ kinks, Q-balls, oscillons, cosmic strings and
monopoles with the gauge sector; the NLSE/GPE family — spinor BECs,
dipolar GPE, stochastic GPE for finite-temperature and Kibble-Zurek
campaigns; Ginzburg-Landau vortex matter; **micromagnetics** —
Landau-Lifshitz-Gilbert is a damped-precession stepper on exactly the
S² constraint here, and magnetic skyrmions/hopfions are the same
topology this engine counts exactly.

**Reasonable stretches** (one new capability each): Schrödinger-Poisson
/ fuzzy-dark-matter halos (GPE + one spectral Poisson solve — FDM halo
cores *are* solitons); expanding/comoving boxes (time-dependent term
coefficients); fixed curved backgrounds (position-dependent
coefficients, still stencil-local); analogue-gravity flows (sonic
horizons as GPE configurations); Langevin sampling.

**Out of scope by design**: particle/N-body methods, adaptive mesh
refinement (P3 commits to uniform lattices), constrained-hyperbolic
numerical relativity, and equilibrium Monte Carlo sampling. These are
different paradigms with mature codes; pretending otherwise is how
libraries become toys.

## Planned module layout

| Module | Contents | Status |
|---|---|---|
| `grid` | `BoxGrid` — periodic box, explicit dtype, sharding spec | working |
| `model` | `Model` / `EnergyTerm` / `Constraint` protocols | working |
| `models/` | Faddeev-Skyrme (E₂ + area-form E₄, S² constraint; CP¹ spinor frame for deep relaxation) | **working, validated** |
| `models/` | Gross-Pitaevskii (kinetic + g-potential, healing-length units) | **working, validated** |
| `models/` | gauged abelian-Higgs (compact-link `|Dφ|²` + plaquette `|F|²` + Higgs potential; **exact** lattice U(1) gauge invariance; magnetic-flux charge) | **working, gated** |
| `steppers/` | arrested (backtracking) flow; projected Adam; sphere-constrained velocity-Verlet; GPE split-step (imaginary + real time) | **working, validated** |
| `steppers/` | L-BFGS, ETDRK | porting |
| `topology` | area-form plaquette F_ij, Hopf charge (differentiable) | **working, validated** |
| `seeds` | rational-map hopfion ansatz | **working, validated** |
| `seeds` | solid-angle (VOS) minimal superflow, composition | porting |
| `runs` | `RunConfig`; full-state restartable checkpoints (bit-identical restart, gated in CI); config-hashed run dirs + manifest | **working, gated** |
| `campaign` | the A/B/C/E boundary — `RunRegistry`, `EventSink`, `Admission`, `Executor` protocols + `run_campaign`; local reference backends + `SkyPilotExecutor` stub | **working, gated** |
| `measure` | implicit core-curve tracer (lax.scan predictor-corrector), Gauss linking, arc-length resampling | **working, gated** |
| `runfns` / `examples` | `faddeev_relax_then_id` (relax-then-ID behind the campaign contract) + a runnable two-run campaign | **working, gated** |

Batch-first state (`vmap`) is demonstrated in CI: a vmapped dynamics step
over a stack of fields is verified identical to stepping each field
individually.

The full map of claims → tests → references (exact identities, analytic
solutions, published values) is in [VALIDATION.md](VALIDATION.md).

"Validated" = cross-checked **bit-identically** against the source research
engine (Faddeev: seed energy and Hopf charge match to the last digit at
N=64/fp64; GPE: split-step matches the source stepper to 9e-16 after 10
steps), plus live acceptance gates in CI: area-form Q_H held through
monotone descent, energy conservation + charge retention in real-time
dynamics, core-tracer ring identification, and bit-identical
checkpoint-restart. A GPU validation tier (`SOLITON_GPU_TIER=1 pytest
tests/test_gpu_tier.py`) runs the physics-level Vakulenko-Kapitanskii
spectrum gate with a Derrick depth gate (CP¹ spinor-frame relaxation
reaches the virial point E₂/E₄ ≈ 0.93, gated at [0.8, 1.2]), too slow for
CPU CI.

"Porting" = migrating from a validated private research codebase (relaxation
holds Hopf charge Q=0.998 through minimization; real-time integrator conserves
energy to dH=-0.000%; reproduces the published Faddeev-Hopf Q=1..4 spectrum
and the Vakulenko-Kapitanskii Q^(3/4) scaling to -1.9% on the fitted
exponent). The acceptance gates in `tests/` are the regression contract for
that port.

## Precision

fp32 is ~4.4× faster than fp64 on consumer GPUs for these workloads with
conservation identical to four decimals (memory-bandwidth-bound). That figure
is for energy descent and short integration; long real-time Hamiltonian
evolution (e.g. Kibble–Zurek-length runs) accumulates fp32 phase error that
step-wise conservation hides, so time-resolved dynamics are certified in fp64.
The engine's convention: **hunt in fp32, certify in fp64** — dtype is an
explicit `BoxGrid` parameter, never a global flag.

## Install (development)

```bash
pip install -e ".[test]"
pytest
```

## Quickstart

Relax a unit hopfion and read off its energy and (exactly quantized) Hopf
charge:

```python
import jax
jax.config.update("jax_enable_x64", True)          # certify in fp64

from jax_solitons.grid import BoxGrid
from jax_solitons.seeds import rational_map_hopfion_cp1
from jax_solitons.models.faddeev import faddeev_cp1_model, hopf_charge_cp1
from jax_solitons.steppers.adam import adam_flow

grid = BoxGrid(N=24, L=16.0)                        # dtype defaults to fp32
model = faddeev_cp1_model(c4=4.0)                   # Faddeev-Skyrme, CP¹ frame
z = rational_map_hopfion_cp1(grid, R=3.5, n=1, m=1) # a Q_H = 1 seed
z, _ = adam_flow(model, z, grid, lr=2e-3, steps=2000)   # more steps on a GPU
print(float(model.energy(z, grid)), float(hopf_charge_cp1(z, grid)))  # E, Q_H≈1
```

To run the same physics as a **registered, restartable campaign** (config-hashed
run dirs, streamed ledgers, triggered core-curve capture):

```bash
JAX_ENABLE_X64=1 python examples/faddeev_campaign.py   # writes _campaign_out/
```

A faithful, self-contained reproduction of the **Eto–Hamada–Nitta two-scalar
knot soliton** ([arXiv:2407.11731](https://arxiv.org/abs/2407.11731), PRL 135,
091603) — their exact energy functional and auxiliary-field relaxation, mapped
line-for-line onto the paper's equations, with the electric-sector normalisation
verified against the continuum result:

```bash
python examples/ehn_knot_soliton.py --demo    # normalisation + saddle + g2-vs-electric dilemma
```

## Releasing

Version history: [`CHANGELOG.md`](CHANGELOG.md) (structured) and
[`docs/releases/`](docs/releases/) (narrative release notes, latest
[`v0.0.4`](docs/releases/v0.0.4.md) — campaign robustness: the `FleetExecutor`
parallel fleet, self-healing Vast REST, and `reap --label`).

Maintainers: the runbook for cutting a tagged, Zenodo-archived release is
[docs/RELEASING.md](docs/RELEASING.md) (version bump → PR → GitHub Release →
Zenodo webhook archives → DOI backfill).

## License

MIT.

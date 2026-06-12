# jax-solitons

> **Status: pre-alpha (0.0.x).** The API is being designed in the open and
> **will** change without notice until 0.1. Nothing here is stable yet.

A general JAX engine for classical field-theory solitons — Faddeev-Skyrme
hopfions, gauged abelian-Higgs vortices, Gross-Pitaevskii vortex knots —
designed for GPU farms.

## Why another soliton code?

The field's first public GPU soliton codes (cuSkyrmion, soliton_solver, 2026)
are CUDA C / Numba and single-purpose. `jax-solitons` aims to be the
differentiable, composable, farm-scale counterpart, built on four design
commitments:

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
   discretizations admit none at any resolution (companion methods paper, in
   prep.). All canonical operators are stencil-local, so they JIT, vectorize,
   and shard without all-to-all communication; spectral paths ride
   [jaxDecomp](https://github.com/DifferentiableUniverseInitiative/jaxDecomp)
   when distributed FFTs are needed.
4. **Restartable, registered runs.** A `RunConfig` dataclass serialized into
   every output; full-state checkpoints (field + velocity/optimizer state +
   RNG key) via orbax; config-hashed run manifests; a thin sweep driver.

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
| `models/` | gauged abelian-Higgs | porting |
| `steppers/` | arrested (backtracking) flow; projected Adam; sphere-constrained velocity-Verlet; GPE split-step (imaginary + real time) | **working, validated** |
| `steppers/` | L-BFGS, ETDRK | porting |
| `topology` | area-form plaquette F_ij, Hopf charge (differentiable) | **working, validated** |
| `seeds` | rational-map hopfion ansatz | **working, validated** |
| `seeds` | solid-angle (VOS) minimal superflow, composition | porting |
| `runs` | `RunConfig`; full-state restartable checkpoints (bit-identical restart, gated in CI); config-hashed run dirs + manifest | **working, gated** |
| `measure` | implicit core-curve tracer (lax.scan predictor-corrector), Gauss linking, arc-length resampling | **working, gated** |

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
conservation identical to four decimals (memory-bandwidth-bound). The
engine's convention: **hunt in fp32, certify in fp64** — dtype is an explicit
`BoxGrid` parameter, never a global flag.

## Install (development)

```bash
pip install -e ".[test]"
pytest
```

## License

MIT.

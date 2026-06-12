# Validation

What is tested, where, and against what. Three kinds of evidence, in
decreasing order of strictness:

- **exact** — a mathematical identity of the discretization; any violation
  is a bug, never a tolerance question
- **analytic** — agreement with a closed-form solution of the continuum
  theory
- **reference** — agreement with published numerical results or with the
  validated source research engine

CPU tiers run in CI on every commit (`pytest`). The GPU tier runs on a
GPU box: `SOLITON_GPU_TIER=1 pytest tests/test_gpu_tier.py -v -s`.

## Exact identities (`tests/test_exact.py`, CI)

| Claim | Test | Why it's exact |
|---|---|---|
| Area-form flux quantization: the Berg-Lüscher plaquette sums to 4π × (integer degree) over every closed 2-torus slice, for arbitrary smooth fields | `test_area_form_flux_quantized_on_random_maps` | each plaquette is the exact solid angle of a spherical quadrilateral; closed surfaces tile. The naive same-index discretization of n·(∂ᵢn×∂ⱼn) has no such quantization at any resolution — this property is the library's reason to exist |
| Hopf charge of the rational-map seed (n,m) is n·m | `test_hopf_charge_integer_on_seeds` | degree of the composed map |
| Energy and Q_H invariant under lattice translations | `test_energy_translation_invariant` | periodic stencils; same summands |
| Energy and Q_H invariant under global O(3) target rotations | `test_energy_global_o3_invariant` | the energy is built from invariants of n |
| Constraint algebra: retraction idempotent and on-manifold; tangent projection orthogonal and idempotent | `test_constraint_algebra` | projector algebra (S² and CP¹) |
| Energy density integrates to the energy | `test_faddeev_energy_density_integrates_to_energy` (smoke) | same summands |
| vmapped dynamics ≡ per-sample dynamics | `test_vmap_batch_dynamics_matches_per_sample` (smoke) | batch axis is free (R2) |
| Checkpoint restart is bit-identical | `test_gate_checkpoint_restart_determinism` (acceptance) | full integrator state saved (R4) |

## Analytic solutions (`tests/test_gpe_analytic.py`, CI)

| Claim | Test | Reference value |
|---|---|---|
| Bogoliubov dispersion of the GPE vacuum | `test_bogoliubov_dispersion` | ω(k) = k√(g + k²/4) in healing units; measured to <1% at two k |
| Planar dark soliton | `test_dark_soliton_profile_energy_and_stationarity` | ψ = tanh(√g x); kink energy (4/3)·area at g=1; stationary in real time |
| Vacuum is the imaginary-time fixed point | `test_imaginary_time_relaxes_to_vacuum` | |ψ| → 1 |

## Reference results (acceptance gates + GPU tier)

| Claim | Test | Reference |
|---|---|---|
| Unit-hopfion charge held through monotone descent | `test_gate_unit_hopfion_charge_held` | source engine: Q_H ≈ 0.998 at the E ≈ 1152 minimum |
| Core tracer finds the hopfion ring | `test_gate_core_tracer_closed_ring` | geometry of the relaxed Q=1 state |
| Real-time energy conservation + charge retention | `test_gate_persistence_energy_conservation` | source integrator: dH = −0.000% static |
| Faddeev-Hopf spectrum E(2)/E(1) | `test_gate_vk_q1_q2_ratio` (GPU) | published 1.623 (Battye–Sutcliffe lineage); source engine 1.604; this engine 1.609 |
| Vakulenko–Kapitanskii floor | same (GPU) | theorem: E ≥ c·Q^(3/4) ⟹ ratio ≥ 2^(3/4) = 1.682 for continuum minima; gated at 7% lattice grace |
| Derrick/virial depth | same (GPU) | E2/E4: Q=1 → 0.929, Q=2 → 1.007 (CP¹ spinor-frame relaxation; source lattice-normal ≈ 0.91) |
| Cross-engine bit-identity | (port validation, recorded) | Faddeev seed E and Q_H match the source engine to the last digit at N=64/x64; GPE split-step matches to 9e-16 after 10 steps |

## Known limits

- fp32 endpoints match x64 to 4 significant figures for relaxation and to
  ~conservation level for dynamics (memory-bandwidth-bound); certify
  physics claims in x64 (`hunt fp32, certify x64`).
- Identification of knot topology in a strongly radiating state is
  unreliable for ANY tracer; quench first (a few hundred arrested-flow
  steps cannot create knotting — the area form protects, never ties),
  then identify. Measured case study 2026-06-12: an in-run mid-bath ID
  produced a false decay signature; basin-ID after quench corrected it.

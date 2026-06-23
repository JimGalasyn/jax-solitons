# Design spec — `models/skyrme.py` (the SU(2) Skyrme model)

Status: **proposed** (not yet implemented). Author-of-record: NWT binding-ledger
work. This spec is written against `DESIGN.md` P1–P10 and the `VALIDATION.md`
evidence tiers; it ships with its exactness tests or it doesn't ship (P8).

## 0. Why — and the scope boundary

The engine already has *Faddeev–Skyrme* (`models/faddeev.py`): an **O(3)/CP¹**
field whose solitons are labelled by the **Hopf charge** π₃(S²)=ℤ (knots /
Hopfions). What it does **not** have is *Skyrmions proper*: an **SU(2)/S³** field
whose solitons are labelled by **baryon number** B = deg(S³→S³), π₃(S³)=ℤ — the
nuclear solitons (B=1 nucleon, B=2 deuteron, B=4 alpha, …).

**Primary motivation (the honest one):** the canonical Skyrme model is the
*calibration ground truth* for the multi-soliton **binding** problem the NWT
binding-ledger is stuck on. We keep *citing* "the Skyrmion lesson" (the B=2
bound state is a dedicated toroidal minimisation, not two B=1 solitons brought
together; binding lives in the relative-iso/attractive channel) as the authority
for why naive composition fails. With the model in-engine we can *test our own
methods* — the rigid composition engine, the soft-pin polarisation relaxer —
against a system with **published answers** (Battye–Sutcliffe) before trusting
them on the NWT Hopfion carrier.

**Secondary:** the SU(2) Skyrme model is the most-studied topological soliton in
nuclear physics; "handles Skyrmions out of the box" materially broadens the
engine's reach at low marginal cost (P1: it is a *configuration* of `Model`, not
a new solver).

### In scope (v1 — the thin module)
- `skyrme_model(c2, c4, m_pi=0.0) → Model` (Tikhonov/standard L2+L4, optional L0).
- An **exactly-quantised** baryon-number charge `baryon_charge(state, grid)`.
- S³ state + constraint (reuse, see §2).
- Seeds: B=1 hedgehog (rational map degree 1) and the multi-B rational-map /
  product ansatz (B=2 torus at minimum).
- Validation gates to B=1 and B=2 (§6).

### Out of scope (v1 — explicitly *not* built)
- MeV calibration / pion-mass fitting to nuclear data.
- B>8, fullerene/crystal Skyrmions, the infinite-B lattice.
- Semiclassical **quantisation** (rigid-rotor isospin/spin, Finkelstein–Rubinstein
  constraints) — that's a research programme, not a module.
- Vibrational spectra. These are deliberately deferred; v1 is the *classical
  static energy + topology* engine that the binding work needs.

## 1. Physics — the target (P-citable)

Field `U(x) ∈ SU(2)`. Left current `L_i = U†∂_iU` (su(2)-valued). Static energy

  E = ∫ d³x [ c2·(−½ Tr L_iL_i) + c4·(−1/16 Tr[L_i,L_j]²) + c0·m_π²(1 − ½Tr U) ]

- L2 (`c2`) — the σ-model kinetic term.
- L4 (`c4`) — the Skyrme quartic; the term that (as in Faddeev) supplies Derrick
  stability. Single soliton size is set by the c2/c4 virial balance.
- L0 (`c0`,`m_π`) — optional pion mass; **default 0** (massless v1, like the
  Faddeev default). Needed later for the B≥2 *binding* magnitudes to match nuclei,
  but not for the topology / structure gates.

**Baryon number** B = −(1/24π²) ∫ εⁱʲᵏ Tr(L_iL_jL_k) d³x  = deg(U).
This is the conserved topological charge, the analog of `hopf_charge`.

## 2. State + constraint — REUSE, don't rebuild (P1, R1)

The decisive simplification: **SU(2) ≅ S³, and CP¹ is already an S³ field.**
Represent U by a real unit 4-vector φ=(φ0,φ1,φ2,φ3), |φ|=1, via
`U = φ0·𝟙 + i φ_a σ_a` (a=1,2,3). Then:

- The constraint is **unit-norm on 4 components** — i.e. exactly today's
  `CP1Constraint` (which is `S2Constraint` "one component higher"). Generalise the
  pair to a single `SphereConstraint(d)` (`d=3` → S², `d=4` → S³/CP¹/SU(2)); the
  projector/retraction algebra is identical and already exactness-tested
  (`test_constraint_algebra`). **No new constraint code, just a dimension.**
- In φ-coordinates the energy is an **O(4) σ-model**: with |φ|=1,
  - −½Tr L_iL_i = ∂_iφ·∂_iφ  (the O(4) gradient — structurally `E2Term`, 4 comps),
  - −1/16 Tr[L_i,L_j]² = (∂_iφ·∂_iφ)(∂_jφ·∂_jφ) − (∂_iφ·∂_jφ)²  (the Skyrme
    quartic as an O(4) invariant),
  - 1−½Tr U = 1−φ0  (the mass/potential).
  So the terms are O(4) analogs of the existing O(3) terms — same forward-diff
  stencil discipline (P3), 4 components instead of 3, a different quartic
  contraction. This is the bulk of the new code, and it is small.

State layout: `state` is shape `(4, *grid)` (real), batch axis free (P2/R2),
documented in the module docstring as the model's layout (per `model.py`).

## 3. The hard part — an exactly-quantised baryon density (P3, P8)

This is the one genuinely new exactness deliverable; everything else reuses.
The naive lattice `εⁱʲᵏ Tr(L_iL_jL_k)` is **not** integer-quantised at finite
resolution (exactly the failure the area form fixes for the Hopf index). The
baryon number is a degree S³→S³, so the exact construction is the 3-D analog of
the Berg–Lüscher solid-angle area form:

- Triangulate each lattice cube into 6 tetrahedra (a fixed Kuhn/Freudenthal
  decomposition, shardable with halo — P3).
- The image of each tetrahedron's 4 vertices is 4 points on S³; the **signed
  spherical-volume of the geodesic 3-simplex on S³** (oriented, in units of the
  S³ volume 2π²) is the exact local degree contribution.
- Sum over tetrahedra → exactly an integer for any closed config; differentiable
  for descent. Cite the same lineage as the area form (solid-angle exactness),
  one dimension up.

Acceptance: **`baryon_density` sums to 4π²·... → exact integer on adversarial
random unit-4-vector fields** (Tier-1, the analog of
`test_area_form_flux_quantized_on_random_maps`). Tolerance wider than float
rounding = design smell (P8). This test is the gate the module is built around.

(Implementation note: the signed S³-simplex volume has a clean closed form via
the 4×4 determinant of the vertex 4-vectors plus an arccos normalisation; verify
the orientation convention against a known degree-1 hedgehog before trusting
multi-B.)

## 4. Seeds (`seeds.py` additions)

- **B=1 hedgehog**: U = exp(i f(r) r̂·σ), f(0)=π, f(∞)=0 → φ0=cos f,
  φ_a=sin f·r̂_a. Profile f a smootherstep (same C² profile machinery as
  `seed_hopfion_from_curve`). deg = 1.
- **Multi-B rational map** (Houghton–Manton–Sutcliffe): U from a degree-B rational
  map R(z) composed with a radial profile; B=2 is R(z)=z² → the axially-symmetric
  **torus** (the deuteron). This gives the known multi-soliton seeds directly.
- **Product ansatz** U_A·U_B for two separated B=1's — the input to the rigid
  composition / soft-pin binding tests (the whole point: feed the *same* method
  we use in NWT and check it against the known Skyrme B=2).

## 5. Public API

```python
from jax_solitons.models.skyrme import skyrme_model, baryon_charge
from jax_solitons.seeds import skyrmion_hedgehog, skyrmion_rational_map

model = skyrme_model(c2=1.0, c4=1.0, m_pi=0.0)   # Model: terms + SphereConstraint(4) + (baryon_charge,)
state = skyrmion_hedgehog(grid, B=1)             # (4, N, N, N), |phi|=1
state, _ = adam_flow(model, state, grid, lr=...) # SAME stepper (P1)
B = baryon_charge(state, grid)                    # -> ~integer
```

`skyrme_model` returns a `Model(name="skyrme", terms=(E2_O4, c4*E4_skyrme[, c0*E0]),
constraint=SphereConstraint(4), charges=(baryon_charge,))`. Steppers, campaign,
checkpoint, batch — all unchanged (P1, P4).

## 6. Validation gates (`VALIDATION.md` tiers)

**Exact (`tests/test_skyrme_exact.py`, CI):**

| Claim | Why exact |
|---|---|
| Baryon density → integer degree on adversarial random S³ fields | signed S³-simplex volumes tile a closed 3-manifold (the area-form argument, +1 dim) |
| B of the rational-map seed (degree d) is d | degree of the composed map |
| Energy + B invariant under lattice translation | periodic stencils |
| Energy invariant under global O(4)/chiral SU(2)×SU(2) target rotation | energy built from φ-invariants |
| `SphereConstraint(4)` retraction idempotent/on-manifold; projection orthogonal/idempotent | shared projector algebra (already tested for d=3,4) |

**Analytic / reference (acceptance + GPU tier):**

| Claim | Reference |
|---|---|
| Faddeev–Bogomolny bound: E ≥ 12π²·c·|B| (massless) | continuum inequality; descent endpoint must respect it |
| B=1 hedgehog energy ≈ 1.23 × bound (massless, c2=c4=1) | standard Skyrme result |
| B=2 minimiser is the **torus**, E(B=2) < 2·E(B=1) (binding!) | Battye–Sutcliffe 2002 |
| B=1..8 minimal energies + symmetries (B=4 cube, B=7 dodec.) | Battye–Sutcliffe ratios |

**Binding-ledger cross-check (the reason we're building it):**

| Claim | Ties to |
|---|---|
| Product ansatz of two B=1's is repulsive in the "wrong" relative iso-orientation, attractive in the right one (relative-iso-π channel) | our `rigid_link_potential` iso-orientation scan |
| Soft-pin / constrained relaxation of the B=2 product ansatz recovers the torus binding | our soft-pin polarisation relaxer — *the* calibration |

The last row is the deliverable that retroactively justifies the module: if our
soft-pin method reproduces the **known** Skyrme B=2 binding, we trust it on the
NWT deuteron; if it doesn't, we've found the method's blind spot on a system
where we know the answer.

## 7. Effort + sequencing

- O(4) terms + `SphereConstraint(d)` generalisation: small (reuse).
- **Exact baryon density** (the S³-simplex sum) + its Tier-1 test: the real work,
  the module's centre of gravity.
- Seeds (hedgehog + rational map): small.
- Gates to B=1 (analytic) and B=2 (reference + binding cross-check): the
  acceptance bar.

Estimate: a focused **few days** for v1-thin to the B=2 gate; *not* the months-long
nuclear-physics-validated programme (§0 out-of-scope). **Sequencing:** pick this
up *after* the current N=256/288/320 thin-tube ladder reports — that result
decides whether we need the Skyrme cross-check urgently or can build it calmly.

## 8. Open questions (decide at implementation)
- Kuhn vs Freudenthal cube triangulation — pick the one whose orientation
  bookkeeping is least error-prone; pin it with the B=1 hedgehog sign check.
- Generalise `S2Constraint`/`CP1Constraint` → `SphereConstraint(d)` now, or add
  S³ alongside and unify later? (Lean unify: one constraint, exactness-tested at
  d=3,4.)
- Massless v1 vs ship L0 immediately — L0 is one cheap term and the B≥2 binding
  magnitude is sensitive to it; include it but default `m_pi=0`.

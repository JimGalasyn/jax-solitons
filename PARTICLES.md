# The particle catalog and scene composition

This note designs the **separation of particle *generation* from scene
*composition***: relax each fundamental soliton once (expensive), cache it as a
first-class object, then stage a scene by *placing* cached particles and running
a short settle — no per-run relaxation. It is the cached-object form of the
`TODO.md` item "VOS minimal-superflow seeds + multi-soliton composition", and
the computational backing for [COMPENDIUM.md](COMPENDIUM.md).

Status: **design only** — no code yet. Written so the `ParticleCatalog` +
`compose()` build proceeds against a fixed shape.

## Why caching is clean (and what makes a soliton a reusable object)

A relaxed soliton **decays to exact vacuum beyond its core radius** — the
rational-map seed already guarantees this (`d >= w` ⟹ vacuum), and a relaxed
state inherits it. Spatial compactness is what makes a particle reusable: two of
them placed far apart barely interact until evolved. And the storage primitive
already exists — `runs.save_checkpoint` serializes the full state + `RunConfig`
+ `config_hash`. A catalog entry is just a checkpoint keyed by *physics identity*
rather than *run identity*.

## `ParticleCatalog`

A content-addressed library of canonical relaxed states. Each entry carries:

- **field** — the relaxed configuration, stored at a high reference resolution /
  fp64 so it can be **resampled** onto any target grid (a particle relaxed at
  N=64, L=18 must re-instantiate at N=256, L=72);
- **identity** — model, topological charge (e.g. Q_H), couplings (c4), the
  symmetry it represents;
- **measured validity** — E, E₂/E₄ virial ratio, core radius (a scene builder
  needs the particle's size and whether to trust it);
- **provenance** — the relaxation `config_hash` + the run/Vast ledger, so every
  cached particle is reproducible (P4/P6: a particle that can't name its
  relaxation does not enter the catalog).

**Antiparticles are derived by symmetry, not separately relaxed:** the positron
is the parity/orientation conjugate of the electron (`n → reflected`, or
`Z → conjugate`). Store fundamentals; derive conjugates.

## `compose(scene)`

```python
scene = [ place(electron, at=(-X, 0, 0), boost=+v * x̂),
          place(positron, at=(+X, 0, 0), boost=-v * x̂) ]
field, momentum = compose(scene, grid=BoxGrid(N=256, L=72))
evolve(field, momentum, steps=...)        # relax/integrate *that* scene
```

`compose` resamples each cached particle onto the target grid → translates /
rotates → **glues** (not adds) → returns the field **and** the conjugate
momentum.

## The three gotchas (where the physics actually lives)

1. **You cannot superpose constrained fields.** `n₁ + n₂` violates `|n| = 1`.
   For well-separated, vacuum-decaying particles the fix is **gluing** (a smooth
   partition-of-unity patch, which the seed machinery already does); for
   close-packed cores it is the **product ansatz** (compose in the group rep —
   Skyrme's SU(2) multiplication). Far apart = clean; overlapping cores = hard.
2. **A boost is not just a velocity vector.** The Faddeev model is relativistic:
   "moving toward each other" means each particle needs its **Lorentz-boosted
   profile *and* its conjugate momentum** `π = ∂ₜn`, or the dynamics will not
   carry it. That is why `compose` returns `(field, momentum)`, and why the
   sphere-constrained velocity-Verlet stepper (already shipped) is the right
   evolver. (The GPE is the easy case — velocity is a phase imprint.)
3. **The post-composition settle is real physics, not cleanup.** Relaxing the
   scene radiates the gluing seams and lets the particles find their
   interaction — that is the point ("relax *that* scene") — so the composite is
   not a static solution. Use the **dynamical** stepper, never arrested descent,
   or you quench the collision you meant to stage.

## How it fits the rest

- **Fleet vs local.** Catalog *generation* is the expensive, fleet-scale job
  (deep-relax each fundamental once — exactly what the Vast run does); *scene
  composition* is cheap and local. The farm builds the catalog; staging uses it.
- **Seeds.** `compose` is a new kind of seed — `seeds.from_catalog(...)` — that
  rides the same constraint/retraction machinery as the rational-map ansatz.
- **Reproducibility.** Every catalog entry and every composed scene is a
  `RunConfig` with a hash; a scene is itself a campaign input (P4/P6).

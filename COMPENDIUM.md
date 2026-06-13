# The compendium

The **compendium** is the project's notional collection of particles — destined
to be a browsable "periodic table" on a public site, backed by a database of
each particle's properties, both **derived** (from theory) and **computed**
(from the solver). This note designs that database and, more importantly, the
**firewall** that keeps it honest science rather than numerology-with-rendering.

Status: **vision / design only.** The computational backing is
[PARTICLES.md](PARTICLES.md); the engine itself stays neutral (see the firewall
below).

## Three layers per particle

Each row of the periodic table joins three sources:

| Layer | Source | Examples |
|---|---|---|
| **Derived** | the theory's analytics | mass, couplings, winding numbers (p,q), Koide angle, charges |
| **Computed** | the `jax-solitons` solver | relaxed E, E₂/E₄, topological charge, core geometry / size, stability, collision outcomes |
| **Field data** | the relaxed configuration itself | renders the periodic-table visuals **and** seeds scene composition |

The **Field data** and most of **Computed** come straight from the
`ParticleCatalog` (PARTICLES.md): each catalog entry *is* a row's computational
half — content-addressed, provenance-stamped (relaxation `config_hash` + ledger),
regenerable on the fleet. "Click electron → its relaxed core curve, its E₂/E₄,
beside its derived mass" is one row.

## Schema sketch

```
ParticleRow:
  id:                  str          # stable slug, e.g. "electron"
  name, symbol:        str
  derived:             dict         # theory predictions (mass, p, q, charges, ...)
  computed:            dict         # solver observables (E, E2/E4, Q, core_radius, ...)
  field_ref:           str          # ParticleCatalog content hash -> the relaxed state
  model:               str          # WHICH solver model produced `computed` (e.g. "faddeev")
  correspondence:      str          # "unvalidated" | "conjectured" | "validated"
  provenance:          dict         # relaxation config_hash, ledger ref, code version
```

Two fields carry the scientific weight: **`model`** (what was actually computed)
and **`correspondence`** (whether that computed object is *believed/known to be*
the theory's particle).

## The firewall (the load-bearing part)

The engine deliberately computes **neutral** soliton physics — the gallery rule
is "engine physics, no downstream interpretation." For the **Computed** column
to legitimately mean *"the electron's properties,"* the cached soliton must
actually **be** the theory's electron. Concretely, in the Null Worldtube
ontology a particle is a **(2,1) torus-knot null worldtube** (toroidal EM /
self-sustaining oscillation), which is closer to the **gauged abelian-Higgs**
model (still "porting" in the repo) than to the bare Faddeev-Skyrme S² hopfion
the engine relaxes today. Therefore:

- **Derived and Computed are separate columns**, never silently merged.
- Every row names its **`model`** and a **`correspondence`** status. A `faddeev`
  hopfion standing in for an electron is `correspondence: unvalidated` — and the
  site must show it as such.
- The open scientific question — **does the theory's (n,m)-winding object carry a
  Hopf charge, and which one?** — is a real, publishable correspondence result.
  Flipping a row to `validated` requires establishing that map, not asserting it.

This separation is exactly what makes the compendium credible: the table can
display a beautiful relaxed hopfion next to the electron's derived mass while
being explicit that the *identification* is not yet earned.

## Build order

1. **`ParticleCatalog` + `compose()`** (PARTICLES.md) — neutral, model-agnostic;
   works today with hopfions. Enables rapid scene composition now.
2. **The compendium schema** — the derived⊕computed⊕field join, with explicit
   `model` + `correspondence` fields, backed by the catalog.
3. **The right model** — gauged abelian-Higgs (or a dedicated NWT model) so the
   Computed column is the *correct* object; then individual rows' correspondence
   can be argued toward `validated`.
4. **The website** — render the DB as the browsable periodic table; optionally
   let visitors stage scenes (PARTICLES.md `compose`).

## First entry, honest from row one

The deep-relaxed **Q=1 hopfion** measured on real hardware
(E = 1108.0, Q_H = 0.9985, E₂/E₄ = 0.929) is **computed-side row #1**:
`model = faddeev`, `correspondence = unvalidated`. That honesty — a real,
reproducible computation explicitly *not yet* claimed to be a physical
particle — is the whole point of the firewall.

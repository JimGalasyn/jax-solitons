# The compendium — and what jax-solitons contributes to it

The **compendium is defined canonically elsewhere**: `null-worldtube/docs/`
`compendium/` is a populated SQLite database (`nwt_particles.db`, built by
`build_db.py`) of **27 particles** — leptons, quarks, gauge bosons, mesons,
baryons, mixing parameters — destined for a browsable "periodic table" website.
This note is **not** a redefinition of that schema; it describes the one layer
`jax-solitons` adds to it, and the firewall that keeps the addition honest.

Status: **design / integration note.** The compendium and its DB live in
`null-worldtube`; the catalog mechanics are in [PARTICLES.md](PARTICLES.md).

## What the compendium already has (and what it doesn't)

Each row carries NWT **topology** (`p`, `q`, `genus`, `carrier`, `m_radial`,
`n_q`), a **mass** computed from that topology (`formula_text`, `computed`),
the **PDG** value and residual, integer traceability, family, and papers. So
the `computed` column is the **NWT analytic mass** — not a field-solver output.
Example: the electron is `(p=2, q=1, genus=0, carrier="unknot", m_radial=3)`,
`computed = 0.511 MeV` vs PDG to −0.0002%.

What it does **not** have is any check that a **field-theoretic soliton with the
claimed topology actually exists and is stable**. That is the gap `jax-solitons`
fills.

## The carrier vocabulary IS the soliton classification

The compendium's `carrier` column is a knot/manifold type — and it is exactly
what this engine measures (Hopf charge + core-curve knot ID):

| `carrier` | particles | what jax-solitons computes it as |
|---|---|---|
| **hopf** | W, Z, Higgs | Faddeev **Hopf charge** Q_H (the relaxed hopfion) |
| **unknot** | e, μ, τ | a relaxed soliton whose **core curve is an unknotted ring** |
| **trefoil** | u, d, s, c, b, t | core curve = **trefoil** — the `measure` tracer + Alexander determinant (the trefoil-determinant gate in TODO.md) |
| **cinquefoil** | ν₁₋₃, P_c pentaquarks | core curve = **(2,5) torus knot** |
| S²×S² | mesons | (a different target manifold) |

The topology tower is literal: `(2,1)` unknot → `(2,3)` trefoil → `(2,5)`
cinquefoil, genus 0→1→2. **`jax-solitons` is the natural existence-and-stability
checker for every one of these claims** — relax the field, trace the core,
identify the knot / count the Hopf charge, compare to `carrier`.

## The integration: a solver-evidence table

`jax-solitons` contributes a table linked to `particles` (by id / carrier), the
field-theoretic half:

```
soliton_evidence:
  particle_id:        FK -> particles.id
  model:              str     # e.g. "faddeev", "gauged-abelian-higgs"
  exists:             bool    # a stable soliton was found
  hopf_charge:        int     # measured Q_H (area form)
  core_knot:          str     # identified core-curve knot type
  E, E2_over_E4:      real    # relaxed energetics
  field_ref:          str     # ParticleCatalog content hash (renders + composition)
  correspondence:     str     # unvalidated | conjectured | validated
  provenance:         json    # relaxation config_hash + ledger + code version
```

A row's `correspondence` flips to `validated` only when the solver produces a
**stable** soliton whose **measured topology matches the compendium's
`carrier`** — not when a formula agrees.

## The firewall (unchanged in spirit, sharper in form)

The NWT `computed` mass (analytic, from topology) and the solver evidence
(field-theoretic existence) are **two independent computations of the same
claim**. The engine stays neutral ("engine physics, no downstream
interpretation"); the *join* is where interpretation happens, and it is explicit:
`model` + `correspondence` per row. A hopfion standing in for a boson is
`unvalidated` until the gauge sector is right and the match is shown.

## First evidence — and a correction

The deep-relaxed **Q=1 hopfion** measured on real hardware
(E=1108.0, Q_H=0.9985, E₂/E₄=0.929) is existence evidence for the
**`carrier = "hopf"`** family — **W, Z, Higgs — not the electron** (the electron
is a `unknot` carrier; a hopfion is not). Its row: `model=faddeev`,
`exists=true`, `hopf_charge=1`, `correspondence=unvalidated` (the bosons need
the gauge sector, still "porting"). Getting *which family the evidence supports*
right, from the first row, is the firewall working as intended.

## Build order

1. **`ParticleCatalog` + `compose()`** (PARTICLES.md) — neutral, works today.
2. **`soliton_evidence`** table + a writer that runs a relax-then-ID campaign
   per `carrier` and records existence/knot/Hopf-charge against the DB.
3. **Knot ID** (`measure` Alexander determinant — the trefoil gate) and the
   **gauge sector** (gauged abelian-Higgs), so knotted and boson carriers can be
   checked, not just the bare hopfion.
4. The website (in `null-worldtube`) joins NWT mass + solver evidence per row.

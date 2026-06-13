# The compendium, and why jax-solitons does *not* (yet) feed it

The **compendium is defined canonically in another repo**:
`null-worldtube/docs/compendium/nwt_particles.db` — a SQLite database of the
project's particle catalogue, destined for a browsable "periodic table" site.
This note exists to draw an **honest boundary**: the compendium and this engine
are different objects, and a recent audit showed the naive bridge between them
*fails*. Documenting that keeps future work from conflating them.

Status: **boundary note.** No code; no claim of a jax-solitons → compendium link.

## What the compendium actually is (post-rebuild)

As of the Paper-21a rebuild (`build_db_v21a.py`), the DB holds **25 K₇ closed
walks + 4 predicted species** — charged leptons and the hadron spectrum. Each
row carries NWT **topology** (`(p,q)`, `carrier`, `n_q`, Steane syndrome) and a
**Paper-6 mass**. The `carrier` is a *substrate-algebra* label, mapped by sector:

| carrier | sector | particles |
|---|---|---|
| unknot | lepton | e, μ, τ |
| **hopf** | **meson** | π, K, D, J/ψ, ω, Υ, ρ, η |
| trefoil | hyperon | τ-class, Λ, Σ*, Ξ, Δ, Ω |
| cinquefoil | nucleon | p, n, Σ |

(Note `hopf` is the **meson** carrier here — not bosons. Quarks, neutrinos,
and gauge bosons are *not* K₇ walks and are absent by design.)

## Why jax-solitons does not feed it (the audit finding)

The tempting story was: jax-solitons relaxes a field, traces its core curve,
identifies the knot / Hopf charge, and confirms the compendium's `carrier`. **That
story is refuted by Paper 21a §"Walk-knot versus carrier-knot":** the carrier is
fixed by the *substrate algebra* (σ-orbit + Φ-shell via rule-I), **not** by any
3-D embedding. Lifting each walk to a 3-D embedding and reading its knot gives
**mostly the unknot, with ≤14% agreement** with the carrier. So a classical
field soliton's core-curve knot — exactly what this engine measures — would *not*
match the compendium's carrier. There is no direct walk-to-field-soliton map.

Two consequences, stated plainly:
- **The NWT compendium objects are substrate-algebra K₇ walks, not classical
  field-theory solitons.** jax-solitons relaxes the latter; the compendium
  catalogues the former. Different ontologies.
- Whether any NWT walk has a **classical-field-soliton realization at all** is
  an open question this engine has *not* answered — and §178 is evidence against
  the obvious topological correspondence.

## The firewall, at its most conservative

No relaxed soliton in this engine is identified with a compendium particle. The
deep-relaxed Q=1 Faddeev hopfion measured on Vast is a **hopfion** — a result in
classical field theory on its own terms — **not** a row in `nwt_particles.db`
(it isn't a meson, and its core knot isn't the substrate carrier). If a bridge
between the two is ever established, it will be a *result*, derived and checked,
never an assumption smuggled in by shared vocabulary.

## What this leaves

jax-solitons stands on its scope (README): a differentiable, farm-scale engine
for classical field-theory solitons. The particle **catalog + scene composition**
in [PARTICLES.md](PARTICLES.md) is about *this engine's own* solitons (energies,
charges, core curves, collisions), with no claim on NWT identity. The compendium
is the NWT project's artifact and is maintained there. The two may inform each
other someday; today they are kept apart, on purpose.

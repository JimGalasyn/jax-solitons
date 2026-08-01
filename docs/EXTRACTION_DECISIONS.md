<!--
MIGRATED 2026-08-01 into jax-solitons, verbatim below this comment.

Origin: null-worldtube-private, EXTRACTION_DECISIONS.md, written 2026-07-29.
That repo is being retired, and this document is cited twice from here -- by
CHANGELOG.md and by src/jax_solitons/ehn/__init__.py -- as the REASON the EHN
engine is a package module rather than an example. A design rationale cited by
surviving code should not live in a repo that is going away; its own opening
paragraph makes the argument better than I can:

    "A decision that lives only in a conversation is a decision that gets made
    twice."

Nothing below is edited. Two notes on reading it now:

  - Its "Status: DECIDED, NOT YET EXECUTED" is stale. Steps 1-3 were executed on
    2026-08-01: the engine landed as jax_solitons.ehn (PR #77) and the relabelled
    compendium as soliton_playground.ehn_lab. Step 4's generic instruments landed
    too, but split differently than planned -- the certificate and envelope
    frameworks went to soliton-playground, the topology measures here. Step 5,
    the run-farm preflight wrapper, is not built.

  - Its CARRY-ONS section warned of unpushed commits. Those were found and
    pushed on 2026-08-01: three branches in null-worldtube-private carrying 8
    commits that existed on no remote. Its claim that "push is blocked by policy
    on every remote, so a human has to do it" was true of the machine it was
    written on and not of Bender.

WHERE THE FILES IT NAMES LIVE NOW. The body cites them at their 2026-07-29
locations; this table is the translation, so a reader does not chase paths into a
repo that is going away. The body itself is unedited.

  ehn_relax.py, ehn_energy.py, ehn_knot_batch.py, ehn_cross_linking.py
        -> jax_solitons/ehn/{relax,energy,knot_batch,cross_linking}.py  (#77)
  simulations/gpe_vortex_topology.py
        -> jax_solitons/vortex_topology.py                              (#77)
  standard_box.py, particle_catalog.py, chamber.py
        -> soliton-playground src/soliton_playground/ehn_lab/
  compose_pair.py
        -> soliton-playground experiments/compose_pair.py
  BESTIARY.md
        -> soliton-playground docs/BESTIARY.md

  gauged_relaxer/core_knot_id.py  NOT migrated. Its knot identification was
        superseded by jax_solitons.knots.identify_knot, which
        vortex_topology.knot_determinants now calls; its core_curves_from_n and
        curve_energy_scores have no equivalent here.
  run_ehn_relax_fleet.py, vu_confirm.py, papers/numerical_methods_outline.md
        NOT migrated -- still only in null-worldtube-private.

  build_db_v21a.py, cited from COMPENDIUM.md, is in null-worldtube (a DIFFERENT
        repo, not the one being retired) and is safe there.
-->

# Extraction decisions (2026-07-29)

Where the code in this repo is going. Written down because three times in one
session I re-derived something this repo already knew — the flat on-box layout
(documented in `ehn_cross_linking.py`), the relax-then-ID discipline (in
`nwt-audit`'s PREREG), and RunPod SECURE (in `run_ehn_relax_fleet.py`,
`vu_confirm.py` and `numerical_methods_outline.md`). Each cost time or money. A
decision that lives only in a conversation is a decision that gets made twice.

Status: **DECIDED, NOT YET EXECUTED.** Nothing moves until the in-flight B2 leg
lands (moving the engine changes `standard_box.py`'s `import ehn_knot_batch as EK`
and would break the pending local scoring pass).

## DECIDED

**The EHN engine goes to `jax_solitons/ehn/`** — a package module, not an example.

  ehn_relax.py · ehn_energy.py · ehn_knot_batch.py · ehn_cross_linking.py

Rationale:
- `standard_box.py` does `import ehn_knot_batch as EK`. Examples are not
  importable, so an example could not be driven by the battery or the compendium.
- The slot is already taken by a *different* thing worth keeping:
  `examples/ehn_knot_soliton.py` is the published-benchmark reproduction (linked
  rings, `build_ic_knot` = "nlink phi1 rings threaded on one phi2 ring"). It has NO
  torus-knot seed — no `tp`/`tq` — so it cannot produce the trefoil/cinquefoil/
  septafoil/T(3,4)/T(3,5) results the retrospective salvages. The two coexist:
  **example = the published N_link>=4 benchmark; module = the torus-knot extension
  below that floor.** Document the relationship; do not merge them.
- The formatter hazard that argued against jax-solitons is GONE: provenance is now
  a git commit, not a content hash of source files, so a `ruff --fix` no longer
  orphans certificates. This decision depended on that one.

It needs its own stepper. EHN relaxation is three interleaved updates (Eqs
12/13/11) with a Gauss-law iteration and an auxiliary field — not a
`terms + constraint + charges` energy that `arrested_flow` can drive. `steppers/`
is the home.

**The relabelled compendium goes to soliton-playground.** Ten entries, now
knot-theoretic only (`electron` -> `unknot_framed_twist1`, `lepton_bare` ->
`unknot_bare`, "Neutron-labeled" stripped; 0 of 10 carry an SM label in any
descriptive field). The charter conflict was the relabelling, and the relabelling
is done.

Two conditions on landing it:

1. **Fold it into `BESTIARY.md`'s schema, do not add a second catalog.** The
   bestiary already carries `preset` + `protecting_charge` + a NAMED CLOCK per
   entry. Adopting that makes the collision this session started with legible
   instead of confusing:

       trefoil T(2,3)  preset gpe-dimensionless  charge none (knot type)  UNSTABLE
       trefoil_t23     preset ehn-two-scalar     charge Lk = -3 lock      HELD 36k

   Same knot, opposite verdicts, distinguishable at a glance. That is precisely
   what the preset/protecting_charge fields were added for. Two catalogs with two
   schemas would re-create the ambiguity they were introduced to kill.

2. **`source_out_dir` keeps `out_electron_n192` / `out_lepton_bare_n192`.** These
   are historical filesystem paths -- `out_electron_n192` is still in this tree --
   and rewriting them to satisfy a regex would falsify provenance rather than
   honour the charter. The charter forbids identifying a STRUCTURE with a particle,
   not recording where bytes were written in July. This needs to be an explicit
   exemption in soliton-playground, since that is the repo with the strict rule;
   noted in each entry's provenance_note.

## STILL OPEN

Nothing blocking. Both destinations are decided.

## ORDER, with a real acceptance gate

1. Land the in-flight B2 leg; score it; bank the citable certificate.
2. Move the four engine files to `jax_solitons/ehn/`. Adjust ONLY non-hashed
   files — though note this matters less now that provenance is git-pinned.
3. **Acceptance test**: re-run `--battery --quick --legs B2` from the new layout
   and require the physics to match bit-for-bit:
       lk = -3.0001967524330566 / +3.0001572806935375   det 3   486 segments
       Q = -2.6319014015507802   total E = 3668.097642218374
   Unusually strong for a refactor: the answer is bit-reproducible, so the move is
   either exactly right or provably wrong. `engine_sha` WILL change (it is a
   commit now) — that is expected; the physics must not.
4. Extract the generic instruments (certificate framework, envelope framework,
   topology measures) into jax-solitons, deduping against `measure.py`/`knots.py`
   — `knot_determinants` vs `identify_knot` is near-duplicate work.
5. Build the run-farm preflight wrapper against the settled layout, not before.

## CARRY-ONS

- 12 unpushed commits here; 1 in jax-solitons on branch
  `knot-labels-drop-particle-sectors`. Push is blocked by policy on every remote,
  so a human has to do it. They must travel with whatever moves.
- The dependency graph spans THREE directories and broke two rented runs:
  `engine_dogfood/`, `simulations/gpe_vortex_topology.py`,
  `gauged_relaxer/core_knot_id.py`. Whatever the destination, the payload is 7
  files, not 5.
- `certificates/*/entry.json` record `source_out_dir` and `git_commit` pointing
  into THIS repo. Those references dangle after a move; keep the provenance notes.

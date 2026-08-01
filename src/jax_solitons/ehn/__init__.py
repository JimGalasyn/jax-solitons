"""Eto-Hamada-Nitta gauged two-scalar knot solitons — the torus-knot extension.

The faithful EHN scheme (arXiv:2407.11731 supplemental, their Eqs. 5/11/12/13):
naive 2nd-order central differences, an independent auxiliary field B_i softly
constrained to curl A, and A0 solved by a separate Gauss-law iteration with its
own step beta. The whole system co-relaxes from a knot initial condition with the
Chern-Simons coupling on from the start, which is what keeps it inside the linked
sector instead of letting B decouple from the phi1 strings.

RELATIONSHIP TO `examples/ehn_knot_soliton.py` — the two coexist deliberately and
must not be merged:

    example = the published benchmark. `build_ic_knot` places N_link phi1 rings
              threaded on one phi2 ring; reproduces EHN's own N_link >= 4 results.
    module  = this package. Adds the T(p,q) torus-knot seed (`--geom torus`,
              `--tp`/`--tq`), which is what produced the held trefoil T(2,3) at
              N_link = 3 — BELOW the floor EHN report — plus T(2,5), T(2,7),
              T(3,4) and T(3,5). The example has no torus seed and cannot make
              those states.

Provenance: moved here from `null-worldtube-private/simulations/engine_dogfood/`
per that repo's EXTRACTION_DECISIONS.md (2026-07-29), migrated here 2026-08-01 as
docs/EXTRACTION_DECISIONS.md since that repo is being retired. It chose a package module
over an example because the SB-1 battery and the particle catalog both `import`
the engine, and examples are not importable. The move's acceptance test is
bit-reproducibility of the N=96 quick battery, since the physics is deterministic:
the refactor is either exactly right or provably wrong.

Still to do (recorded so it is not rediscovered): EHN relaxation is three
interleaved updates plus a Gauss-law iteration and an auxiliary field, not a
`terms + constraint + charges` energy that `arrested_flow` can drive. It wants a
real stepper under `steppers/`. Deliberately NOT done as part of the move, because
rewriting the integrator and relocating it in one step would leave a bit-exactness
failure with two possible causes.

  python -m jax_solitons.ehn.relax --geom torus --tp 2 --tq 3 --R 33.792
"""
__all__ = ["cross_linking", "energy", "knot_batch", "relax"]


def __getattr__(name):
    """Lazy submodule access (PEP 562).

    Deliberately not `from . import relax` at module scope, for two reasons:
    importing the package would then pull in JAX and compile-time machinery even
    for a caller that only wants `cross_linking`; and `python -m
    jax_solitons.ehn.relax` would find `relax` already in sys.modules and run it a
    second time, which runpy warns about as "unpredictable behaviour". The battery
    drives the relaxer through exactly that -m entry point.
    """
    if name in __all__:
        import importlib
        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

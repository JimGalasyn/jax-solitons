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

WHY WE REACH N_link = 3 AND EHN DID NOT — the seed, not the step size.
The obvious competing explanation was descent aggressiveness: EHN's alpha=4e-4
would sit at ~82% of this functional's stability bound (2/H = 2.5e-4) against
our 1e-4 at 40%, and a large step can skip a shallow basin -- which EHN's own
"smaller electric charges" remark says a low-N_link state should have. TESTED
2026-08-02 and FALSE. Two runs seeded fresh from n=0, identical but for alpha
(N=192, L=153.6, R=0.22L, torus tp=2 tq=3, screened, cramp 8000, wrapped, 12k
steps):

    alpha=1e-4     Lk=-3.0  phi1_knot=[[978, 3]]  E=5076.5   (control)
    alpha=2.05e-4  Lk=-3.0  phi1_knot=[[978, 3]]  E=3794.0

Same knot, same determinant 3, same 978-segment skeleton. The trefoil forms at
82% of the bound as readily as at 40%, so the step size is not what separates us
from EHN. (The energy gap is only descent distance: 2.05x the step buys ~24.6k
steps' worth in 12k, and 3794 lies between the control's 12k value and the
settled 3333. Neither arm is fully relaxed; the claim tested is that the
topology FORMS and holds, not that it has converged.)

The control reproduced the archived trefoil to 0.06% on E and exactly on
topology (978 segs, Lk -3.0, link -25%, el 253 vs 254), so the comparison arm is
trustworthy. What remains as explanation: this seed (EHN's rings IC cannot
express a single phi1 curve winding p times round and q times through, at any
alpha), and the wrapped phase discretisation, which retains linking flux where
the bilinear form drains it. EHN's own conjecture -- that N_link < 4 needs a
LARGER box than theirs -- points the wrong way: this box is 1.67x smaller on a
side than their 320^3.

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

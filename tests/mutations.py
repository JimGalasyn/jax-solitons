"""Named mutations as contracts on the topology suite (run with `mutgate`).

    mutgate run tests/mutations.py            # every contract
    mutgate run tests/mutations.py -v         # show which tests fired
    mutgate list tests/mutations.py

Each entry is a deliberate defect at a named site in the code under test plus
the list of tests that MUST go red under it (`fires`), or the statement that
nothing may (`invisible=True`). `mutgate` applies the change in a sandbox copy
of the tracked files, runs `TESTS`, and judges the contract: a named test that
stays green is a DECORATION, an unnamed test that goes red is OVERREACH, and a
mutation whose `old` string does not occur exactly `count` times is
NOT_APPLIED rather than a silent pass.

Why these sites. `total_helicity` returned half of the true value for as long
as it existed (commit 1f2bf3e): the offending 0.5 carried a comment explaining
itself, so reading did not find it, and no test noticed until an independent
measurement (`linking_invariants.writhe`) disagreed by a factor of two. The
contracts below pin that the guards written afterwards are load-bearing, and
extend the same discipline to the curve invariants the guards rely on.

Conventions. Entries in `fires` / `may_fire` are substrings of pytest node ids.
Every `invisible=True` invariance is paired with a firing mutation at the same
site in this file, so the file itself proves the site is reached by `TESTS`
(an unpaired OK on an invariance is unevidenced). Where a test that names the
defect does NOT fire under it, that is recorded in `note` rather than silently
omitted.
"""
from mutgate import Mutation

# The modules these mutations concern. Each mutation reruns all of them
# (~45 s on one core), so the set is deliberately narrow.
TESTS = (
    "tests/test_total_helicity.py",
    "tests/test_writhe.py",
    "tests/test_topology_linking.py",
)

_VT = "src/jax_solitons/vortex_topology.py"
_LI = "src/jax_solitons/invariants/linking_invariants.py"

# pytest node-id fragments, named once so a rename shows up in one place
ONE_TUBE = "test_helicity_of_one_tube_is_its_writhe"
HALVED_GUARD = "test_the_halved_convention_would_fail_this"
NO_CANCEL = "test_the_regularisation_does_not_cancel_between_configurations"
WR_CONVERGED = "test_writhe_is_converged_in_sampling_and_in_skip"
WR_PYKNOTID = "test_agrees_with_pyknotid_s_writhing_integral"

MUTATIONS = [
    # ------------------------------------------------------------------
    # total_helicity: the factor-2 regression, and the near-pair exclusion
    # ------------------------------------------------------------------
    Mutation(
        "helicity-halve-the-double-sum",
        file=_VT,
        # two lines: the return statement alone also occurs in `_gauss`
        old=("    num = np.where(near, 0.0, num)\n"
             "    return float(dx ** 2 / (4.0 * np.pi) * np.sum(num / d3))"),
        new=("    num = np.where(near, 0.0, num)\n"
             "    return float(dx ** 2 / (4.0 * np.pi) * 0.5 * np.sum(num / d3))"),
        fires=(ONE_TUBE, NO_CANCEL),
        note=("Restores the bug of 1f2bf3e verbatim. The one-tube ratio test "
              "fires at both resolutions and the clasped-pair reading misses "
              "its predicted value. The 'halved convention' guard does NOT "
              "fire: it checks that the band excludes 0.5, and a doubly "
              "halved ratio (~0.26) still lies outside the band. It guards "
              "the guard, not the code."),
    ),
    Mutation(
        "helicity-drop-near-pair-exclusion",
        file=_VT,
        old="    num = np.where(near, 0.0, num)\n",
        new="",
        fires=(ONE_TUBE, NO_CANCEL, HALVED_GUARD),
        note=("Without the exclusion the adjacent-segment 1/r^3 terms of the "
              "staircase skeleton dominate the sum, so the one-tube ratio and "
              "the clasped-pair value leave their bands. The halved guard fires "
              "too, and legitimately: it asserts that HALF the ratio stays "
              "below 0.7, which a divergent reading of the right sign also "
              "breaks. Under this defect every helicity test is red, where "
              "under the historical factor-2 defect one stays green."),
    ),

    # ------------------------------------------------------------------
    # writhe: sign, and the along-curve neighbour exclusion (`skip`)
    # ------------------------------------------------------------------
    Mutation(
        "writhe-flip-sign",
        file=_LI,
        old="    return total / (4.0 * np.pi)\n",
        new="    return -total / (4.0 * np.pi)\n",
        fires=(WR_CONVERGED, WR_PYKNOTID, ONE_TUBE, NO_CANCEL),
        note=("Every invariance test (reflection, rigid motion, "
              "reparametrisation, scale, clasp, planar) is blind to a global "
              "sign by construction; only the pinned value, the independent "
              "implementation, and the helicity cross-checks see it."),
    ),
    Mutation(
        "writhe-drop-neighbour-exclusion",
        file=_LI,
        old="        total += float(np.sum(np.where(near[i], 0.0, contrib)))\n",
        new="        total += float(np.sum(contrib))\n",
        invisible=True,
        note=("MEASURED INVARIANCE, and a correction to the docstring's account "
              "of `skip`. Declared first as a firing mutation against the "
              "converged-value test; nothing fired. Measured on the default "
              "trefoil: with the exclusion deleted the writhe is IDENTICAL to "
              "`skip=0` at six decimals (n = 240, 480, 960), because for "
              "adjacent segments the midpoint separation lies in the span of "
              "the two segment vectors and the triple product vanishes "
              "exactly. There is no adjacent-segment singularity to exclude "
              "under the midpoint rule. What `skip > 0` removes is real "
              "signal (-3.27926 -> -3.27643 from skip 0 to 4 at n = 240), "
              "inside the 5e-3 band the test allows. Paired with "
              "writhe-flip-sign at the return just after this loop, which proves "
              "the body is reached."),
    ),

    # ------------------------------------------------------------------
    # gauss_linking_number: normalisation
    # ------------------------------------------------------------------
    Mutation(
        "gauss-lose-the-factor-two",
        file=_LI,
        old="    return float(total / (4.0 * np.pi))\n",
        new="    return float(total / (2.0 * np.pi))\n",
        fires=(
            "test_hopf_link_is_unit",
            "test_hopf_offdiagonal_unit",
            "test_alpha_cluster_is_complete_K4_total_six",
            "test_alpha_deletion_keeps_links",
            "test_alpha_cluster_obstructed",       # gcd of six 2s is 2, not 1
            NO_CANCEL,                              # asserts Lk(clasp) == -1
        ),
        note=("Every test that pins |Lk| = 1 must fire. The zero-linking "
              "tests (unlink, concentric, Borromean), the symmetry and "
              "antisymmetry tests, and the hand-built Milnor matrices are "
              "blind to a uniform scale by construction."),
    ),

    # ------------------------------------------------------------------
    # linking_matrix: symmetry is filled in, not recomputed
    # ------------------------------------------------------------------
    Mutation(
        "linking-matrix-upper-triangle-only",
        file=_LI,
        old="            L[i, j] = L[j, i] = gauss_linking_number(curves[i], curves[j])\n",
        new="            L[i, j] = gauss_linking_number(curves[i], curves[j])\n",
        fires=(
            "test_shape_symmetry_zero_diagonal",
            "test_alpha_cluster_is_complete_K4_total_six",   # L.sum()/2 halves
        ),
        note=("Every other consumer reads the upper triangle (`triu_indices`, "
              "`L[0, 1]`) and so cannot see the missing lower half. That is "
              "a fact about the tests worth knowing, not a defect in them."),
    ),

    # ------------------------------------------------------------------
    # milnor_indeterminacy: one firing mutation, two paired invariances
    # ------------------------------------------------------------------
    Mutation(
        "milnor-max-instead-of-gcd",
        file=_LI,
        old="        g = gcd(g, int(round(x)))\n",
        new="        g = max(g, int(round(x)))\n",
        fires=("test_common_factor",),
        note=("The known-positive control for the two invariances below: it "
              "proves `TESTS` reach this loop body. Only the hand-built "
              "[2, 2, 4] matrix separates gcd from max; every other fixture "
              "has entries in {0, 1}, where the two agree."),
    ),
    Mutation(
        "milnor-gcd-of-absolute-values",
        file=_LI,
        old="        g = gcd(g, int(round(x)))\n",
        new="        g = gcd(g, abs(int(round(x))))\n",
        invisible=True,
        note=("gcd is sign-agnostic, so the indeterminacy modulus cannot "
              "depend on clasp handedness. Paired with "
              "milnor-max-instead-of-gcd at the same line."),
    ),
    Mutation(
        "milnor-include-the-diagonal",
        file=_LI,
        old="    iu = np.triu_indices(L.shape[0], k=1)\n",
        new="    iu = np.triu_indices(L.shape[0], k=0)\n",
        invisible=True,
        note=("A linking matrix has a zero diagonal and gcd(g, 0) = g, so "
              "including it changes nothing. Paired with "
              "milnor-max-instead-of-gcd in the same function."),
    ),
    # Not declared: `int(round(x))` -> `int(x)` (truncation). The only fixture
    # that reaches `milnor_indeterminacy` with non-integer entries is the
    # measured alpha cluster, whose six linking numbers all sit at 1.00053,
    # where truncation and rounding agree. Unfirable by these tests, so it
    # would read DECORATION for a reason that is not the tests' fault. A
    # negative near-integer link (the clasped trefoils give -0.999) would fire
    # it, once something in `TESTS` passes one to this function.
]

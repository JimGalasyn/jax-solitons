"""The manifest must identify which IC produced a state — and must not guess.

Before these, `--geom rings --nlink 3` and `--geom torus --tp 2 --tq 3` wrote
byte-identical params (the torus branch sets nlink = tq, so both recorded
nlink=3). Nothing in the suite exercised it, so nothing stopped the two from
collapsing back together.

The resume cases are the ones that matter most. A resumed run builds no IC, so
this invocation's --geom/--tp/--tq/--R describe nothing; reading them anyway
makes the manifest confidently wrong rather than merely ambiguous. That is the
dominant path in practice — soliton-playground's `_relax_cmd` drops geom_args
whenever it passes --resume.
"""
import json

import pytest

from jax_solitons.ehn.relax import _seed_from_resumed, _seed_params


def test_rings_and_torus_are_distinguishable():
    """The property the whole change exists for."""
    rings = _seed_params("rings", 2, 3, 14.0, 6.3, 0, nlink=3)
    torus = _seed_params("torus", 2, 3, 14.0, 6.3, 0, nlink=3)
    assert rings != torus, "rings and torus wrote identical params again"
    assert rings["geom"] == "rings" and torus["geom"] == "torus"
    # nlink alone cannot tell them apart — that is the original bug.
    assert rings["nlink"] == torus["nlink"] == 3


def test_torus_records_the_winding_and_rings_does_not():
    torus = _seed_params("torus", 2, 5, 33.792, 15.2064, -1, nlink=5)
    assert (torus["tp"], torus["tq"], torus["twist"]) == (2, 5, -1)
    assert torus["R"] == pytest.approx(33.792)
    rings = _seed_params("rings", 2, 5, 33.792, 15.2064, 0, nlink=4)
    for k in ("tp", "tq", "twist", "rminor"):
        assert k not in rings, f"{k} is meaningless for a rings IC"


def test_rminor_is_the_value_used_not_a_rounded_default():
    """Caller resolves 0.45*R once; this records that number exactly.

    A second copy of the default here would drift from the one the IC used, and
    rounding it defeats the point of a field whose job is to be exact.
    """
    rr = 0.45 * 33.792
    torus = _seed_params("torus", 2, 3, 33.792, rr, 0, nlink=3)
    assert torus["rminor"] == rr


def test_resume_carries_the_seed_forward_and_ignores_argv(tmp_path):
    """The regression this guards: argv says rings/14.0, the state says torus."""
    (tmp_path / "manifest.json").write_text(json.dumps({"params": {
        "geom": "torus", "tp": 2, "tq": 3, "twist": 0,
        "R": 33.792, "rminor": 15.2064, "nlink": 3, "alpha": 1e-4}}))

    seed = _seed_from_resumed(tmp_path / "field.npz")

    assert seed["geom"] == "torus", "read argv instead of the resumed state"
    assert seed["R"] == pytest.approx(33.792), "would have filed the 14.0 default"
    assert seed["nlink"] == 3, "floor would be computed from nlink=4"
    assert (seed["tp"], seed["tq"]) == (2, 3)
    assert seed["seed_provenance"] == "carried-forward"
    assert seed["resumed_from"].endswith("field.npz")
    # Params that are NOT seed geometry belong to this run, not the prior one.
    assert "alpha" not in seed


@pytest.mark.parametrize("prior", [None, "", "{not json", '{"params": {}}'])
def test_resume_without_usable_provenance_records_nothing_not_a_guess(tmp_path, prior):
    """Recording nothing is recoverable; recording the wrong thing is not."""
    if prior is not None:
        (tmp_path / "manifest.json").write_text(prior)

    seed = _seed_from_resumed(tmp_path / "field.npz")

    assert seed["geom"] is None
    assert seed["seed_provenance"] == "unavailable"
    assert "R" not in seed, "invented a radius from the CLI default"
    assert seed["resumed_from"].endswith("field.npz")


# --- end-to-end -----------------------------------------------------------
# The unit tests above pin the helpers; these pin the CALL SITES in run(),
# which is where the regression actually lived — _seed_params was correct and
# was simply being handed argv on a path that builds no IC. N=8 runs in well
# under a second on CPU, so this is cheap enough to keep in the default suite.

def test_run_carries_seed_across_a_real_resume(tmp_path):
    """The exact shape standard_box uses: resume with NO seed flags at all."""
    from jax_solitons.ehn.relax import run
    a, b = tmp_path / "a", tmp_path / "b"
    run(N=8, L=6.4, R=1.5, geom="torus", tp=2, tq=3,
        steps=2, samples=2, save_every=2, out=str(a))
    seeded = json.loads((a / "manifest.json").read_text())["params"]
    assert (seeded["geom"], seeded["tp"], seeded["tq"], seeded["nlink"]) \
        == ("torus", 2, 3, 3)

    # No --geom/--tp/--tq/--R, exactly as _relax_cmd invokes a resume.
    run(N=8, L=6.4, steps=4, samples=2,
        resume=str(a / "field.npz"), out=str(b))
    resumed = json.loads((b / "manifest.json").read_text())["params"]

    assert resumed["geom"] == "torus", "fell back to the rings default"
    assert resumed["R"] == pytest.approx(1.5), "filed the R=14.0 default"
    assert resumed["nlink"] == 3, "floor would come from nlink=4"
    assert resumed["seed_provenance"] == "carried-forward"


def test_run_records_the_functional_parameters(tmp_path):
    """Seed geometry alone does not reproduce a state; these must be there too."""
    from jax_solitons.ehn.relax import run
    out = tmp_path / "a"
    run(N=8, L=6.4, R=1.5, geom="torus", tp=2, tq=3,
        steps=2, samples=2, out=str(out))
    params = json.loads((out / "manifest.json").read_text())["params"]
    for k in ("lam", "kappa", "eps_a", "q1", "q2", "c4", "core", "n_ic",
              "N", "L", "C", "alpha", "beta", "U", "ic", "cramp", "agrad"):
        assert k in params, f"{k} missing — manifest cannot reproduce its state"


# --- the φ₁ self-knot determinant, in the manifest --------------------------
# These pin the CALL SITE, not the measurement: knot_determinants has its own
# tests, and the defect worth guarding here is a flag that never reaches the
# manifest. That is not hypothetical — --topo-every shipped defaulting to 0 with
# nothing exercising the wiring, and a 2026-08-03 rental was configured without
# it and would have produced no topology series at all. A diagnostic nobody
# asserts on is a diagnostic that can quietly not happen.

def test_det_every_records_a_determinant_series(tmp_path):
    """--det-every K must put a det1 on every sampled traj entry."""
    from jax_solitons.ehn.relax import run
    out = tmp_path / "d"
    run(N=8, L=6.4, R=1.5, geom="torus", tp=2, tq=3,
        steps=2, samples=2, det_every=1, out=str(out))
    traj = json.loads((out / "manifest.json").read_text())["traj"]
    assert traj, "no samples recorded at all"
    missing = [e["n"] for e in traj if "det1" not in e]
    assert not missing, f"samples without det1: {missing}"


def test_end_state_determinant_is_recorded_even_with_the_series_off(tmp_path):
    """det_every gates the SERIES only; the end-state number is unconditional.

    It is one call on a state already in host memory, and it is the number a
    torus-knot run is judged by — so it must not depend on remembering a flag.
    """
    from jax_solitons.ehn.relax import run
    out = tmp_path / "d"
    run(N=8, L=6.4, R=1.5, geom="torus", tp=2, tq=3,
        steps=2, samples=2, det_every=0, out=str(out))
    m = json.loads((out / "manifest.json").read_text())
    assert "det1" in m, "end-state determinant missing from the manifest"
    assert not any("det1" in e for e in m["traj"]), \
        "det_every=0 must leave the per-sample series off"

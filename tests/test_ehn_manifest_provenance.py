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

"""Shared remote-core tests: run_fn-by-ref loading and the run_one worker unit
(execute_config against a file registry), exercised without a GPU via a trivial
injected RunFn."""

import numpy as np
import pytest

import jax_solitons.campaign.remote as remote
from jax_solitons.campaign.remote import load_run_fn, run_one
from jax_solitons.runs import RunConfig

CFG = RunConfig(model="faddeev_cp1", N=8, L=8.0, params={"R": 2.0})


def test_load_run_fn_resolves_real_ref():
    fn = load_run_fn("jax_solitons.runfns:faddeev_relax_then_id")
    assert callable(fn) and fn.__name__ == "faddeev_relax_then_id"


def test_load_run_fn_rejects_bad_ref():
    with pytest.raises(ValueError, match="module:function"):
        load_run_fn("not_a_ref")


def test_load_run_fn_rejects_non_callable():
    with pytest.raises(TypeError):
        load_run_fn("jax_solitons.campaign.worker:RESULT_PREFIX")  # a str constant


def _trivial_runfn(config, ctx):
    """A stand-in RunFn: stream one event + a full-state checkpoint, then finish."""
    ctx.emit({"step": 0, "R": config.params["R"]})
    ctx.checkpoint({"z": np.zeros(3)}, 0)
    return {"ran": True, "R": config.params["R"]}


def test_run_one_executes_and_is_idempotent(tmp_path, monkeypatch):
    monkeypatch.setattr(remote, "load_run_fn", lambda ref: _trivial_runfn)
    out = run_one(CFG.to_json(), "x:y", str(tmp_path))
    assert out["run"] == CFG.run_name()
    assert out["result"] == {"ran": True, "R": 2.0} and out["skipped"] is False
    # A/B/C artifacts landed under the config-hashed run dir.
    run = tmp_path / CFG.run_name()
    assert (run / "DONE.json").exists()
    assert (run / "checkpoint.npz").exists()
    assert (run / "events.jsonl").exists()
    # Second call is the idempotent skip (already complete).
    again = run_one(CFG.to_json(), "x:y", str(tmp_path))
    assert again["skipped"] is True and again["result"] is None


def test_run_one_enables_x64_for_float64(tmp_path, monkeypatch):
    """A float64 config makes the worker enable jax x64 (else it'd silently run
    in float32 on a fresh remote process)."""
    import jax
    monkeypatch.setattr(remote, "load_run_fn", lambda ref: _trivial_runfn)
    cfg = RunConfig("faddeev_cp1", N=8, L=8.0, dtype="float64", params={"R": 2.0})
    run_one(cfg.to_json(), "x:y", str(tmp_path))
    assert jax.config.read("jax_enable_x64") is True

"""The campaign contract, exercised end-to-end on a real physics RunFn.

Proves the boundary closes: the Faddeev relax-then-ID pipeline (runfns.py),
driven through run_campaign over the local reference backends, produces the
registered/streamed/triggered artifacts the contract promises (A/B/C/E), and
the optimizer-state resume is bit-identical (B, P4).
"""

import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from jax_solitons.campaign import (  # noqa: E402
    AdmissionError,
    FileRunRegistry,
    JsonlEventSink,
    LocalExecutor,
    ProbeAdmission,
    run_campaign,
)
from jax_solitons.campaign.protocols import RunContext  # noqa: E402
from jax_solitons.campaign.reference import HostReport  # noqa: E402
from jax_solitons.models.faddeev import faddeev_cp1_model  # noqa: E402
from jax_solitons.runfns import faddeev_relax_then_id  # noqa: E402
from jax_solitons.runs import RunConfig, load_checkpoint  # noqa: E402
from jax_solitons.seeds import rational_map_hopfion_cp1  # noqa: E402
from jax_solitons.steppers.adam import adam_flow  # noqa: E402


def test_campaign_drives_full_pipeline(tmp_path):
    """A/B/C: run_campaign registers, streams a ledger, checkpoints, finishes,
    and captures the relaxed core as a triggered full-state event."""
    registry = FileRunRegistry(tmp_path)
    sink = JsonlEventSink()
    admission = ProbeAdmission(require_gpu=False)  # CPU CI: skip the GPU gate
    cfg = RunConfig(model="faddeev_cp1", N=20, L=14.0, dtype="float64",
                    steps=120, params={"R": 3.0, "n": 1, "m": 1, "segments": 3})

    run_campaign([cfg], faddeev_relax_then_id, registry=registry, sink=sink,
                 admission=admission, executor=LocalExecutor())

    run = tmp_path / cfg.run_name()
    # A: config-hashed dir + a manifest line
    assert (run / "config.json").exists()
    assert len((tmp_path / "MANIFEST.jsonl").read_text().splitlines()) == 1
    # B: a full-state checkpoint was written
    assert (run / "checkpoint.npz").exists()
    # C: streamed ledger (seed row + one per segment)
    rows = [json.loads(line)
            for line in (run / "events.jsonl").read_text().splitlines()]
    assert len(rows) == 4
    assert rows[-1]["step"] == 120
    # finish: result record with the core census
    done = json.loads((run / "DONE.json").read_text())
    assert abs(done["Q_H"] - 1.0) < 0.1          # a unit hopfion, lightly relaxed
    assert done["n_curves"] == 1                 # the core ring is found
    assert done["core_length"] > 2 * np.pi       # ~ 2*pi*R for R=3
    # C trigger: exactly one full-state capture for the one identified core
    assert len(list((run / "triggered").glob("*.npz"))) == 1


def test_campaign_idempotent_skip(tmp_path):
    """D recovery: a completed run is skipped on re-submission (no new work,
    no duplicate manifest line)."""
    registry = FileRunRegistry(tmp_path)
    sink = JsonlEventSink()
    admission = ProbeAdmission(require_gpu=False)
    cfg = RunConfig(model="faddeev_cp1", N=20, L=14.0, dtype="float64",
                    steps=60, params={"R": 3.0, "segments": 2})

    run_campaign([cfg], faddeev_relax_then_id, registry=registry, sink=sink,
                 admission=admission, executor=LocalExecutor())
    handle = registry.register(cfg)
    assert registry.is_complete(handle)
    events = tmp_path / handle.name / "events.jsonl"
    rows_before = len(events.read_text().splitlines())

    # Re-run: skipped, so neither the ledger nor the manifest grows.
    run_campaign([cfg], faddeev_relax_then_id, registry=registry, sink=sink,
                 admission=admission, executor=LocalExecutor())
    rows_after = len(events.read_text().splitlines())
    assert rows_after == rows_before
    assert len((tmp_path / "MANIFEST.jsonl").read_text().splitlines()) == 1


def test_campaign_resume_bit_identical(tmp_path):
    """B (headline): a run preempted MID-segments and resumed from its last
    full-state checkpoint reaches a bit-identical final state to the
    uninterrupted run.

    This exercises the `ctx.resume` branch in `faddeev_relax_then_id` end to
    end (its building blocks -- idempotent skip and standalone Adam resume --
    are covered above, but the wired restart path was not). A `Preempt` raised
    from the checkpoint callback after 2 of 4 segments stands in for a spot
    kill: the checkpoint is on disk, no DONE marker, so the next `run_campaign`
    must load it and finish the remaining segments.
    """
    cfg = RunConfig(model="faddeev_cp1", N=16, L=12.0, dtype="float64",
                    steps=120, params={"R": 2.6, "n": 1, "m": 1, "segments": 4})
    admission = ProbeAdmission(require_gpu=False)

    # --- uninterrupted reference (never round-trips a checkpoint mid-run) ---
    ref_reg = FileRunRegistry(tmp_path / "ref")
    run_campaign([cfg], faddeev_relax_then_id, registry=ref_reg,
                 sink=JsonlEventSink(), admission=admission,
                 executor=LocalExecutor())
    ref_handle = ref_reg.register(cfg)
    ref_state, _, ref_step = load_checkpoint(ref_handle.dir / "checkpoint.npz")

    # --- preempted run: abort after the 2nd-segment checkpoint lands ---
    reg = FileRunRegistry(tmp_path / "pre")
    sink = JsonlEventSink()
    handle = reg.register(cfg)

    class Preempt(Exception):
        pass

    saved = {"n": 0}

    def checkpoint(state, step):
        reg.save(handle, state, step)        # full state hits disk first...
        saved["n"] += 1
        if saved["n"] == 2:                  # ...then the host "dies"
            raise Preempt

    ctx = RunContext(resume=None, checkpoint=checkpoint,
                     emit=lambda record: sink.emit(handle, record),
                     trigger=lambda state, reason: sink.trigger(handle, state, reason))
    with pytest.raises(Preempt):
        faddeev_relax_then_id(cfg, ctx)
    assert (handle.dir / "checkpoint.npz").exists()
    assert not reg.is_complete(handle)       # no DONE -> the resume is required
    _, _, pre_step = load_checkpoint(handle.dir / "checkpoint.npz")
    assert pre_step == 60                     # 2 of 4 segments (120 // 4 * 2)

    # --- resume: run_campaign loads the checkpoint and finishes the rest ---
    run_campaign([cfg], faddeev_relax_then_id, registry=reg, sink=sink,
                 admission=admission, executor=LocalExecutor())
    assert reg.is_complete(handle)

    # Bit-identical final state, and the event stream continued without
    # re-seeding or replaying finished segments (steps strictly increasing).
    res_state, _, res_step = load_checkpoint(handle.dir / "checkpoint.npz")
    assert res_step == ref_step == 120
    assert np.array_equal(np.asarray(res_state["z"]), np.asarray(ref_state["z"]))
    steps = [json.loads(line)["step"] for line in
             (handle.dir / "events.jsonl").read_text().splitlines()]
    assert steps == [0, 30, 60, 90, 120]


def test_adam_resume_bit_identical():
    """B foundation: a segmented Adam run with carried optimizer state is
    bit-identical to the uninterrupted run (what makes spot preemption free)."""
    grid_kw = dict(N=16, L=12.0, dtype=jnp.float64)
    from jax_solitons.grid import BoxGrid
    grid = BoxGrid(**grid_kw)
    model = faddeev_cp1_model(c4=4.0)
    z0 = rational_map_hopfion_cp1(grid, R=2.6, n=1, m=1)

    z_full, _ = adam_flow(model, z0, grid, lr=2e-3, steps=80)
    z_a, _, opt = adam_flow(model, z0, grid, lr=2e-3, steps=40,
                            return_opt_state=True)
    z_b, _, _ = adam_flow(model, z_a, grid, lr=2e-3, steps=40, opt_state=opt,
                          return_opt_state=True)
    assert np.array_equal(np.asarray(z_full), np.asarray(z_b))


def test_admission_probe_or_bail():
    """E (P9): admission rejects a host that cannot ship results, and passes a
    healthy one — independent of the CI host's real hardware."""
    adm = ProbeAdmission(min_mem_gb=4.0, min_mbps=1.0)

    bad = HostReport(has_gpu=True, device_name="x", free_mem_gb=8.0,
                     outbound_mbps=0.0)            # zero outbound — the case study
    good = HostReport(has_gpu=True, device_name="x", free_mem_gb=8.0,
                      outbound_mbps=100.0)
    adm.probe = lambda: bad
    with pytest.raises(AdmissionError):
        adm.guard()
    adm.probe = lambda: good
    assert adm.guard().outbound_mbps == 100.0


def test_campaign_step_count_exact(tmp_path):
    """#4 provenance: when steps isn't divisible by segments, the executed
    steps still sum to exactly config.steps (the remainder is distributed, not
    dropped), and the ledger labels match."""
    registry = FileRunRegistry(tmp_path)
    sink = JsonlEventSink()
    admission = ProbeAdmission(require_gpu=False)
    cfg = RunConfig(model="faddeev_cp1", N=12, L=10.0, dtype="float64",
                    steps=100, params={"R": 2.2, "segments": 3})  # 100/3 -> 34,33,33

    run_campaign([cfg], faddeev_relax_then_id, registry=registry, sink=sink,
                 admission=admission, executor=LocalExecutor())

    run = tmp_path / cfg.run_name()
    steps = [json.loads(line)["step"]
             for line in (run / "events.jsonl").read_text().splitlines()]
    assert steps == [0, 34, 67, 100]                       # no dropped remainder
    assert json.loads((run / "DONE.json").read_text())["step"] == 100


def test_adam_observer_global_step():
    """adam.py: across a resume (opt_state carried), the observer sees the
    GLOBAL step, not a per-segment reset — matching the bias-correction/LR
    counter. Without the fix the second segment would relabel from 0."""
    from jax_solitons.grid import BoxGrid
    grid = BoxGrid(N=12, L=10.0, dtype=jnp.float64)
    model = faddeev_cp1_model(c4=4.0)
    z0 = rational_map_hopfion_cp1(grid, R=2.2, n=1, m=1)

    seen = []
    obs = lambda step, state: seen.append(int(step))
    z1, _, opt = adam_flow(model, z0, grid, lr=2e-3, steps=4,
                           observe_every=2, observer=obs, return_opt_state=True)
    adam_flow(model, z1, grid, lr=2e-3, steps=4, observe_every=2, observer=obs,
              opt_state=opt, return_opt_state=True)
    assert seen == [0, 2, 4, 4, 6, 8]        # monotone & global across the resume


def test_admission_rejects_failed_probe():
    """E (P9): a host whose probe FAILED (probe_ok=False) is a hard reject even
    with require_gpu=False — never 'runs anyway' on an unprobed host. Without
    the fix, require_gpu=False would bypass the GPU/mem gates and admit it."""
    adm = ProbeAdmission(require_gpu=False, min_mem_gb=0.0, min_mbps=0.0)
    adm.probe = lambda: HostReport(
        has_gpu=False, device_name="probe-failed: boom", free_mem_gb=0.0,
        outbound_mbps=float("inf"), probe_ok=False)
    with pytest.raises(AdmissionError):
        adm.guard()


def test_admission_device_probe_paths(monkeypatch):
    """E: _device reads free memory from a GPU's memory_stats, and treats an
    UNREADABLE capacity as UNKNOWN (+inf), not 0 — so a healthy GPU is never
    falsely rejected. Both 'unknown' routes are covered: a present-but-empty
    memory_stats (missing keys) and a device with no memory_stats attribute at
    all (the getattr fallback)."""
    import jax

    class FakeDev:                       # has memory_stats(); returns given dict
        platform = "gpu"
        device_kind = "FakeGPU"
        def __init__(self, stats): self._stats = stats
        def memory_stats(self): return self._stats

    class FakeDevNoStats:                 # no memory_stats attribute at all
        platform = "gpu"
        device_kind = "FakeGPU-nostats"

    # GPU reporting stats: 10 - 2 = 8 GB free.
    monkeypatch.setattr(jax, "devices", lambda: [FakeDev(
        {"bytes_limit": 10_000_000_000, "bytes_in_use": 2_000_000_000})])
    r = ProbeAdmission(min_mem_gb=4.0).probe()
    assert r.has_gpu and r.probe_ok and abs(r.free_mem_gb - 8.0) < 0.1
    ProbeAdmission(min_mem_gb=4.0).guard()            # admits: 8 GB >= 4

    # memory_stats present but EMPTY (no bytes_limit key): UNKNOWN -> +inf.
    monkeypatch.setattr(jax, "devices", lambda: [FakeDev({})])
    r2 = ProbeAdmission(min_mem_gb=4.0).probe()
    assert r2.has_gpu and r2.free_mem_gb == float("inf")
    ProbeAdmission(min_mem_gb=4.0).guard()            # admits: unknown != blocked

    # NO memory_stats attribute: getattr fallback -> {} -> UNKNOWN -> +inf.
    monkeypatch.setattr(jax, "devices", lambda: [FakeDevNoStats()])
    r3 = ProbeAdmission(min_mem_gb=4.0).probe()
    assert r3.has_gpu and r3.free_mem_gb == float("inf")
    ProbeAdmission(min_mem_gb=4.0).guard()


def test_admission_probe_exception_is_hard_reject(monkeypatch):
    """E (P9): if the device query itself throws, _device reports probe_ok=False
    and guard() hard-rejects — even with require_gpu=False."""
    import jax
    def boom():
        raise RuntimeError("no driver")
    monkeypatch.setattr(jax, "devices", boom)
    r = ProbeAdmission(require_gpu=False).probe()
    assert r.probe_ok is False and r.device_name.startswith("probe-failed")
    with pytest.raises(AdmissionError):
        ProbeAdmission(require_gpu=False).guard()

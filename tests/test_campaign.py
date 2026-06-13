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
from jax_solitons.campaign.reference import HostReport  # noqa: E402
from jax_solitons.models.faddeev import faddeev_cp1_model  # noqa: E402
from jax_solitons.runfns import faddeev_relax_then_id  # noqa: E402
from jax_solitons.runs import RunConfig  # noqa: E402
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
    assert sum(1 for _ in (tmp_path / "MANIFEST.jsonl").open()) == 1
    # B: a full-state checkpoint was written
    assert (run / "checkpoint.npz").exists()
    # C: streamed ledger (seed row + one per segment)
    rows = [json.loads(line) for line in (run / "events.jsonl").open()]
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
    rows_before = sum(1 for _ in (tmp_path / handle.name / "events.jsonl").open())

    # Re-run: skipped, so neither the ledger nor the manifest grows.
    run_campaign([cfg], faddeev_relax_then_id, registry=registry, sink=sink,
                 admission=admission, executor=LocalExecutor())
    rows_after = sum(1 for _ in (tmp_path / handle.name / "events.jsonl").open())
    assert rows_after == rows_before
    assert sum(1 for _ in (tmp_path / "MANIFEST.jsonl").open()) == 1


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

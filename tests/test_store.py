"""Shared object-store backend: BlobStore + ObjectStoreRunRegistry/EventSink.

Backend mechanics run fast over a MemoryBlobStore; one end-to-end test drives
the real relax-then-ID pipeline through run_campaign over the object store to
prove it is a drop-in A/B/C backend (and that the streamed ledger reads back).
"""

import json

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from jax_solitons.campaign import run_campaign  # noqa: E402
from jax_solitons.campaign.protocols import RunContext  # noqa: E402
from jax_solitons.campaign.reference import ProbeAdmission, LocalExecutor  # noqa: E402
from jax_solitons.campaign.store import (  # noqa: E402
    BlobStore,
    MemoryBlobStore,
    ObjectStoreEventSink,
    ObjectStoreRunRegistry,
)
from jax_solitons.runfns import faddeev_relax_then_id  # noqa: E402
from jax_solitons.runs import RunConfig  # noqa: E402

CFG = RunConfig(model="faddeev_cp1", N=10, L=10.0, params={"R": 2.0})


def test_memory_blobstore_is_a_blobstore():
    s = MemoryBlobStore()
    assert isinstance(s, BlobStore)
    assert s.get("x") is None and not s.exists("x")
    s.put("a/b", b"hi")
    assert s.get("a/b") == b"hi" and s.exists("a/b")
    s.put("a/c", b"yo")
    assert sorted(s.list("a/")) == ["a/b", "a/c"] and s.list("z/") == []


def test_registry_roundtrip_and_manifest():
    reg = ObjectStoreRunRegistry(MemoryBlobStore())
    h = reg.register(CFG)
    assert not reg.is_complete(h)
    assert reg.load(h) is None
    state = {"z": np.arange(6, dtype=np.float64).reshape(2, 3)}
    reg.save(h, state, step=7)
    got, step = reg.load(h)
    assert step == 7 and np.array_equal(np.asarray(got["z"]), state["z"])  # B fidelity
    reg.finish(h, {"E": 1.5})
    assert reg.is_complete(h)
    assert reg.manifest() == [{"run": h.name, "config": json.loads(CFG.to_json())}]


def test_register_is_idempotent():
    store = MemoryBlobStore()
    reg = ObjectStoreRunRegistry(store)
    reg.register(CFG)
    reg.register(CFG)                                  # second time: no dup manifest
    assert len(store.list("runs/_manifest/")) == 1


def test_sink_streams_and_reads_back():
    store = MemoryBlobStore()
    sink = ObjectStoreEventSink(store)
    h = ObjectStoreRunRegistry(store).register(CFG)
    for i in range(3):
        sink.emit(h, {"step": i * 10, "E": 100 - i})
    sink.trigger(h, {"z": np.zeros(4)}, reason="quench")
    rows = sink.events(h.name)
    assert [r["step"] for r in rows] == [0, 10, 20]    # streamed, in order
    assert len(store.list(f"runs/{h.name}/triggered/")) == 1
    # A FRESH sink over the SAME store continues the sequence (resume-safe).
    sink2 = ObjectStoreEventSink(store)
    sink2.emit(h, {"step": 30})
    assert [r["step"] for r in sink2.events(h.name)] == [0, 10, 20, 30]


def test_shared_store_seen_by_a_second_registry():
    """The unlock: a registry instance in 'another process' (a fresh object over
    the same store) sees completion -> global is_complete / cross-provider dedup."""
    store = MemoryBlobStore()
    a = ObjectStoreRunRegistry(store)
    h = a.register(CFG)
    a.finish(h, {"ok": True})
    b = ObjectStoreRunRegistry(store)                  # a different worker/provider
    assert b.is_complete(b.register(CFG))              # sees A's DONE


def test_read_helpers_skip_vanished_blobs():
    """list()->get() isn't atomic; a key whose blob is gone (eventual
    consistency / a concurrent prune) is skipped, not a TypeError."""
    class Flaky(MemoryBlobStore):
        def list(self, prefix):                      # advertise a ghost key
            return super().list(prefix) + [f"{prefix}ghost.json"]

    store = Flaky()
    reg = ObjectStoreRunRegistry(store)
    sink = ObjectStoreEventSink(store)
    h = reg.register(CFG)
    sink.emit(h, {"step": 0})
    assert sink.events(h.name) == [{"step": 0}]      # ghost event skipped, no crash
    assert reg.manifest() == [
        {"run": h.name, "config": json.loads(CFG.to_json())}]  # ghost manifest skipped


def test_run_campaign_over_object_store():
    """Drop-in: drive the real pipeline through run_campaign over the object
    store; the registered/streamed/finished artifacts all land in the store."""
    store = MemoryBlobStore()
    reg = ObjectStoreRunRegistry(store)
    sink = ObjectStoreEventSink(store)
    cfg = RunConfig(model="faddeev_cp1", N=20, L=14.0, dtype="float64",
                    steps=120, params={"R": 3.0, "n": 1, "m": 1, "segments": 3})
    run_campaign([cfg], faddeev_relax_then_id, registry=reg, sink=sink,
                 admission=ProbeAdmission(require_gpu=False),
                 executor=LocalExecutor())
    h = reg.register(cfg)
    assert reg.is_complete(h)
    done = json.loads(store.get(f"runs/{h.name}/DONE.json"))
    assert abs(done["Q_H"] - 1.0) < 0.1 and done["n_curves"] == 1
    steps = [r["step"] for r in sink.events(h.name)]
    assert steps == [0, 40, 80, 120]                   # streamed ledger in the store
    assert store.exists(f"runs/{h.name}/checkpoint.npz")
    assert len(store.list(f"runs/{h.name}/triggered/")) == 1   # the core capture

    # Re-running is the idempotent skip across the shared store (no new events).
    run_campaign([cfg], faddeev_relax_then_id, registry=reg, sink=sink,
                 admission=ProbeAdmission(require_gpu=False),
                 executor=LocalExecutor())
    assert [r["step"] for r in sink.events(h.name)] == [0, 40, 80, 120]

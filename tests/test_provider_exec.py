"""ProviderExecutor tests: a FakeProvider (no spend) + monkeypatched SSH cover
the campaign-over-rented-fleet path -- happy run, per-host failover, and the
leak-proof teardown that must fire on every exit.
"""

import contextlib
import json

import pytest

import jax_solitons.campaign.provider_exec as pe
from jax_solitons.campaign import (HostProbeFailed, HostSpec, LaunchSpec, Offer,
                                    RentedHost)
from jax_solitons.campaign.provider_exec import ProviderExecutor
from jax_solitons.campaign.worker import RESULT_PREFIX
from jax_solitons.runs import RunConfig

LAUNCH = LaunchSpec(image="img:12.2", onstart="echo hi", disk_gb=24)
SPEC = HostSpec(gpu_name="RTX 3090", max_dph=0.30)
CONFIGS = [RunConfig(model="faddeev_cp1", N=16, L=12.0, params={"R": 2.6}),
           RunConfig(model="faddeev_cp1", N=16, L=12.0, params={"R": 3.0})]


def _offer(oid, dph=0.12):
    return Offer(id=oid, dph=dph, gpu_name="RTX 3090", num_gpus=1,
                 reliability=0.99, inet_down_mbps=800, cuda_max=12.4,
                 geolocation="x", provider="fake")


class FakeProvider:
    """offers() + leak-proof rent() with a `live` set; bad_ids fail to come up."""

    name = "fake"

    def __init__(self, offers, bad_ids=()):
        self._offers = list(offers)
        self._bad = set(bad_ids)
        self.live: set[str] = set()
        self.rented: list[str] = []
        self._n = 0

    def offers(self, spec):
        return list(self._offers)

    @contextlib.contextmanager
    def rent(self, offer, launch, *, timeout_s=600):
        iid = f"inst-{self._n}"; self._n += 1
        self.rented.append(offer.id); self.live.add(iid)
        try:
            if offer.id in self._bad:
                raise HostProbeFailed(f"bad host {offer.id}")
            yield RentedHost(id=iid, ssh_host="10.0.0.1", ssh_port=22,
                             offer=offer)
        finally:
            self.live.discard(iid)            # leak-proof teardown


def _fake_ssh_factory(*, ready=True, run_rc=0):
    """Build a fake _ssh: the 'import jax_solitons' readiness probe and the
    worker invocation both routed by inspecting the command string."""
    def fake_ssh(key, host, port, cmd, timeout=120):
        if "import jax_solitons" in cmd:
            return (0, "") if ready else (1, "ModuleNotFoundError")
        if "campaign.worker" in cmd:
            rec = {"run": "r", "result": {"ok": True}, "skipped": False}
            return (run_rc, RESULT_PREFIX + json.dumps(rec) + "\n")
        return (0, "")
    return fake_ssh


@pytest.fixture
def patched(monkeypatch):
    monkeypatch.setattr(pe.time, "sleep", lambda s: None)
    monkeypatch.setattr(pe, "_scp_down", lambda *a, **k: (0, ""))

    def apply(**kw):
        monkeypatch.setattr(pe, "_ssh", _fake_ssh_factory(**kw))
    return apply


def test_runs_all_configs_and_tears_down(patched):
    patched(ready=True)
    prov = FakeProvider([_offer("a")])
    ex = ProviderExecutor(prov, "pkg.mod:fn", LAUNCH, host_spec=SPEC,
                          ready_timeout=1)
    results = ex.run(CONFIGS)
    assert len(results) == 2 and all(r["result"] == {"ok": True} for r in results)
    assert prov.live == set()                 # host torn down
    assert prov.rented == ["a"]


def test_fails_over_bad_host(patched):
    patched(ready=True)
    prov = FakeProvider([_offer("bad"), _offer("good")], bad_ids={"bad"})
    ex = ProviderExecutor(prov, "pkg.mod:fn", LAUNCH, host_spec=SPEC,
                          ready_timeout=1)
    results = ex.run(CONFIGS)
    assert len(results) == 2                   # ran on the good host
    assert prov.rented == ["bad", "good"]      # failed over past the bad one
    assert prov.live == set()                  # both attempts torn down


def test_engine_never_ready_fails_over_then_raises(patched):
    patched(ready=False)                       # import check always fails
    prov = FakeProvider([_offer("a"), _offer("b")])
    ex = ProviderExecutor(prov, "pkg.mod:fn", LAUNCH, host_spec=SPEC,
                          ready_timeout=0.05)
    with pytest.raises(RuntimeError, match="all .* offers failed"):
        ex.run(CONFIGS)
    assert prov.rented == ["a", "b"]           # tried both, both timed out ready
    assert prov.live == set()                  # neither leaked


def test_no_offers_raises(patched):
    patched(ready=True)
    ex = ProviderExecutor(FakeProvider([]), "pkg.mod:fn", LAUNCH, host_spec=SPEC)
    with pytest.raises(RuntimeError, match="no offers"):
        ex.run(CONFIGS)


def test_empty_configs_is_noop(patched):
    patched(ready=True)
    prov = FakeProvider([_offer("a")])
    assert ProviderExecutor(prov, "pkg.mod:fn", LAUNCH).run([]) == []
    assert prov.rented == []                    # never even rented

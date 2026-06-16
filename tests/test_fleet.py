"""FleetExecutor tests: a FakeProvider (no spend) + monkeypatched ssh/scp cover
the parallel one-host-per-leg script fleet -- happy run, per-leg failover on a
bad host / offer race, fast-fail via API status (#27), offer-pool refresh (#28),
resume/skip (#26), the NO_OFFERS / RUN_FAIL / NO_RESULT outcomes, and the
signal-safe teardown backstop (#24).
"""

import contextlib
import signal
import threading
import types

import pytest

import jax_solitons.campaign.fleet as fleet
from jax_solitons.campaign import (FleetExecutor, FleetLeg, HostProbeFailed,
                                   HostSpec, LaunchSpec, Offer, RentedHost,
                                   RentUnavailable, SentinelReady, fleet_status)

LAUNCH = LaunchSpec(image="img:12.2", onstart="echo hi", disk_gb=24, label="farm-x")
SPEC = HostSpec(gpu_name="RTX 3090", max_dph=0.30)


def _offer(oid, dph=0.12):
    return Offer(id=oid, dph=dph, gpu_name="RTX 3090", num_gpus=1, reliability=0.99,
                 inet_down_mbps=800, cuda_max=12.4, geolocation="x", provider="fake")


def _leg(label, *, fetch="", done_when="", command="run.sh"):
    return FleetLeg(label=label, command=command, ship=("driver.py",),
                    fetch=fetch, done_when=done_when)


class FakeProvider:
    """offers() (with one refill) + leak-proof rent(); bad_ids fail to come up,
    race_ids are taken (RentUnavailable), dead_ids never get ready but report
    dead via the API (the #27 fast-fail path). destroy/list_instances/dead_reason
    round out the surface FleetExecutor + status use."""

    name = "fake"

    def __init__(self, offers, *, bad_ids=(), race_ids=(), dead_ids=(), refill=None):
        self._offers = list(offers)
        self._bad, self._race, self._dead = set(bad_ids), set(race_ids), set(dead_ids)
        self._refill = refill
        self.offers_calls = 0
        self.live: dict[str, Offer] = {}
        self.rented: list[str] = []
        self.destroyed: list[str] = []
        self._n = 0
        self._lock = threading.Lock()

    def offers(self, spec):
        with self._lock:
            self.offers_calls += 1
            return list(self._offers if self.offers_calls == 1 else (self._refill or []))

    @contextlib.contextmanager
    def rent(self, offer, launch, *, timeout_s=600):
        if offer.id in self._race:                        # taken between offers/rent
            raise RentUnavailable(f"offer {offer.id} taken")
        with self._lock:
            iid = str(1000 + self._n); self._n += 1
            self.rented.append(offer.id); self.live[iid] = offer
        try:
            if offer.id in self._bad:
                raise HostProbeFailed(f"bad host {offer.id}")
            yield RentedHost(id=iid, ssh_host="10.0.0.1", ssh_port=22, offer=offer)
        finally:
            with self._lock:
                self.live.pop(iid, None)                  # leak-proof teardown

    def dead_reason(self, instance_id):
        offer = self.live.get(str(instance_id))
        if offer is not None and offer.id in self._dead:
            return f"container error on {offer.id}"
        return None

    def destroy(self, instance_id):
        with self._lock:
            self.destroyed.append(str(instance_id))
            self.live.pop(str(instance_id), None)

    def list_instances(self):
        with self._lock:
            return [types.SimpleNamespace(id=int(i), status="running", dph=o.dph)
                    for i, o in self.live.items()]


def _fake_ssh_factory(*, ready=True, run_rc=0, run_out="done"):
    """Route the readiness probe (import / sentinel ls) vs the leg command."""
    def fake_ssh(key, host, port, cmd, timeout=120):
        if "import jax_solitons" in cmd:
            return (0, "") if ready else (1, "ModuleNotFoundError")
        if "worker-ready" in cmd:                          # SentinelReady probe
            return (0, "/tmp/worker-ready") if ready else (0, "")
        return (run_rc, run_out)                            # the leg command
    return fake_ssh


@pytest.fixture
def patched(monkeypatch):
    monkeypatch.setattr(fleet.time, "sleep", lambda s: None)
    monkeypatch.setattr(fleet, "_scp_up", lambda *a, **k: (0, ""))
    monkeypatch.setattr(fleet, "_scp_down", lambda *a, **k: (0, ""))

    def apply(**kw):
        monkeypatch.setattr(fleet, "_ssh", _fake_ssh_factory(**kw))
    return apply


def _exec(prov, tmp_path, **kw):
    kw.setdefault("ready_timeout", 0.2)                     # short busy-wait on timeout
    return FleetExecutor(prov, LAUNCH, local_out_dir=str(tmp_path),
                         host_spec=SPEC, jitter_s=0, ready_poll_s=0,
                         log=lambda *_a: None, **kw)


def test_happy_path_runs_all_legs_and_tears_down(patched, tmp_path):
    patched(ready=True)
    prov = FakeProvider([_offer("a"), _offer("b")])
    legs = [_leg("L1"), _leg("L2")]
    results = _exec(prov, tmp_path).run(legs)
    assert {r.status for r in results} == {"OK"}
    assert len(prov.rented) == 2 and prov.live == {}        # both torn down
    assert [r.label for r in results] == ["L1", "L2"]       # order preserved


def test_resume_skips_already_complete_leg(patched, tmp_path):
    patched(ready=True)
    # pre-create L1's output marker -> it should be SKIPped, only L2 runs
    (tmp_path / "L1").mkdir()
    (tmp_path / "L1" / "manifest.json").write_text("{}")
    legs = [_leg("L1", fetch="out", done_when="manifest.json"),
            _leg("L2", fetch="out", done_when="manifest.json")]
    prov = FakeProvider([_offer("a")])
    # L2's marker won't exist after the no-op fetch -> NO_RESULT (proves it ran)
    results = {r.label: r for r in _exec(prov, tmp_path).run(legs)}
    assert results["L1"].status == "SKIP"
    assert results["L2"].status == "NO_RESULT"
    assert prov.rented == ["a"]                             # only L2 rented a box


def test_fails_over_bad_host(patched, tmp_path):
    patched(ready=True)
    prov = FakeProvider([_offer("bad"), _offer("good")], bad_ids={"bad"})
    [r] = _exec(prov, tmp_path).run([_leg("L1")])
    assert r.status == "OK" and prov.rented == ["bad", "good"]
    assert prov.live == {}                                   # both attempts torn down


def test_fails_over_offer_race(patched, tmp_path):
    patched(ready=True)
    prov = FakeProvider([_offer("taken"), _offer("free")], race_ids={"taken"})
    [r] = _exec(prov, tmp_path).run([_leg("L1")])
    assert r.status == "OK" and prov.rented == ["free"]      # raced offer never rented


def test_fast_fail_dead_host_via_api_status(patched, tmp_path):
    """A host that comes up SSH-reachable but whose container died never passes
    the ready probe; dead_reason fast-fails it to the next offer (#27). Here both
    hosts stay 'not ready': 'dead' fast-fails at once via the API, 'good' only
    times out -- both get tried, then the pool exhausts."""
    patched(ready=False)                                    # ready probe never passes
    prov = FakeProvider([_offer("dead"), _offer("good")], dead_ids={"dead"})
    [r] = _exec(prov, tmp_path, max_refills=0).run([_leg("L1")])
    assert prov.rented == ["dead", "good"]
    assert r.status == "NO_OFFERS"
    assert prov.live == {}


def test_offer_pool_refreshes_when_drained(patched, tmp_path):
    """First offer is bad and drains the pool; a refill query supplies a good
    one so the leg still completes (#28)."""
    patched(ready=True)
    prov = FakeProvider([_offer("bad")], bad_ids={"bad"}, refill=[_offer("good")])
    [r] = _exec(prov, tmp_path).run([_leg("L1")])
    assert r.status == "OK" and prov.rented == ["bad", "good"]
    assert prov.offers_calls >= 2                            # re-queried after drain


def test_no_offers_when_pool_exhausts(patched, tmp_path):
    patched(ready=True)
    prov = FakeProvider([_offer("bad")], bad_ids={"bad"})    # no refill
    [r] = _exec(prov, tmp_path, max_refills=0).run([_leg("L1")])
    assert r.status == "NO_OFFERS"
    assert prov.live == {}


def test_run_fail_on_nonzero_command(patched, tmp_path):
    patched(ready=True, run_rc=1, run_out="boom")
    prov = FakeProvider([_offer("a")])
    [r] = _exec(prov, tmp_path).run([_leg("L1")])
    assert r.status == "RUN_FAIL" and "boom" in r.detail
    assert prov.live == {}                                   # torn down anyway


def test_no_result_when_marker_absent(patched, tmp_path):
    patched(ready=True)
    prov = FakeProvider([_offer("a")])
    [r] = _exec(prov, tmp_path).run([_leg("L1", fetch="out", done_when="manifest.json")])
    assert r.status == "NO_RESULT"                           # ran ok but no marker


def test_duplicate_labels_rejected(patched, tmp_path):
    patched(ready=True)
    ex = _exec(FakeProvider([_offer("a")]), tmp_path)
    with pytest.raises(ValueError, match="unique"):
        ex.run([_leg("dup"), _leg("dup")])


def test_empty_legs_is_noop(patched, tmp_path):
    patched(ready=True)
    prov = FakeProvider([_offer("a")])
    assert _exec(prov, tmp_path).run([]) == []
    assert prov.rented == []


def test_sentinel_ready_probe(patched, tmp_path):
    patched(ready=True)
    prov = FakeProvider([_offer("a")])
    ex = _exec(prov, tmp_path, ready=SentinelReady())
    [r] = ex.run([_leg("L1")])
    assert r.status == "OK"


def test_sentinel_bad_network_fails_over(monkeypatch, tmp_path):
    """The onstart net-probe sentinel makes a throttled host fail over at once."""
    monkeypatch.setattr(fleet.time, "sleep", lambda s: None)
    monkeypatch.setattr(fleet, "_scp_up", lambda *a, **k: (0, ""))
    monkeypatch.setattr(fleet, "_scp_down", lambda *a, **k: (0, ""))

    def ssh(key, host, port, cmd, timeout=120):
        if "worker-ready" in cmd:                           # the SentinelReady ls
            return (0, "/tmp/worker-bad-network")           # only the BAD sentinel
        return (0, "done")
    monkeypatch.setattr(fleet, "_ssh", ssh)
    # every host reports bad-network -> each fails over -> pool exhausts
    prov = FakeProvider([_offer("bad"), _offer("good")])
    [r] = _exec(prov, tmp_path, ready=SentinelReady(), max_refills=0).run([_leg("L1")])
    assert r.status == "NO_OFFERS"
    assert prov.rented == ["bad", "good"]


def test_destroy_live_backstop(tmp_path):
    """The signal backstop force-destroys every tracked rental."""
    prov = FakeProvider([_offer("a")])
    ex = _exec(prov, tmp_path)
    ex._track("1000"); ex._track("1001")
    ex._destroy_live()
    assert set(prov.destroyed) == {"1000", "1001"}


def test_signal_guard_installs_and_restores(tmp_path):
    prov = FakeProvider([_offer("a")])
    ex = _exec(prov, tmp_path)
    before = signal.getsignal(signal.SIGTERM)
    with ex._signal_guard():
        assert signal.getsignal(signal.SIGTERM) not in (before, None)  # installed
    assert signal.getsignal(signal.SIGTERM) is before                  # restored


def test_signal_handler_destroys_then_chains(tmp_path):
    prov = FakeProvider([_offer("a")])
    ex = _exec(prov, tmp_path)
    ex._track("1000")
    chained = []
    guard = fleet._SignalGuard(ex)
    guard._prev = {signal.SIGTERM: lambda *_a: chained.append(True)}
    guard._handle(signal.SIGTERM, None)
    assert prov.destroyed == ["1000"] and chained == [True]


def test_fleet_status_reports_live_and_spend(tmp_path):
    from jax_solitons.campaign.vast import VastLedger
    prov = FakeProvider([_offer("a")])
    prov.live["1000"] = _offer("a", dph=0.2)
    led = VastLedger(tmp_path / "l.jsonl")
    led.record("destroyed", outcome="ok", billed_s=120, est_cost_usd=0.05)
    snap = fleet_status(prov, led)
    assert snap["live_dph"] == 0.2 and len(snap["live"]) == 1
    assert snap["ledger"]["total_est_cost_usd"] == 0.05
    assert snap["ledger"]["by_outcome"] == {"ok": 1}

"""VastClient lifecycle tests with a mocked HTTP layer — no network, no spend.

All HTTP goes through `vast._req`, so a scriptable fake covers offers/create/
status/list/destroy and the cost-safety `rent()` teardown logic (the part that
must surface a leak loudly rather than pass silently).
"""

import pytest

import jax_solitons.campaign.vast as vast
from jax_solitons.campaign.vast import (
    HostProbeFailed,
    Offer,
    VastClient,
    VastError,
    VastLedger,
)

OFFER = Offer(id=111, dph=0.15, gpu_name="RTX 3090", num_gpus=1, reliability=0.99,
              inet_down_mbps=900, cuda_max=12.5, geolocation=", CA")


class FakeVast:
    """Scriptable replacement for vast._req, routing by (method, url)."""

    def __init__(self, *, start_status="running", status_msg="",
                 destroy_fail=False, destroy_noop=False):
        self.inst = {}                       # id -> instance dict
        self.start_status, self.status_msg = start_status, status_msg
        self.destroy_fail, self.destroy_noop = destroy_fail, destroy_noop
        self.offers = [
            dict(id=111, dph_total=0.15, gpu_name="RTX 3090", num_gpus=1,
                 reliability2=0.99, inet_down=900, cuda_max_good=12.5, geolocation=", CA"),
            dict(id=112, dph_total=0.60, gpu_name="RTX 3090", num_gpus=1,
                 reliability2=0.99, inet_down=900, cuda_max_good=12.5, geolocation=", CA"),
        ]

    def __call__(self, method, url, key, payload=None, timeout=30):
        if method == "POST" and "/bundles/" in url:
            return {"offers": self.offers}
        if method == "PUT" and "/asks/" in url:
            self.inst[9001] = {"actual_status": self.start_status,
                               "dph_total": 0.15, "status_msg": self.status_msg}
            return {"new_contract": 9001}
        if method == "GET" and "/api/v1/instances/" in url:
            return {"instances": [{"id": i, **d} for i, d in self.inst.items()]}
        if method == "GET" and "/api/v0/instances/" in url:
            iid = int(url.split("/instances/")[1].split("/")[0])
            return {"instances": self.inst.get(iid, {})}
        if method == "DELETE" and "/instances/" in url:
            if self.destroy_fail:
                raise VastError("destroy boom")
            if not self.destroy_noop:
                iid = int(url.split("/instances/")[1].split("/")[0])
                self.inst.pop(iid, None)
            return {}
        return {}


@pytest.fixture
def mk(monkeypatch):
    monkeypatch.setenv("VAST_API_KEY", "testkey")
    monkeypatch.setattr(vast.time, "sleep", lambda s: None)   # don't really wait

    def make(fake, ledger=None):
        monkeypatch.setattr(vast, "_req", fake)
        return VastClient(ledger=ledger)
    return make


def test_cheapest_offer_filters_by_price(mk):
    o = mk(FakeVast()).cheapest_offer(
        max_dph=0.30, min_reliability=0.9, min_inet_mbps=100, min_cuda=12.0)
    assert o.id == 111 and o.dph == 0.15        # 112 ($0.60) filtered out


def test_rent_happy_path_destroys_and_verifies_gone(mk, tmp_path):
    led = VastLedger(tmp_path / "l.jsonl")
    c = mk(FakeVast(start_status="running"), ledger=led)
    with c.rent(OFFER, image="img", onstart_cmd="cmd", timeout_s=5) as iid:
        assert iid == 9001
    evs = led.events()
    assert {e["event"] for e in evs} == {"rented", "running", "destroyed"}
    d = next(e for e in evs if e["event"] == "destroyed")
    assert d["destroyed"] is True and d["verify"] == "gone"
    assert "est_cost_usd" in d


def test_rent_raises_loudly_on_failed_destroy(mk, tmp_path):
    led = VastLedger(tmp_path / "l.jsonl")
    c = mk(FakeVast(start_status="running", destroy_fail=True), ledger=led)
    with pytest.raises(VastError, match="LEAK RISK"):
        with c.rent(OFFER, image="img", onstart_cmd="cmd", timeout_s=5):
            pass
    d = next(e for e in led.events() if e["event"] == "destroyed")
    assert d["destroyed"] is False             # recorded, not silently passed


def test_rent_raises_when_instance_still_present(mk):
    # destroy "succeeds" but the instance never leaves the list -> confirmed leak
    c = mk(FakeVast(start_status="running", destroy_noop=True))
    with pytest.raises(VastError, match="LEAK RISK"):
        with c.rent(OFFER, image="img", onstart_cmd="cmd", timeout_s=5):
            pass


def test_wait_running_bails_on_bad_host(mk):
    fake = FakeVast(start_status="loading",
                    status_msg="failed to resolve auth.docker.io")
    c = mk(fake)
    iid = c.create(OFFER.id, image="img", onstart_cmd="cmd")
    with pytest.raises(HostProbeFailed):
        c.wait_running(iid, timeout_s=5, poll_s=0)


def test_list_instances_uses_v1(mk):
    fake = FakeVast()
    c = mk(fake)
    fake.inst[42] = {"actual_status": "running", "dph_total": 0.2}
    assert [i.id for i in c.list_instances()] == [42]


def test_read_key_from_file(tmp_path, monkeypatch):
    monkeypatch.delenv("VAST_API_KEY", raising=False)
    kf = tmp_path / "vast_key"
    kf.write_text("filekey\n")
    monkeypatch.setattr(vast, "_KEY_PATHS", (str(kf),))
    assert vast._read_key() == "filekey"


def test_req_raises_vasterror_on_http_error(monkeypatch):
    import io
    import urllib.error

    def boom(req, timeout=30):
        raise urllib.error.HTTPError(req.full_url, 410, "Gone", {}, io.BytesIO(b"dead"))
    monkeypatch.setattr(vast.urllib.request, "urlopen", boom)
    with pytest.raises(VastError, match="410"):
        vast._req("GET", "https://x/api/v0/instances/", "k")


def test_logs_polls_result_url(monkeypatch):
    monkeypatch.setenv("VAST_API_KEY", "k")
    monkeypatch.setattr(vast, "_req",
                        lambda *a, **k: {"result_url": "https://s3/log.txt"})

    class Resp:
        status = 200
        def read(self): return b"onstart output\n=== DONE ==="
        def __enter__(self): return self
        def __exit__(self, *a): return False
    monkeypatch.setattr(vast.urllib.request, "urlopen",
                        lambda url, timeout=20: Resp())
    assert "DONE" in VastClient().logs(123)


def test_rent_failed_over_host_logs_outcome(mk, tmp_path):
    led = VastLedger(tmp_path / "l.jsonl")
    c = mk(FakeVast(start_status="loading", status_msg="failed to resolve x"),
           ledger=led)
    with pytest.raises(HostProbeFailed):
        with c.rent(OFFER, image="img", onstart_cmd="cmd", timeout_s=5):
            pass
    d = next(e for e in led.events() if e["event"] == "destroyed")
    assert d["outcome"] == "host_failed" and d["verify"] == "gone"

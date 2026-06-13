"""Direct Vast.ai HTTP client for the campaign Executor (stdlib only).

SkyPilot's Vast provider AND the official `vastai` SDK both break against Vast's
current API: the bare collection endpoint ``GET /api/v0/instances/`` returns
HTTP 410 Gone, and both route instance-listing there. Probing showed the rest
of v0 is alive (``/instances/{id}/`` -> 200), and v1 serves the list. So this
talks straight to the endpoints that work -- **v1 for listing, v0 sub-resources
for create/destroy/show/logs** -- with no third-party dependency (urllib only),
which also means nothing to break when the SDK lags the API again.

Cost safety is structural: ``rent()`` ALWAYS destroys on exit (success,
exception, or Ctrl-C) and then verifies via the independent v1 list endpoint
that the instance is actually gone. A leaked GPU bills by the second.

API key: ``$VAST_API_KEY`` or ``~/.config/vastai/vast_api_key``.
"""

from __future__ import annotations

import contextlib
import dataclasses
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

V0 = "https://console.vast.ai/api/v0"
V1 = "https://console.vast.ai/api/v1"
_KEY_PATHS = ("~/.config/vastai/vast_api_key", "~/.vast_api_key")

# Host-failure signatures seen in an instance's status_msg DURING provisioning.
# A host can advertise high reliability + bandwidth and still be unable to
# resolve DNS / pull an image -- the metadata lies (P9). Catching these lets
# wait_running bail in seconds instead of waiting out the full timeout.
_BAD_HOST_SIGNS = (
    "failed to resolve", "i/o timeout", "failed to authorize", "no such host",
    "connection refused", "error response from daemon", "manifest unknown",
    "unauthorized", "temporary failure in name resolution", "no space left",
)


class VastError(RuntimeError):
    """A Vast API call failed (status quoted, key never echoed)."""


class HostProbeFailed(VastError):
    """A rented host failed to come up usable (DNS/image-pull/disk) -- the P9
    'hosts lie' case. The executor should tear it down and fail over."""


def _read_key(explicit: str | None = None) -> str:
    if explicit:
        return explicit
    if os.environ.get("VAST_API_KEY"):
        return os.environ["VAST_API_KEY"].strip()
    for p in _KEY_PATHS:
        fp = Path(p).expanduser()
        if fp.exists():
            return fp.read_text().strip()
    raise RuntimeError(
        "no Vast API key: set $VAST_API_KEY or write ~/.config/vastai/vast_api_key")


def _req(method: str, url: str, key: str, payload=None, timeout: float = 30):
    data = json.dumps(payload).encode() if payload is not None else None
    headers = {"Authorization": f"Bearer {key}", "Accept": "application/json"}
    if data is not None:
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read()
            return json.loads(body) if body else {}
    except urllib.error.HTTPError as e:
        detail = e.read()[:200].decode(errors="replace")
        raise VastError(f"{method} {url.split('?')[0]} -> HTTP {e.code}: {detail}")


@dataclasses.dataclass(frozen=True)
class Offer:
    """A rentable offer (the fields the executor decides on)."""

    id: int
    dph: float                 # $ per hour, all-in
    gpu_name: str
    num_gpus: int
    reliability: float         # 0..1 (Vast's reliability2)
    inet_down_mbps: float
    cuda_max: float
    geolocation: str

    def __str__(self) -> str:
        return (f"offer {self.id}: {self.num_gpus}x {self.gpu_name} "
                f"${self.dph:.3f}/hr  rel={self.reliability:.3f}  "
                f"down={self.inet_down_mbps:.0f}Mbps  cuda<={self.cuda_max}  "
                f"[{self.geolocation.strip(', ')}]")


@dataclasses.dataclass(frozen=True)
class Instance:
    id: int
    status: str
    dph: float
    raw: dict


class VastLedger:
    """Append-only JSONL receipts of Vast host outcomes (P9: write what you
    measured). Every rental, its probe result, and its cost, logged as it
    happens -- the infrastructure analogue of the campaign EventSink. Records:
    `rented`, `running`, `destroyed` (carrying outcome + billed seconds + est
    cost), one JSON object per line.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    @classmethod
    def default(cls) -> "VastLedger":
        return cls("~/.jax-solitons/vast-ledger.jsonl")

    def record(self, event: str, **fields) -> dict:
        rec = {"ts": round(time.time(), 3), "event": event, **fields}
        with self.path.open("a") as f:
            f.write(json.dumps(rec, sort_keys=True) + "\n")
        return rec

    def events(self) -> list[dict]:
        if not self.path.exists():
            return []
        return [json.loads(ln) for ln in self.path.read_text().splitlines()
                if ln.strip()]

    def summary(self) -> dict:
        """Tally outcomes and total spend across all logged rentals."""
        evs = self.events()
        destroyed = [e for e in evs if e["event"] == "destroyed"]
        by_outcome: dict[str, int] = {}
        for e in destroyed:
            o = e.get("outcome", "?")
            by_outcome[o] = by_outcome.get(o, 0) + 1
        return {
            "rentals": sum(1 for e in evs if e["event"] == "rented"),
            "by_outcome": by_outcome,
            "total_billed_min": round(sum(e.get("billed_s", 0)
                                          for e in destroyed) / 60, 1),
            "total_est_cost_usd": round(sum(e.get("est_cost_usd", 0)
                                            for e in destroyed), 6),
        }


class VastClient:
    """Direct HTTP client for Vast.ai, scoped to what the executor needs."""

    def __init__(self, api_key: str | None = None,
                 ledger: "VastLedger | None" = None):
        self.key = _read_key(api_key)
        self.ledger = ledger

    def _log(self, event: str, **fields) -> None:
        if self.ledger is not None:
            self.ledger.record(event, **fields)

    # -- discovery (free; no spend) ------------------------------------------
    def offers(self, *, gpu_name: str = "RTX_3090", num_gpus: int = 1,
               max_dph: float = 0.40, min_reliability: float = 0.95,
               min_inet_mbps: float = 100.0, min_cuda: float = 12.0) -> list[Offer]:
        """Rentable offers meeting the bar, cheapest first.

        These filters are the P9 admission spirit at SELECTION time -- but the
        live failure (a 0.99-reliability host that can't resolve DNS) shows
        selection metadata is necessary, not sufficient. The executor must still
        probe-and-bail per host (see wait_running) and fail over to the next."""
        q = {
            "verified": {"eq": True}, "rentable": {"eq": True},
            "rented": {"eq": False},
            "gpu_name": {"eq": gpu_name.replace("_", " ")},
            "num_gpus": {"eq": num_gpus},
            "reliability2": {"gte": min_reliability},
            "inet_down": {"gte": min_inet_mbps},
            "cuda_max_good": {"gte": min_cuda},
            "order": [["dph_total", "asc"]], "type": "on-demand",
            "limit": 64, "allocated_storage": 5.0,
        }
        raw = _req("POST", f"{V0}/bundles/", self.key, q).get("offers", [])
        return [
            Offer(id=int(o["id"]), dph=float(o["dph_total"]), gpu_name=o["gpu_name"],
                  num_gpus=int(o["num_gpus"]),
                  reliability=float(o.get("reliability2", 0)),
                  inet_down_mbps=float(o.get("inet_down", 0)),
                  cuda_max=float(o.get("cuda_max_good", 0)),
                  geolocation=o.get("geolocation", ""))
            for o in raw if o.get("dph_total", 1e9) <= max_dph]

    def cheapest_offer(self, **kw) -> Offer | None:
        offs = self.offers(**kw)
        return offs[0] if offs else None

    # -- lifecycle -----------------------------------------------------------
    def create(self, offer_id: int, *, image: str, onstart_cmd: str,
               disk: float = 40.0, label: str = "jax-solitons",
               runtype: str = "ssh") -> int:
        """Create an instance from an offer; returns the instance id."""
        blob = {"client_id": "me", "image": image, "env": {}, "disk": disk,
                "label": label, "onstart": onstart_cmd, "runtype": runtype}
        res = _req("PUT", f"{V0}/asks/{offer_id}/", self.key, blob)
        new_id = res.get("new_contract") or res.get("id")
        if not new_id:
            raise VastError(f"create returned no instance id: {res}")
        return int(new_id)

    def status(self, instance_id: int) -> Instance:
        d = _req("GET", f"{V0}/instances/{instance_id}/?owner=me", self.key)
        inst = d.get("instances", d) or {}
        return Instance(id=instance_id, status=inst.get("actual_status", "?"),
                        dph=float(inst.get("dph_total", 0) or 0), raw=inst)

    def list_instances(self) -> list[Instance]:
        """All of the account's instances, via the LIVE v1 endpoint (the v0
        collection is 410-dead). This is the cost-safety verify."""
        d = _req("GET", f"{V1}/instances/", self.key)
        return [Instance(id=int(x["id"]), status=x.get("actual_status", "?"),
                         dph=float(x.get("dph_total", 0) or 0), raw=x)
                for x in d.get("instances", [])]

    def wait_running(self, instance_id: int, *, timeout_s: float = 600,
                     poll_s: float = 10) -> Instance:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            inst = self.status(instance_id)
            if inst.status == "running":
                return inst
            msg = str(inst.raw.get("status_msg") or "")
            if any(s in msg.lower() for s in _BAD_HOST_SIGNS):
                raise HostProbeFailed(
                    f"host probe failed (instance {instance_id}): {msg[:160]}")
            if inst.status in ("error", "exited"):
                raise HostProbeFailed(
                    f"instance {instance_id} -> {inst.status}: {msg[:160]}")
            time.sleep(poll_s)
        raise TimeoutError(f"instance {instance_id} not running in {timeout_s}s")

    def logs(self, instance_id: int, tail: int = 2000) -> str:
        """Onstart/container logs: request, then poll the (auth-less) result URL."""
        rj = _req("PUT", f"{V0}/instances/request_logs/{instance_id}/",
                  self.key, {"tail": str(tail)})
        url = rj.get("result_url")
        if not url:
            return json.dumps(rj)
        for _ in range(12):
            try:
                with urllib.request.urlopen(url, timeout=20) as resp:
                    if resp.status == 200:
                        return resp.read().decode(errors="replace")
            except urllib.error.HTTPError as e:
                if e.code not in (403, 404):
                    raise
            time.sleep(2)
        return ""

    def destroy(self, instance_id: int) -> None:
        _req("DELETE", f"{V0}/instances/{instance_id}/", self.key, {})

    # -- the safety primitive ------------------------------------------------
    @contextlib.contextmanager
    def rent(self, offer: Offer, *, image: str, onstart_cmd: str,
             disk: float = 40.0, wait: bool = True, timeout_s: float = 600):
        """Rent an instance and GUARANTEE teardown on exit, logging the full
        lifecycle to the ledger.

        Destroys in a finally block on any exit -- normal, exception, or
        Ctrl-C -- then verifies via the independent v1 list that the instance
        is gone. Teardown is not best-effort: a leaked GPU bills by the second.
        `offer` is an Offer (not a bare id) so cost and geo land in the ledger.
        """
        t0 = time.monotonic()
        instance_id = self.create(offer.id, image=image, onstart_cmd=onstart_cmd,
                                  disk=disk)
        self._log("rented", offer_id=offer.id, instance_id=instance_id,
                  gpu=offer.gpu_name, dph=offer.dph, reliability=offer.reliability,
                  geo=offer.geolocation.strip(", "))
        outcome, reason = "ok", ""
        try:
            if wait:
                self.wait_running(instance_id, timeout_s=timeout_s)
                self._log("running", offer_id=offer.id, instance_id=instance_id,
                          provision_s=round(time.monotonic() - t0, 1))
            yield instance_id
        except HostProbeFailed as e:
            outcome, reason = "host_failed", str(e)
            raise
        except TimeoutError as e:
            outcome, reason = "timeout", str(e)
            raise
        except BaseException as e:               # propagate, but record the cause
            outcome, reason = type(e).__name__, str(e)[:200]
            raise
        finally:
            billed_s = time.monotonic() - t0
            destroyed = False
            for _ in range(5):
                try:
                    self.destroy(instance_id)
                    destroyed = True
                    break
                except Exception:
                    time.sleep(2)
            leaked = False
            with contextlib.suppress(Exception):
                leaked = any(i.id == instance_id for i in self.list_instances())
            self._log("destroyed", offer_id=offer.id, instance_id=instance_id,
                      outcome=outcome, reason=reason[:200],
                      billed_s=round(billed_s, 1),
                      est_cost_usd=round(offer.dph * billed_s / 3600, 4),
                      destroyed=destroyed, leaked=leaked)
            if leaked:
                raise VastError(
                    f"LEAK: instance {instance_id} still present after destroy -- "
                    f"run `vastai destroy instance {instance_id}`")

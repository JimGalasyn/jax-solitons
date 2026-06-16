"""FleetExecutor: a parallel, one-host-per-leg *script* fleet over any Provider.

The 2026-06-15 farming session ran three near-identical hand-rolled drivers
(`run_eps_fleet`, `run_stability_fleet`, `run_eps_kick_fleet`), each copy-pasting
the same loop: pull an offer, rent it, wait for the worker, scp a driver script
up, ssh-run it, scp the output dir back, fail over bad hosts. `ProviderExecutor`
runs the *structured* campaign worker (one RunConfig -> one RunFn record) and
sequentially on a single host; it does not cover the "ship an arbitrary script +
args, fetch an output glob, one rented host per leg, in parallel" shape those
drivers need. `FleetExecutor` is that shape, factored once (#25): a fleet run is
**data** -- a list of `FleetLeg(label, command, ship, fetch)` -- not a forked
script, so the three drivers collapse to thin callers that build legs.

It is physics-agnostic (no model/stepper import): the only thing crossing the
boundary is a shell `command` the box runs and the files it ships/fetches. The
robustness the farming session paid for is built in:

  - per-leg failover on a bad/unreachable host (`HostProbeFailed`) or an offer
    race (`RentUnavailable`) -> pull the next offer;
  - **fast-fail a corpse** via the provider's API status (#27) instead of
    ssh-polling a container that never came up for the full deadline;
  - **refresh the offer pool** when it drains under heavy failover (#28);
  - **resume**: a leg whose output already exists locally is skipped (#26);
  - **launch jitter** so N legs don't fire N simultaneous DNS lookups (#29);
  - **signal-safe teardown** (#24): a SIGTERM/SIGINT mid-run still destroys every
    in-flight rental, a backstop to each `rent()`'s own leak-proof teardown.

Transient REST retry (#23) and the leak-proof teardown live in the Provider
(`vast.py`), so they apply here for free.
"""

from __future__ import annotations

import concurrent.futures as cf
import dataclasses
import os
import shlex
import signal
import threading
import time
from collections import deque
from collections.abc import Iterable
from pathlib import Path

from jax_solitons.campaign.protocols import (
    HostProbeFailed,
    HostSpec,
    LaunchSpec,
    Provider,
    RentedHost,
    RentUnavailable,
)
from jax_solitons.campaign.provider_exec import (
    DEFAULT_KEY,
    _scp_down,
    _scp_up,
    _ssh,
)


@dataclasses.dataclass(frozen=True)
class FleetLeg:
    """One unit of fleet work: ship inputs, run a command, fetch outputs.

      label    unique id; names the local output subdir AND the resume key
      command  shell command run on the box, cwd = `remote_work_dir`
      ship     local paths scp'd up to `remote_work_dir` before the command
      fetch    remote path (relative to `remote_work_dir`, or absolute) scp'd
               back into `<local_out_dir>/<label>/` after a rc==0 command
      done_when  local path under `<local_out_dir>/<label>/` whose existence
               means "already complete" -- the resume/skip marker (#26). Defaults
               to the basename of `fetch`; a more precise marker (e.g.
               ``"out_kick/manifest.json"``) makes resume robust to a partial
               fetch.
    """

    label: str
    command: str
    ship: tuple[str, ...] = ()
    fetch: str = ""
    done_when: str = ""

    def marker(self) -> str:
        """The local relative path that signals this leg is complete."""
        return self.done_when or (os.path.basename(self.fetch.rstrip("/"))
                                  if self.fetch else "")


@dataclasses.dataclass(frozen=True)
class LegResult:
    """The outcome of one leg. `status` is one of:
    OK | SKIP | NO_RESULT | RUN_FAIL | NO_OFFERS | LEAK | ERROR."""

    label: str
    status: str
    host_id: str | None = None
    detail: str = ""

    @property
    def ok(self) -> bool:
        return self.status in ("OK", "SKIP")


# --------------------------------------------------------------- readiness ----
# How a leg knows its box is ready to run work. The Provider's rent() only
# guarantees SSH-reachable; the onstart bootstrap (engine install, net probe)
# may still be running, so probe for it (P9). A probe returns True (ready) /
# False (not yet) or raises HostProbeFailed (the host announced it is bad).
class ImportReady:
    """Ready when `python -c 'import <module>'` succeeds on the box -- the engine
    bootstrap (onstart) has finished installing jax-solitons."""

    def __init__(self, python: str = "/workspace/jaxenv/bin/python",
                 module: str = "jax_solitons"):
        self.python, self.module = python, module

    def check(self, ssh, host: RentedHost) -> bool:
        rc, _out = ssh(f"{self.python} -c 'import {self.module}'", timeout=30)
        return rc == 0


class SentinelReady:
    """Ready when an onstart-written sentinel file appears; a `bad` sentinel
    (the onstart net-probe bailing on a throttled host) raises HostProbeFailed so
    the leg fails over at once instead of waiting out the deadline."""

    def __init__(self, ok: str = "/tmp/worker-ready",
                 bad: str = "/tmp/worker-bad-network"):
        self.ok, self.bad = ok, bad

    def check(self, ssh, host: RentedHost) -> bool:
        probe = f"ls {shlex.quote(self.ok)} {shlex.quote(self.bad)} 2>/dev/null"
        _rc, out = ssh(probe, timeout=30)
        if self.bad and os.path.basename(self.bad) in out:
            raise HostProbeFailed("onstart net-probe bailed (throttled host)")
        return os.path.basename(self.ok) in out


# --------------------------------------------------------------- offer pool ----
class _OfferPool:
    """A thread-safe, cheapest-first offer queue that re-queries the Provider
    when it drains under failover (#28). `offers()` returns offers sorted asc, so
    popleft is the cheapest remaining. Refills are capped so a genuinely empty
    marketplace ends the run instead of spinning."""

    def __init__(self, provider: Provider, spec: HostSpec, *, max_refills: int = 3):
        self._provider, self._spec, self._max_refills = provider, spec, max_refills
        self._lock = threading.Lock()
        self._q: deque = deque(provider.offers(spec))
        self._refills = 0
        self.initial = len(self._q)

    def get(self):
        """Next offer, refilling once if drained; None when truly exhausted."""
        with self._lock:
            if not self._q and self._refills < self._max_refills:
                self._refills += 1
                self._q.extend(self._provider.offers(self._spec))
            return self._q.popleft() if self._q else None


# --------------------------------------------------------------- executor ----
class FleetExecutor:
    """Run a list of `FleetLeg`s, one rented host per leg, in parallel."""

    def __init__(self, provider: Provider, launch: LaunchSpec, *,
                 local_out_dir: str, host_spec: HostSpec | None = None,
                 ready=None, key_path: str = DEFAULT_KEY,
                 remote_work_dir: str = "/workspace",
                 max_parallel: int = 12, rent_timeout: float = 600,
                 ready_timeout: float = 1200, ready_poll_s: float = 15,
                 run_timeout: float = 9000, jitter_s: float = 2.0,
                 max_refills: int = 3, ledger=None, log=print):
        self.provider = provider
        self.launch = launch
        self.local_out_dir = Path(local_out_dir)
        self.host_spec = host_spec or HostSpec()
        self.ready = ready or ImportReady()
        self.key_path = key_path
        self.remote_work_dir = remote_work_dir
        self.max_parallel = max(1, max_parallel)
        self.rent_timeout = rent_timeout
        self.ready_timeout = ready_timeout
        self.ready_poll_s = ready_poll_s
        self.run_timeout = run_timeout
        self.jitter_s = jitter_s
        self.max_refills = max_refills
        self.ledger = ledger
        self._log = log
        # Live rentals, for the signal-safe teardown backstop (#24): instance id
        # -> True while a leg holds the host. Guarded; touched from worker threads.
        self._live: dict[str, bool] = {}
        self._live_lock = threading.Lock()

    # -- resume (#26) --------------------------------------------------------
    def _leg_dir(self, leg: FleetLeg) -> Path:
        return self.local_out_dir / leg.label

    def _complete(self, leg: FleetLeg) -> bool:
        """True if this leg's output already exists locally -- skip it on a
        relaunch. With no marker we cannot tell, so it is never pre-skipped."""
        marker = leg.marker()
        return bool(marker) and (self._leg_dir(leg) / marker).exists()

    # -- per-host steps ------------------------------------------------------
    def _wait_ready(self, host: RentedHost) -> None:
        """Block until the box is ready (the `ready` probe passes), failing over
        if it announces bad OR the provider's API reports it dead (#27)."""
        def ssh(cmd, timeout=30):
            return _ssh(self.key_path, host.ssh_host, host.ssh_port, cmd, timeout)

        dead_reason = getattr(self.provider, "dead_reason", None)
        deadline = time.monotonic() + self.ready_timeout
        while time.monotonic() < deadline:
            if self.ready.check(ssh, host):              # may raise HostProbeFailed
                return
            if dead_reason is not None:                  # fast-fail a corpse (#27)
                reason = dead_reason(host.id)
                if reason:
                    raise HostProbeFailed(f"fast-fail {host.id}: {reason}")
            time.sleep(self.ready_poll_s)
        raise HostProbeFailed(
            f"worker not ready on {host.id} within {self.ready_timeout}s")

    def _ship(self, host: RentedHost, leg: FleetLeg) -> None:
        for src in leg.ship:
            rc, out = _scp_up(self.key_path, host.ssh_host, host.ssh_port,
                              str(src), self.remote_work_dir + "/")
            if rc != 0:
                raise HostProbeFailed(f"scp-up {src} -> {host.id} failed: {out[-160:]}")

    def _fetch(self, host: RentedHost, leg: FleetLeg) -> None:
        if not leg.fetch:
            return
        remote = (leg.fetch if leg.fetch.startswith("/")
                  else f"{self.remote_work_dir}/{leg.fetch}")
        leg_dir = self._leg_dir(leg)
        leg_dir.mkdir(parents=True, exist_ok=True)
        _scp_down(self.key_path, host.ssh_host, host.ssh_port, remote, str(leg_dir))

    # -- one leg, with failover ---------------------------------------------
    def _run_leg(self, pool: _OfferPool, leg: FleetLeg, idx: int) -> LegResult:
        # Launch jitter (#29): stagger starts within a parallel wave so we don't
        # fire max_parallel simultaneous resolver lookups (the thundering-herd
        # insurance, independent of any one cause). Deterministic (no RNG): the
        # i-th of each wave waits i*jitter.
        if self.jitter_s:
            time.sleep((idx % self.max_parallel) * self.jitter_s)
        tried = 0
        while True:
            offer = pool.get()
            if offer is None:
                return LegResult(leg.label, "NO_OFFERS", None,
                                 f"offer pool drained after {tried} attempt(s)")
            tried += 1
            try:
                with self.provider.rent(offer, self.launch,
                                        timeout_s=self.rent_timeout) as host:
                    self._track(host.id)
                    try:
                        self._wait_ready(host)
                        self._ship(host, leg)
                        cmd = f"cd {shlex.quote(self.remote_work_dir)} && {leg.command}"
                        rc, out = _ssh(self.key_path, host.ssh_host, host.ssh_port,
                                       cmd, timeout=self.run_timeout)
                        # host.id is the rented INSTANCE id (not offer.id), so a
                        # result correlates to the provider's live-instance list,
                        # `_track`, and `dead_reason`.
                        if rc != 0:
                            return LegResult(leg.label, "RUN_FAIL", host.id,
                                             f"rc={rc}: {out[-240:]}")
                        self._fetch(host, leg)
                        done = self._complete(leg) or not leg.marker()
                        return LegResult(leg.label, "OK" if done else "NO_RESULT",
                                         host.id)
                    finally:
                        self._untrack(host.id)
            except (HostProbeFailed, TimeoutError, RentUnavailable) as e:
                self._log(f"  {leg.label}: offer {offer.id} "
                          f"{type(e).__name__} -> failing over")
                continue                                  # bad host / race -> next
            except Exception as e:                        # noqa: BLE001
                # No host handle here (rent may have failed before yielding), so
                # the offer id is the best correlation we have for a terminal error.
                msg = str(e)
                status = "LEAK" if "LEAK" in msg.upper() else "ERROR"
                return LegResult(leg.label, status, offer.id,
                                 f"{type(e).__name__}: {msg[:200]}")

    # -- live-rental registry (signal-safe teardown backstop, #24) -----------
    def _track(self, instance_id: str) -> None:
        with self._live_lock:
            self._live[instance_id] = True

    def _untrack(self, instance_id: str) -> None:
        with self._live_lock:
            self._live.pop(instance_id, None)

    def _destroy_live(self) -> None:
        """Force-destroy every in-flight rental. Each `rent()` already tears down
        on its own exit; this is the backstop for a signal that would otherwise
        kill the process before those finallys run."""
        destroy = getattr(self.provider, "destroy", None)
        if destroy is None:
            return
        # RentedHost.id is a provider-opaque string in the Provider contract, so
        # pass it through as-is (no int cast) and keep the manual-cleanup hint
        # provider-agnostic -- this backstop must not assume a Vast id or CLI.
        name = getattr(self.provider, "name", "provider")
        with self._live_lock:
            ids = list(self._live)
        for iid in ids:
            try:
                destroy(iid)
                self._log(f"  signal teardown: destroyed {iid}")
            except Exception as e:                        # noqa: BLE001
                self._log(f"  signal teardown: FAILED to destroy {iid} ({e}) "
                          f"-- destroy instance {iid} manually via the {name} provider")

    # -- the run -------------------------------------------------------------
    def run(self, legs: Iterable[FleetLeg]) -> list[LegResult]:
        """Run every leg (parallel, one host each), failing over bad hosts and
        skipping already-complete legs. Returns one `LegResult` per input leg."""
        legs = list(legs)
        if not legs:
            return []
        labels = [leg.label for leg in legs]
        if len(set(labels)) != len(labels):
            raise ValueError("FleetLeg labels must be unique (they key output "
                             "dirs and the resume marker)")

        results: dict[str, LegResult] = {}
        pending: list[FleetLeg] = []
        for leg in legs:
            if self._complete(leg):                       # resume/skip (#26)
                results[leg.label] = LegResult(leg.label, "SKIP", None,
                                               "output already present")
            else:
                pending.append(leg)
        if pending:
            self._log(f"{len(pending)} leg(s) to run, {len(results)} skipped "
                      f"(already complete); up to {self.max_parallel} parallel")
            pool = _OfferPool(self.provider, self.host_spec,
                              max_refills=self.max_refills)
            with self._signal_guard():
                with cf.ThreadPoolExecutor(max_workers=self.max_parallel) as ex:
                    futs = {ex.submit(self._run_leg, pool, leg, i): leg
                            for i, leg in enumerate(pending)}
                    for fut in cf.as_completed(futs):
                        r = fut.result()
                        results[r.label] = r
                        self._log(f"LEG {r.label}: {r.status}"
                                  + (f" ({r.detail})" if r.detail else ""))
        return [results[leg.label] for leg in legs]

    def _signal_guard(self):
        """For the duration of `run()`, make SIGTERM/SIGINT tear down live
        rentals before unwinding -- the proactive half of orphan prevention
        (#24), a backstop to each `rent()`'s own leak-proof teardown."""
        return _SignalGuard(self)


class _SignalGuard:
    """Context manager that, for its lifetime, makes SIGTERM/SIGINT tear down the
    executor's live rentals before re-raising. Restores prior handlers on exit.
    Only the MAIN thread can install signal handlers, so a guard requested from a
    worker thread is a no-op (the run() that owns the threads installs it)."""

    _SIGNALS = (signal.SIGTERM, signal.SIGINT)

    def __init__(self, executor: FleetExecutor):
        self._exec = executor
        self._prev: dict[int, object] = {}

    def __enter__(self):
        if threading.current_thread() is not threading.main_thread():  # pragma: no cover
            return self                                   # can't set handlers off-main
        for sig in self._SIGNALS:
            try:
                self._prev[sig] = signal.getsignal(sig)
                signal.signal(sig, self._handle)
            except (ValueError, OSError):                 # pragma: no cover
                self._prev.pop(sig, None)                 # not settable here
        return self

    def _handle(self, signum, frame):
        self._exec._log(f"signal {signum}: tearing down live rentals")
        self._exec._destroy_live()
        prev = self._prev.get(signum)
        if callable(prev):
            prev(signum, frame)                           # chain prior handler
        else:
            raise KeyboardInterrupt                       # default: unwind the run

    def __exit__(self, *exc):
        for sig, prev in self._prev.items():
            if prev is None:                              # was set from C, not Python
                continue                                  # can't restore via signal()
            try:
                signal.signal(sig, prev)                  # restore
            except (ValueError, OSError, TypeError):      # pragma: no cover
                pass
        return False

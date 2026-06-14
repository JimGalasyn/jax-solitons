"""ProviderExecutor: run a campaign over a rented fleet from any `Provider` (F).

The executor (D) that consumes the Provider seam: pull offers, rent a host with
per-host **failover** (a bad host -> next offer), wait for the engine to come up,
ship each config and run the campaign `worker` over SSH, sync the artifacts back,
and rely on the Provider's leak-proof `rent()` for teardown. This is the
principled generalization of the hand-rolled `run_eps_fleet` driver -- it works
over `VastProvider`, `RunPodProvider`, or any future `Provider`.

The physics crosses by name (`run_fn_ref`, a ``'module:function'`` ref the box
imports), never as a closure; configs travel as JSON on the command line. The
box must have jax-solitons installed by the `LaunchSpec.onstart` bootstrap.

v1 runs all configs sequentially on a single rented host. Multi-host parallel
fan-out (the `run_eps_fleet` ThreadPool pattern) is a documented follow-up.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import time
from collections.abc import Iterable

from jax_solitons.campaign.protocols import (
    HostProbeFailed,
    HostSpec,
    LaunchSpec,
    Provider,
    RentedHost,
)
from jax_solitons.campaign.remote import RunFnRef
from jax_solitons.campaign.worker import RESULT_PREFIX
from jax_solitons.runs import RunConfig

DEFAULT_KEY = "~/.ssh/vastai"
# Matches the engine_dogfood vast/onstart.sh, which builds the engine into this env.
DEFAULT_REMOTE_PYTHON = "/workspace/jaxenv/bin/python"


def _ssh(key: str, host: str, port: int, cmd: str, timeout: float = 120):
    """Run one command on the box; returns (rc, combined stdout+stderr)."""
    r = subprocess.run(
        ["ssh", "-i", os.path.expanduser(key), "-o", "StrictHostKeyChecking=no",
         "-o", "ConnectTimeout=15", "-p", str(port), f"root@{host}", cmd],
        capture_output=True, text=True, timeout=timeout)
    return r.returncode, r.stdout + r.stderr


def _scp_down(key: str, host: str, port: int, remote: str, local: str,
              timeout: float = 600):
    r = subprocess.run(
        ["scp", "-i", os.path.expanduser(key), "-o", "StrictHostKeyChecking=no",
         "-P", str(port), "-r", f"root@{host}:{remote}", local],
        capture_output=True, text=True, timeout=timeout)
    return r.returncode, r.stdout + r.stderr


class ProviderExecutor:
    """Run a campaign over hosts rented from a `Provider`, with failover."""

    name = "provider"

    def __init__(self, provider: Provider, run_fn_ref: RunFnRef,
                 launch: LaunchSpec, *, host_spec: HostSpec | None = None,
                 key_path: str = DEFAULT_KEY,
                 remote_python: str = DEFAULT_REMOTE_PYTHON,
                 remote_work_dir: str = "/workspace/runs",
                 local_work_dir: str = "campaign_out",
                 ready_timeout: float = 900, run_timeout: float = 3600,
                 rent_timeout: float = 600):
        self.provider = provider
        self.run_fn_ref = run_fn_ref
        self.launch = launch
        self.host_spec = host_spec or HostSpec()
        self.key_path = key_path
        self.remote_python = remote_python
        self.remote_work_dir = remote_work_dir
        self.local_work_dir = local_work_dir
        self.ready_timeout = ready_timeout
        self.run_timeout = run_timeout
        self.rent_timeout = rent_timeout

    # -- per-host steps ------------------------------------------------------
    def _wait_engine_ready(self, host: RentedHost) -> None:
        """Block until `import jax_solitons` succeeds on the box (the onstart
        bootstrap finished), or raise HostProbeFailed so the run fails over.

        The Provider's rent() only guarantees the host is SSH-reachable; the
        engine install (onstart) may still be running, so probe for it (P9)."""
        check = f"{self.remote_python} -c 'import jax_solitons'"
        deadline = time.monotonic() + self.ready_timeout
        last = ""
        while time.monotonic() < deadline:
            rc, out = _ssh(self.key_path, host.ssh_host, host.ssh_port, check,
                           timeout=30)
            if rc == 0:
                return
            last = out
            time.sleep(10)
        raise HostProbeFailed(
            f"engine not ready on {host.id} within {self.ready_timeout}s: "
            f"{last[-160:]}")

    def _run_config(self, host: RentedHost, config: RunConfig) -> dict:
        """Run the worker for one config over SSH; parse its result record."""
        cmd = (f"{self.remote_python} -m jax_solitons.campaign.worker "
               f"--config-json {shlex.quote(config.to_json())} "
               f"--run-fn {shlex.quote(self.run_fn_ref)} "
               f"--work-dir {shlex.quote(self.remote_work_dir)}")
        rc, out = _ssh(self.key_path, host.ssh_host, host.ssh_port, cmd,
                       timeout=self.run_timeout)
        for line in out.splitlines():
            if line.startswith(RESULT_PREFIX):
                return json.loads(line[len(RESULT_PREFIX):])
        return {"run": config.run_name(), "result": None, "skipped": False,
                "error": f"rc={rc}: {out[-200:]}"}

    def _sync_back(self, host: RentedHost) -> None:
        """Best-effort: pull the run artifacts (checkpoints/events/triggers)."""
        os.makedirs(self.local_work_dir, exist_ok=True)
        _scp_down(self.key_path, host.ssh_host, host.ssh_port,
                  self.remote_work_dir, self.local_work_dir)

    # -- the campaign --------------------------------------------------------
    def run(self, configs: Iterable[RunConfig], *, admission=None) -> list[dict]:
        """Rent a host (failing over bad ones), run every config on it, sync the
        artifacts back, and tear down. Returns the per-config result records.

        Teardown is the Provider's leak-proof `rent()` invariant -- it fires on
        every exit, including the failover `continue` and any exception here.
        """
        configs = list(configs)
        if not configs:
            return []
        offers = self.provider.offers(self.host_spec)
        if not offers:
            raise RuntimeError(
                f"{self.provider.name}: no offers match {self.host_spec}")
        last_err: Exception | None = None
        for offer in offers:
            try:
                with self.provider.rent(offer, self.launch,
                                        timeout_s=self.rent_timeout) as host:
                    self._wait_engine_ready(host)        # may raise -> failover
                    results = [self._run_config(host, c) for c in configs]
                    self._sync_back(host)
                    return results
            except (HostProbeFailed, TimeoutError) as e:
                last_err = e                              # bad host -> next offer
                continue
        raise RuntimeError(
            f"{self.provider.name}: all {len(offers)} offers failed to run; "
            f"last error: {last_err}")

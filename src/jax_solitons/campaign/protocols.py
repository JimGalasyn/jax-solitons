"""The campaign boundary: four physics-agnostic protocols (design: CAMPAIGN.md).

A campaign is 10^4-10^6 registered, restartable runs over a rented fleet. A
2026-06-12 literature sweep found NO library covers the full contract: the
executor layer (spot-fleet recovery) is solved by SkyPilot/dstack, but the
provenance/restart/event-record/admission contract is unserved -- and
host-probing admission (P9) exists nowhere. So this package owns A/B/C/E and
delegates D to a pluggable Executor.

Contract letters map to DESIGN.md principles:
  A  RunRegistry   config-hashed registry + idempotent skip          (P4)
  B  RunRegistry   full-integrator-state checkpoints, exact restart  (P4)
  C  EventSink     event-records-not-fields + triggered capture      (P6, P7)
  D  Executor      spot-fleet fan-out + preemption recovery          (adopted)
  E  Admission     probe-or-bail on flaky marketplace hosts          (P9)

The ONLY soliton-specific thing crossing this boundary is `RunFn`: the physics
is injected as a callable. Nothing under `campaign/` imports a model, stepper,
or jax -- that discipline is what keeps the layer extractable into a standalone
package at rule-of-three.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any, Callable, Protocol, runtime_checkable

from jax_solitons.runs import RunConfig

# A run's full integrator state: an opaque flat dict of arrays (field, velocity,
# optimizer moments, RNG key, ...). The campaign layer never inspects it; only
# the injected physics (RunFn) understands its keys.
State = dict[str, Any]


@dataclasses.dataclass(frozen=True)
class RunHandle:
    """Identity of one run: its config, its hashed directory, its name.

    Returned by `RunRegistry.register`; threaded through every capability so
    the registry, sink, and executor all agree on which run they touch.
    """

    config: RunConfig
    dir: Path
    name: str


@dataclasses.dataclass(frozen=True)
class HostReport:
    """What an `Admission` probe measured about a candidate host (P9).

    Hosts, networks, and devices lie; this is the measured truth a host is
    admitted or rejected on. Fields are deliberately concrete -- the standing
    case study is a 0.996-reliability host with ZERO outbound bandwidth.
    """

    has_gpu: bool
    device_name: str
    free_mem_gb: float
    outbound_mbps: float
    notes: str = ""


@dataclasses.dataclass
class RunContext:
    """The four orchestration capabilities handed to the physics, and nothing
    else. This dataclass IS the seam: `RunFn` receives exactly this.

      resume      prior full state to continue from, or None for a fresh run (B)
      checkpoint  persist full integrator state at a step (B)
      emit        stream one small event record -- a ledger row, a census (C)
      trigger     capture full fields on a flagged event, after a quench (C/P7)
    """

    resume: State | None
    checkpoint: Callable[[State, int], None]
    emit: Callable[[dict], None]
    trigger: Callable[[State, str], None]


# The single soliton-specific injection. Returns a small result record (the
# run's summary ledger), never raw fields -- those go through ctx.trigger (P6).
RunFn = Callable[[RunConfig, RunContext], dict]


@runtime_checkable
class RunRegistry(Protocol):
    """A (config-hashed) registry of runs with full-state restart (A + B, P4).

    A result that cannot name its config hash does not exist. Restart is
    bit-identical at fixed dtype/devices because checkpoints carry FULL
    integrator state, not just artifacts.
    """

    def register(self, config: RunConfig) -> RunHandle:
        """Create/locate the config-hashed run dir; append one manifest line."""
        ...

    def is_complete(self, handle: RunHandle) -> bool:
        """True if this run already finished -- the idempotent-skip that makes
        spot preemption free (re-submitting a done run is a no-op)."""
        ...

    def load(self, handle: RunHandle) -> tuple[State, int] | None:
        """Latest full-state checkpoint as (state, step), or None if none yet."""
        ...

    def save(self, handle: RunHandle, state: State, step: int) -> None:
        """Write a full-state checkpoint (field + velocity/optimizer + RNG)."""
        ...

    def finish(self, handle: RunHandle, result: dict) -> None:
        """Mark the run complete and record its summary result."""
        ...


@runtime_checkable
class EventSink(Protocol):
    """Streaming event records, with full fields only on triggered events (C).

    At campaign scale the product of a run is a small record, not its fields
    (P6). Cheap classifiers stream via `emit` on every event; expensive
    full-state capture fires through `trigger`, and ONLY after a quench, since
    descent cannot create topology -- relax-then-ID is faithful, in-bath is not
    (P7).
    """

    def emit(self, handle: RunHandle, record: dict) -> None:
        """Append one small event record (a charge/energy ledger row, a census)."""
        ...

    def trigger(self, handle: RunHandle, state: State, reason: str) -> None:
        """Capture full fields for a flagged event (the rare, kept snapshot)."""
        ...

    def close(self, handle: RunHandle) -> None:
        """Flush this run's stream."""
        ...


@runtime_checkable
class Admission(Protocol):
    """Probe-or-bail admission control for flaky fleets (E, P9).

    Served by no existing orchestrator: they assume reliable hosts. Anything
    that touches infrastructure probes first, writes what it measured, and
    bails early -- never "runs anyway" on unverified capacity.
    """

    def probe(self) -> HostReport:
        """Measure this host's real compute + network capacity."""
        ...

    def guard(self) -> HostReport:
        """Probe and raise `AdmissionError` if the host fails the bar. Called on
        each fleet node before it pulls work. Returns the report on success."""
        ...


@runtime_checkable
class Executor(Protocol):
    """Fan tasks out over a fleet and recover from preemption (D -- adopted).

    The crowded, solved layer: adopt SkyPilot/dstack, never rebuild it. The
    executor's only campaign-specific duty is to guard each worker via
    `Admission` before it runs, and to lean on `RunRegistry.is_complete` so a
    preempted task is simply re-submitted and skips its finished work.
    """

    def run(self, tasks: list[Callable[[], None]], admission: Admission) -> None:
        """Execute every task thunk across the fleet, guarding each worker."""
        ...


class AdmissionError(RuntimeError):
    """A host failed its capacity probe and must not run work (P9)."""

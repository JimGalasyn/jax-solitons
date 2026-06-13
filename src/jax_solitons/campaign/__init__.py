"""The campaign boundary (design note: CAMPAIGN.md).

Physics-agnostic orchestration for 10^4-10^6 registered, restartable runs over
a rented fleet. Four protocols -- RunRegistry (A/B), EventSink (C), Admission
(E), Executor (D) -- plus a physics-blind `run_campaign` driver. The only
soliton-specific value crossing the boundary is the injected `RunFn`.

A 2026-06-12 literature sweep confirmed no library covers the A/B/C/E contract
(host-probing admission, E, exists nowhere); the executor layer D is adopted
from SkyPilot/dstack, never rebuilt. This module stays internal until
rule-of-three, then lifts out as a standalone package.
"""

from jax_solitons.campaign.driver import run_campaign
from jax_solitons.campaign.protocols import (
    Admission,
    AdmissionError,
    EventSink,
    Executor,
    HostReport,
    RunContext,
    RunFn,
    RunHandle,
    RunRegistry,
    State,
)
from jax_solitons.campaign.reference import (
    FileRunRegistry,
    JsonlEventSink,
    LocalExecutor,
    ProbeAdmission,
    SkyPilotExecutor,
)

__all__ = [
    # protocols (the contract)
    "RunRegistry", "EventSink", "Admission", "Executor",
    "RunContext", "RunFn", "RunHandle", "HostReport", "State", "AdmissionError",
    # reference implementations
    "FileRunRegistry", "JsonlEventSink", "ProbeAdmission",
    "LocalExecutor", "SkyPilotExecutor",
    # driver
    "run_campaign",
]

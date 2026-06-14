"""The campaign boundary (design note: CAMPAIGN.md).

Physics-agnostic orchestration for 10^4-10^6 registered, restartable runs over
a rented fleet. Five protocols -- RunRegistry (A/B), EventSink (C), Admission
(E), Executor (D), Provider (F) -- plus a physics-blind `run_campaign` driver.
The only soliton-specific value crossing the boundary is the injected `RunFn`.

A 2026-06-12 literature sweep confirmed no library covers the A/B/C/E contract
(host-probing admission, E, exists nowhere); the executor layer D is adopted
from SkyPilot/dstack, never rebuilt. F (the cloud broker) is the second
build-thin seam -- the marketplaces SkyPilot's providers can't drive plug in
behind one Protocol; `VastProvider` is the reference adapter. This module stays
internal until rule-of-three, then lifts out as a standalone package.
"""

from jax_solitons.campaign.driver import execute_config, run_campaign
from jax_solitons.campaign.protocols import (
    Admission,
    AdmissionError,
    EventSink,
    Executor,
    HostProbeFailed,
    HostReport,
    HostSpec,
    LaunchSpec,
    Offer,
    Provider,
    RentedHost,
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
from jax_solitons.campaign.multi import CampaignReport, run_multi, split_configs
from jax_solitons.campaign.provider_exec import ProviderExecutor
from jax_solitons.campaign.remote import load_run_fn, run_one
from jax_solitons.campaign.runpod import RunPodProvider
from jax_solitons.campaign.vast import VastLedger, VastProvider

# ModalExecutor is intentionally NOT imported here: `modal` is an optional
# dependency, so `import jax_solitons.campaign` must not require it. Use
# `from jax_solitons.campaign.modal_exec import ModalExecutor` when modal is
# installed.

__all__ = [
    # protocols (the contract)
    "RunRegistry", "EventSink", "Admission", "Executor", "Provider",
    "RunContext", "RunFn", "RunHandle", "HostReport", "State", "AdmissionError",
    # F (cloud broker) contract types
    "HostSpec", "LaunchSpec", "Offer", "RentedHost", "HostProbeFailed",
    # reference implementations
    "FileRunRegistry", "JsonlEventSink", "ProbeAdmission",
    "LocalExecutor", "SkyPilotExecutor", "ProviderExecutor",
    "VastProvider", "VastLedger", "RunPodProvider",
    # driver + remote-worker core
    "run_campaign", "execute_config", "run_one", "load_run_fn",
    # multi-provider partition-and-merge
    "run_multi", "split_configs", "CampaignReport",
]

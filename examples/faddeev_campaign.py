"""Run a small Faddeev hopfion campaign through the registered-run contract.

A runnable example (and a smoke of `jax_solitons.campaign`): relax two
(n, m) = (1, 1) hopfions in the CP^1 spinor frame, stream each run's
charge/energy ledger, and trace + capture each relaxed core. Writes a
config-hashed run directory per run plus a MANIFEST.jsonl registry under
./_campaign_out (or the path given as argv[1]).

    JAX_ENABLE_X64=1 python examples/faddeev_campaign.py

Swap LocalExecutor for SkyPilotExecutor (and drop require_gpu=False) to fan the
same campaign across a spot fleet — the RunFn and records are unchanged.
"""

from __future__ import annotations

import sys

from jax_solitons.campaign import (
    FileRunRegistry,
    JsonlEventSink,
    LocalExecutor,
    ProbeAdmission,
    run_campaign,
)
from jax_solitons.runfns import faddeev_relax_then_id
from jax_solitons.runs import RunConfig


def main(out: str = "_campaign_out") -> None:
    registry = FileRunRegistry(out)
    sink = JsonlEventSink()
    admission = ProbeAdmission(require_gpu=False)   # set True (default) on a fleet
    executor = LocalExecutor()                      # -> SkyPilotExecutor() at scale

    configs = [
        RunConfig(model="faddeev_cp1", N=24, L=16.0, dtype="float64", steps=4000,
                  params={"R": 3.5, "n": 1, "m": 1, "segments": 8})
        for _ in range(2)
    ]
    run_campaign(configs, faddeev_relax_then_id, registry=registry, sink=sink,
                 admission=admission, executor=executor)
    print(f"wrote {len(configs)} runs + MANIFEST.jsonl under {out}/")


if __name__ == "__main__":
    main(*sys.argv[1:])

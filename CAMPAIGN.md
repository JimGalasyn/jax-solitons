# Campaigns

The campaign layer was **extracted to [run-farm](https://github.com/JimGalasyn/run-farm)**
in 2026-07, at rule-of-three. This document set the gate: keep the layer internal
until *either* a second consumer appears *or* the A/B/C/E API stabilizes through one
real campaign. The **second** disjunct was met — the Provider (F) seam absorbed
three backends (Vast, RunPod, Modal) with zero Protocol changes, and C/E shipped and
were CI-gated — with jax-morpho's evolution loop as the credible near-future second
consumer the extraction unblocks. The full A/B/C/D/E/F contract, the 2026-06
literature sweep, and the P9/P10 case studies now live in run-farm's
[`docs/DESIGN.md`](https://github.com/JimGalasyn/run-farm/blob/main/docs/DESIGN.md).

## What jax-solitons keeps

| | |
|---|---|
| `jax_solitons.runs.RunConfig` | This engine's config shape. **Byte-stable** — its `to_json` names every run directory in every `campaign_out` ledger ever written, so it did NOT move. It structurally satisfies `run_farm.RunConfig` (pinned by `tests/test_protocol_conformance.py`). |
| `jax_solitons.runfns.faddeev_relax_then_id` | The `RunFn` — the one seam the physics crosses (`Callable[[RunConfig, RunContext], dict]`). |
| `jax_solitons.farm_config.soliton_leg_to_config` | The `FarmCampaign` `config_factory` for this engine's config shape (its `reserved` set pinned byte-identical by `tests/test_farm_config_goldens.py`). |
| `tests/test_campaign.py` | The downstream integration test: a real engine still closes the A/B/C contract across the package boundary, bit-identical resume included. (It caught a real run-farm bug during extraction — `FileRunRegistry.load` rebuilding with the wrong config type, fixed in run-farm 0.1.1.) |

## Running one

```python
from run_farm import (FileRunRegistry, JsonlEventSink, LocalExecutor,
                      ProbeAdmission, run_campaign)
from jax_solitons.runfns import faddeev_relax_then_id
from jax_solitons.runs import RunConfig

cfg = RunConfig(model="faddeev_cp1", N=24, L=16.0, dtype="float64", steps=4000,
                params={"R": 3.5, "n": 1, "m": 1, "segments": 8})
run_campaign([cfg], faddeev_relax_then_id,
             registry=FileRunRegistry("out"), sink=JsonlEventSink(),
             admission=ProbeAdmission(require_gpu=False), executor=LocalExecutor())
```

See `examples/faddeev_campaign.py`. To fan the same campaign across a rented spot
fleet, swap `LocalExecutor` for a `run_farm` `ProviderExecutor`/`FleetExecutor` over
`VastProvider` — the `RunFn` and the records are unchanged.

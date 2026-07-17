"""The `FarmCampaign` config factory for this engine's config shape.

When the campaign layer was extracted to run-farm, `campaign.farm.leg_to_config`
split in two: the physics-agnostic params construction (copy semantics, unspoofable
`gtag`/`required_shas`) became `run_farm.farm.leg_params`, and the soliton-shaped
field assignment -- which leg keys become `RunConfig(model=, N=, L=, seed=)` -- stayed
here as the engine's own `config_factory`.

Pass `soliton_leg_to_config` to `FarmCampaign(config_factory=...)` to plan a farm
over `jax_solitons.RunConfig` instead of run-farm's default `SimpleRunConfig`.
"""

from __future__ import annotations

from run_farm.farm import leg_params

from jax_solitons.runs import RunConfig


def soliton_leg_to_config(leg: dict, gtag: str, required_shas: dict) -> RunConfig:
    """A farm leg -> a `jax_solitons.RunConfig` (the FarmCampaign config_factory).

    ⚠ `reserved=("model", "N", "seed")` is BYTE-CRITICAL. Its union with
    `leg_params`'s base set -- ``("rid","cfg","plan","gtag","required_shas")`` -- must
    equal the pre-extraction `campaign.farm.leg_to_config` reserved tuple exactly:
    ``("rid","cfg","plan","N","seed","model","gtag","required_shas")``. Any drift
    changes `params`, which changes `config_hash`, which renames every run in the
    campaign. Pinned byte-for-byte by tests/test_farm_config_goldens.py.

    `N` defaults to ``round(L / dx)`` from the leg's cfg, matching the original.
    """
    params = leg_params(leg, gtag, required_shas, reserved=("model", "N", "seed"))
    cfg = params["cfg"]                       # the same copy leg_params already made
    return RunConfig(
        model=leg.get("model", "farm"),
        N=int(leg.get("N", round(cfg["L"] / cfg["dx"]))),
        L=float(cfg["L"]), seed=int(leg.get("seed", 0)),
        params=params)

"""Frozen leg->config translation: the other half of the extraction's identity pin.

`campaign.farm.leg_to_config` decides which leg keys become top-level RunConfig
fields and which ride in `params`. That split is expressed as one 8-element
`reserved` tuple (farm.py:115) -- and `params` is serialized into `config_hash`, so
getting the tuple wrong by one key silently renames every run in the campaign. See
tests/test_run_config_goldens.py for why a silent rename is the expensive failure.

The extraction splits this function in two: the physics-agnostic params
construction moves to `run_farm.farm.leg_params(..., reserved=...)`, and the
soliton-shaped field assignment stays here as `jax_solitons.farm_config
.soliton_leg_to_config`. The two `reserved` sets must UNION back to today's exact
8 keys. These literals were captured by running the PRE-EXTRACTION
`campaign.farm.leg_to_config`; they are the evidence that the split preserved it.

Do not regenerate these values from the post-split code -- that would pin the
split's own output and prove nothing.
"""
from __future__ import annotations

import pytest

# Imported indirectly so this file survives the extraction by changing ONE line:
#   from jax_solitons.farm_config import soliton_leg_to_config as leg_to_config
#     ->  from jax_solitons.farm_config import soliton_leg_to_config as leg_to_config
from jax_solitons.farm_config import soliton_leg_to_config as leg_to_config

# Exercises every branch at once: explicit model/N/seed, an unreserved passthrough
# key, and campaign-authoritative keys the leg tries to spoof.
LEG_EXPLICIT = {
    "rid": "eps-0.30-seed7",
    "cfg": {"L": 18.0, "dx": 0.5625, "eps": 0.30},
    "plan": ["relax", "kick"],
    "model": "faddeev_cp1", "N": 32, "seed": 7,
    "extra": "rides in params",
    "gtag": "SPOOFED", "required_shas": {"x": "SPOOFED"},
}
# The defaults branch: model -> "farm", N -> round(L/dx), seed -> 0.
LEG_DEFAULTS = {"rid": "r2", "cfg": {"L": 18.0, "dx": 0.5625}}


def test_explicit_leg_params_are_frozen():
    c = leg_to_config(LEG_EXPLICIT, "gtag-v3", {"engine": "abc123"})
    assert c.params == {
        "rid": "eps-0.30-seed7",
        "cfg": {"L": 18.0, "dx": 0.5625, "eps": 0.30},
        "plan": ["relax", "kick"],
        "extra": "rides in params",          # unreserved -> passthrough
        "gtag": "gtag-v3",                   # campaign-authoritative, NOT "SPOOFED"
        "required_shas": {"engine": "abc123"},
    }
    # model/N/seed are consumed as top-level fields and must NOT also be in params
    assert not {"model", "N", "seed"} & set(c.params)


def test_explicit_leg_fields_and_identity_are_frozen():
    c = leg_to_config(LEG_EXPLICIT, "gtag-v3", {"engine": "abc123"})
    assert (c.model, c.N, c.L, c.seed) == ("faddeev_cp1", 32, 18.0, 7)
    assert c.config_hash() == "99bb760bbbc1"
    assert c.run_name() == "faddeev_cp1_N32_99bb760bbbc1"


def test_defaults_leg_is_frozen():
    """N derives from round(L/dx) when the leg doesn't name it: 18.0/0.5625 = 32."""
    c = leg_to_config(LEG_DEFAULTS, "g", {})
    assert c.params == {"rid": "r2", "cfg": {"L": 18.0, "dx": 0.5625},
                        "plan": [], "gtag": "g", "required_shas": {}}
    assert (c.model, c.N, c.L, c.seed) == ("farm", 32, 18.0, 0)
    assert c.config_hash() == "3fd574b430a6"
    assert c.run_name() == "farm_N32_3fd574b430a6"


def test_campaign_authoritative_keys_are_unspoofable():
    """A leg carrying gtag/required_shas must not override the campaign's
    attestation identity -- otherwise a leg could forge its own provenance."""
    c = leg_to_config(LEG_EXPLICIT, "gtag-v3", {"engine": "abc123"})
    assert c.params["gtag"] == "gtag-v3"
    assert c.params["required_shas"] == {"engine": "abc123"}


@pytest.mark.parametrize("mutate", [
    pytest.param(lambda leg: leg["cfg"].__setitem__("L", 99.0), id="cfg"),
    pytest.param(lambda leg: leg["plan"].append("extra-stage"), id="plan"),
])
def test_config_does_not_alias_the_leg(mutate):
    """cfg/plan are COPIED: mutating the leg after the call must not change the
    config's identity. Legs commonly share one cfg object, so aliasing would let
    one leg silently rename another's run directory."""
    leg = {"rid": "r", "cfg": {"L": 18.0, "dx": 0.5625}, "plan": ["relax"]}
    c = leg_to_config(leg, "g", {})
    before = c.config_hash()
    mutate(leg)
    assert c.config_hash() == before


def test_required_shas_does_not_alias_the_campaign_baseline():
    """Per-leg COPY: mutating one config's required_shas must not reach into the
    campaign's verification baseline or any sibling leg."""
    shas = {"engine": "abc123"}
    c = leg_to_config({"rid": "r", "cfg": {"L": 18.0, "dx": 0.5625}}, "g", shas)
    c.params["required_shas"]["engine"] = "TAMPERED"
    assert shas == {"engine": "abc123"}

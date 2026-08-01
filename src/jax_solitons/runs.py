"""Restartable, registered runs (design requirement R4).

RunConfig is the single source of truth for a run: serialized into every
output, hashed into the run directory name. It **structurally satisfies the
`run_farm.RunConfig` Protocol** (pinned by tests/test_protocol_conformance.py),
so the extracted campaign layer drives jax-solitons runs without importing
anything soliton-specific. Its `to_json` bytes are the permanent names of every
run directory in every `campaign_out` ledger ever written -- this class does not
change; its identity is frozen (tests/test_run_config_goldens.py).

The full-state checkpoint helpers (`save_checkpoint`/`load_checkpoint`/`run_dir`)
were physics-agnostic and moved WHOLESALE to `run_farm.config` when the campaign
layer was extracted (2026-07). They are re-exported here so existing call sites
keep working; `load_checkpoint` is bound to this engine's `RunConfig` so it
rebuilds the concrete type rather than run-farm's default.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Any

# Moved to run-farm at extraction; re-exported so `jax_solitons.runs.<helper>`
# still resolves. Orbax replaces this checkpoint layer when sharded multi-device
# arrays land -- that roadmap is now run-farm's.
from run_farm.config import run_dir, save_checkpoint
from run_farm.config import load_checkpoint as _load_checkpoint


@dataclasses.dataclass(frozen=True)
class RunConfig:
    """Declarative description of one run.

    `params` carries model/stepper specifics; top-level fields are the
    invariants every run has. Replaces per-script argparse.
    """

    model: str
    N: int
    L: float
    dtype: str = "float32"
    steps: int = 0
    dt: float = 0.0
    seed: int = 0
    params: dict[str, Any] = dataclasses.field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(dataclasses.asdict(self), sort_keys=True)

    @classmethod
    def from_json(cls, s: str) -> "RunConfig":
        return cls(**json.loads(s))

    def config_hash(self, n: int = 12) -> str:
        """Stable short hash for run-directory naming."""
        return hashlib.sha256(self.to_json().encode()).hexdigest()[:n]

    def run_name(self) -> str:
        return f"{self.model}_N{self.N}_{self.config_hash()}"


def load_checkpoint(path) -> tuple[dict, RunConfig, int]:
    """Read a checkpoint back: (state dict, RunConfig, step).

    `run_farm.config.load_checkpoint` bound to this engine's config type, so a
    restored checkpoint comes back as a `jax_solitons.RunConfig`, not run-farm's
    default `SimpleRunConfig`.
    """
    return _load_checkpoint(path, config_class=RunConfig)


# `save_checkpoint` and `run_dir` are re-exported unchanged from run_farm.config
# (imported at module top); they never read a soliton-specific field.
__all__ = ["RunConfig", "save_checkpoint", "load_checkpoint", "run_dir"]

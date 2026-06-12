"""Restartable, registered runs (design requirement R4).

RunConfig is the single source of truth for a run: serialized into every
output, hashed into the run directory name. Checkpoints carry the FULL
integrator state (field + velocity/optimizer state + RNG key), so a
restarted run reproduces the uninterrupted trajectory bit-identically at
fixed dtype and device count.

Backend note: checkpoints are .npz with the config embedded as JSON --
simple, dependency-free, and deterministic. Orbax replaces this layer when
sharded multi-device arrays land (it adds async + sharding-aware layout,
not different semantics).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np


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


def save_checkpoint(path, state: dict, config: RunConfig, step: int) -> None:
    """Write a full-state checkpoint: a flat dict of arrays (field, velocity,
    optimizer moments, RNG key, ...) + the RunConfig + the step counter."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {f"state__{k}": np.asarray(v) for k, v in state.items()}
    np.savez_compressed(path, __config__=config.to_json(), __step__=step,
                        **arrays)


def load_checkpoint(path) -> tuple[dict, RunConfig, int]:
    """Read a checkpoint back: (state dict of jnp arrays, RunConfig, step)."""
    with np.load(path, allow_pickle=False) as d:
        config = RunConfig.from_json(str(d["__config__"]))
        step = int(d["__step__"])
        state = {k[len("state__"):]: jnp.asarray(d[k])
                 for k in d.files if k.startswith("state__")}
    return state, config, step


def run_dir(base, config: RunConfig) -> Path:
    """Config-hashed run directory with the config serialized into it, and a
    one-line entry appended to the base manifest (the run registry)."""
    base = Path(base)
    d = base / config.run_name()
    d.mkdir(parents=True, exist_ok=True)
    cfg_file = d / "config.json"
    if not cfg_file.exists():
        cfg_file.write_text(config.to_json() + "\n")
        with (base / "MANIFEST.jsonl").open("a") as mf:
            mf.write(json.dumps({"run": config.run_name(),
                                 "config": json.loads(config.to_json())})
                     + "\n")
    return d

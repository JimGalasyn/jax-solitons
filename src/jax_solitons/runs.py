"""Restartable, registered runs (design requirement R4).

RunConfig is the single source of truth for a run: serialized into every
output, hashed into the run directory name. Full-state checkpointing
(field + velocity/optimizer state + RNG key) via orbax lands with the
stepper port.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Any


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

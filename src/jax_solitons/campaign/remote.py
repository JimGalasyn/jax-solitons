"""Shared core for the remote executors (Modal, Provider-over-SSH).

A closure can't cross a network, so the one physics-specific thing a remote
worker needs is shipped by NAME: a `RunFn` import reference like
``jax_solitons.runfns:faddeev_relax_then_id``. The worker imports it, rebuilds a
file-backed registry/sink at a shared-storage path, and runs the same
`execute_config` unit the in-process driver runs. Configs travel as JSON
(`RunConfig.to_json`), so nothing but stdlib + the already-installed engine is
needed on the far side.
"""

from __future__ import annotations

import importlib

from jax_solitons.campaign.driver import execute_config
from jax_solitons.campaign.reference import FileRunRegistry, JsonlEventSink
from jax_solitons.runs import RunConfig

RunFnRef = str  # "module:function"


def load_run_fn(ref: RunFnRef):
    """Import a `RunFn` from a ``'module:function'`` reference.

    The seam's one physics injection, made shippable: the local driver passes a
    callable, a remote worker passes this string and re-imports it on the box.
    """
    if ":" not in ref:
        raise ValueError(
            f"run_fn ref must be 'module:function', got {ref!r}")
    module_name, fn_name = ref.split(":", 1)
    fn = getattr(importlib.import_module(module_name), fn_name)
    if not callable(fn):
        raise TypeError(f"{ref} is not callable")
    return fn


def run_one(config_json: str, run_fn_ref: RunFnRef, work_dir: str) -> dict:
    """Execute ONE config on this machine against a file registry at `work_dir`.

    The unit both remote workers run: deserialize the config, import the RunFn,
    build a `FileRunRegistry`/`JsonlEventSink` rooted at `work_dir` (a Modal
    Volume mount, or a rented box's disk synced back), and run `execute_config`.
    Returns ``{"run": <name>, "result": <record or None>, "skipped": bool}`` --
    small and JSON-safe, so it rides a Modal return value or an SSH stdout line.
    Full artifacts (checkpoints, events, triggered captures) stay in `work_dir`.
    """
    config = RunConfig.from_json(config_json)
    run_fn = load_run_fn(run_fn_ref)
    registry = FileRunRegistry(work_dir)
    sink = JsonlEventSink()
    result = execute_config(config, run_fn, registry=registry, sink=sink)
    return {"run": config.run_name(), "result": result, "skipped": result is None}

"""The campaign driver: wire the four protocols around an injected `RunFn`.

`run_campaign` is the whole orchestration loop, and it is physics-blind -- the
only soliton-specific value is `run_fn`. Each config becomes a task thunk that
the `Executor` runs on the fleet; the thunk skips finished runs (idempotent
preemption recovery), resumes from the latest full-state checkpoint, and hands
the physics a `RunContext` wired to the registry and sink.
"""

from __future__ import annotations

from typing import Iterable

from jax_solitons.campaign.protocols import (
    Admission,
    EventSink,
    RunConfig,
    RunContext,
    RunFn,
    RunRegistry,
)


def run_campaign(
    configs: Iterable[RunConfig],
    run_fn: RunFn,
    *,
    registry: RunRegistry,
    sink: EventSink,
    admission: Admission,
    executor,
) -> None:
    """Run every config through `run_fn` over `executor`, with restart + records.

    Per run: register (A) -> skip if complete (D recovery) -> resume from full
    state or start fresh (B) -> run physics with a wired RunContext (C) ->
    finish. Admission (E) is enforced by the executor on each worker.
    """
    handles = [registry.register(c) for c in configs]

    def task_for(handle):
        def task():
            if registry.is_complete(handle):
                return                                    # preemption no-op (P4/D)
            resume = registry.load(handle)
            ctx = RunContext(
                resume=None if resume is None else resume[0],
                checkpoint=lambda state, step: registry.save(handle, state, step),
                emit=lambda record: sink.emit(handle, record),
                trigger=lambda state, reason: sink.trigger(handle, state, reason),
            )
            result = run_fn(handle.config, ctx)
            registry.finish(handle, result)
            sink.close(handle)
        return task

    executor.run([task_for(h) for h in handles], admission)

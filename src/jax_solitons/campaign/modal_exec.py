"""ModalExecutor: serverless campaign fan-out -- the Executor (D) seam for Modal.

Modal is NOT a `Provider` (F): there is no host to rent and no meter to leak.
You hand it a containerized function and it provisions, scales, per-second
bills, retries, and tears down. So Modal plugs in here, at the executor seam,
and the lifecycle worries the Provider contract carries simply don't exist.

The fan-out is `Function.map` over the configs; each call runs the shared
`run_one` unit (register/skip/resume/run/finish, identical to the in-process
driver) against a **Modal Volume** mounted as the registry root, so checkpoints,
event streams, and triggered captures persist and `is_complete`/resume work
across runs. Small result records ride the `.map` return values; the heavy
artifacts stay on the Volume (fetch with `modal volume get`, or mount it).

`modal` is an optional dependency -- importing this module requires it, but
`import jax_solitons.campaign` does not.

    from jax_solitons.campaign.modal_exec import ModalExecutor
    ex = ModalExecutor("jax_solitons.runfns:faddeev_relax_then_id", gpu="A10G")
    results = ex.run(configs)            # configs: Iterable[RunConfig]
"""

from __future__ import annotations

import sys
from collections.abc import Iterable

import modal

from jax_solitons.campaign.remote import RunFnRef, run_one
from jax_solitons.runs import RunConfig

_MOUNT = "/campaign"
DEFAULT_VOLUME = "jax-solitons-campaign"


def default_image(ref: str = "main") -> "modal.Image":
    """A debian image with git + jax (CUDA 12) + the jax-solitons engine.

    `ref` pins the git ref installed (branch/tag/commit). It MUST contain the
    code the worker runs -- `campaign.remote.run_one` and the `run_fn_ref` -- so
    while a feature branch is unmerged, install from it (e.g.
    ``default_image("provider-seam-f")``); a release pins a tag. (`debian_slim`
    has no git, which the VCS install needs.) Pass a full `image` to ModalExecutor
    to add deps or install a private fork instead.
    """
    py = f"{sys.version_info.major}.{sys.version_info.minor}"
    return (
        modal.Image.debian_slim(python_version=py)
        .apt_install("git")
        .pip_install("jax[cuda12]",
                     f"git+https://github.com/JimGalasyn/jax-solitons@{ref}")
    )


def _modal_worker(config_json: str, run_fn_ref: str, work_dir: str,
                  volume_name: str) -> dict:
    """The function Modal runs in-container (referenced by name, not pickled, so
    the image's own jax-solitons provides it). Reload the Volume to see prior
    checkpoints (resume/skip), run the shared unit, commit to persist artifacts.
    """
    vol = modal.Volume.from_name(volume_name)
    try:
        vol.reload()
    except Exception:
        pass
    out = run_one(config_json, run_fn_ref, work_dir)
    try:
        vol.commit()                # persist checkpoints/events/triggers
    except Exception as e:          # results still return; durability best-effort
        out["volume_commit"] = f"failed: {e}"
    return out


class ModalExecutor:
    """Run a campaign across Modal GPUs via `Function.map`.

    `run_fn_ref` is the ``'module:function'`` RunFn the workers import (the one
    physics injection; see `campaign.remote`). `image` must have jax-solitons
    (and the package holding `run_fn_ref`) installed; defaults to `default_image`.
    """

    name = "modal"

    def __init__(self, run_fn_ref: RunFnRef, *, image: "modal.Image | None" = None,
                 gpu: str = "A10G", app_name: str = "jax-solitons-campaign",
                 volume_name: str = DEFAULT_VOLUME, work_subdir: str = "runs",
                 timeout: int = 3600, retries: int = 2):
        self.run_fn_ref = run_fn_ref
        self.gpu = gpu
        self.volume_name = volume_name
        self.work_dir = f"{_MOUNT}/{work_subdir}"
        self.volume = modal.Volume.from_name(volume_name, create_if_missing=True)
        self.app = modal.App(app_name)
        # Module-level worker referenced by name (not cloudpickled): the image's
        # own jax-solitons provides it, so there's no module-availability or
        # Python-version-match fragility.
        self._worker = self.app.function(
            image=image or default_image(), gpu=gpu,
            volumes={_MOUNT: self.volume}, timeout=timeout,
            retries=retries)(_modal_worker)

    def run(self, configs: Iterable[RunConfig], *, admission=None) -> list[dict]:
        """Fan `configs` out across Modal GPUs; return the small result records.

        `admission` is accepted for parity with the in-process executor but is a
        no-op: Modal manages reliable hosts, so the P9 probe-or-bail (E) -- whose
        whole point is flaky marketplace hosts -- does not apply here.
        """
        cfg_jsons = [c.to_json() for c in configs]
        if not cfg_jsons:
            return []
        with self.app.run():
            return list(self._worker.map(
                cfg_jsons, kwargs={"run_fn_ref": self.run_fn_ref,
                                   "work_dir": self.work_dir,
                                   "volume_name": self.volume_name}))

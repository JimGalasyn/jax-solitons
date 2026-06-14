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


def default_image() -> "modal.Image":
    """A debian image with git + jax (CUDA 12) + the public jax-solitons engine.

    The same recipe validated live on 2026-06-14 (note: `debian_slim` has no git,
    which the VCS install needs). Pass your own `image` to pin a commit, add deps,
    or install a private fork.

    The image Python matches the LOCAL interpreter: a `serialized=True` worker is
    cloudpickled here and unpickled on the box, so the two Pythons must agree.
    """
    py = f"{sys.version_info.major}.{sys.version_info.minor}"
    return (
        modal.Image.debian_slim(python_version=py)
        .apt_install("git")
        .pip_install("jax[cuda12]",
                     "git+https://github.com/JimGalasyn/jax-solitons")
    )


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
        self.work_dir = f"{_MOUNT}/{work_subdir}"
        self.volume = modal.Volume.from_name(volume_name, create_if_missing=True)
        self.app = modal.App(app_name)

        work_dir = self.work_dir            # capture plain strings (serializable)
        vol_name = volume_name

        def _worker(config_json: str, run_fn_ref: str) -> dict:
            # Runs inside the Modal container; the Volume is mounted at _MOUNT.
            vol = modal.Volume.from_name(vol_name)
            try:
                vol.reload()                # see prior checkpoints (resume/skip)
            except Exception:
                pass
            out = run_one(config_json, run_fn_ref, work_dir)
            try:
                vol.commit()                # persist checkpoints/events/triggers
            except Exception as e:
                out["volume_commit"] = f"failed: {e}"   # results still return
            return out

        self._worker = self.app.function(
            image=image or default_image(), gpu=gpu,
            volumes={_MOUNT: self.volume}, timeout=timeout,
            retries=retries, serialized=True)(_worker)

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
                cfg_jsons, kwargs={"run_fn_ref": self.run_fn_ref}))

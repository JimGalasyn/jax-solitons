"""CLI worker: run ONE campaign config on this machine.

The command a `ProviderExecutor` invokes over SSH on a rented box:

    python -m jax_solitons.campaign.worker \
        --config-json '<RunConfig JSON>' --run-fn pkg.mod:fn --work-dir runs

It prints exactly one ``__RESULT__ {json}`` line (the small result record) for
the executor to parse out of stdout; the full artifacts land under --work-dir,
which the executor syncs back. The Modal executor calls `run_one` directly
rather than this CLI -- same unit, no subprocess.
"""

from __future__ import annotations

import argparse
import json

from jax_solitons.campaign.remote import run_one

RESULT_PREFIX = "__RESULT__ "


def main() -> None:
    ap = argparse.ArgumentParser(prog="jax_solitons.campaign.worker")
    ap.add_argument("--config-json", required=True, help="RunConfig as JSON")
    ap.add_argument("--run-fn", required=True, help="'module:function' RunFn ref")
    ap.add_argument("--work-dir", default="runs", help="registry/sink root dir")
    args = ap.parse_args()
    out = run_one(args.config_json, args.run_fn, args.work_dir)
    print(RESULT_PREFIX + json.dumps(out), flush=True)


if __name__ == "__main__":
    main()

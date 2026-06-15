"""Reap orphaned Vast instances -- the "clean up my farm" button.

The leak-proof ``rent()`` contract only protects the NORMAL exit path. A SIGKILL,
a crash, or a teardown REST call that itself fails on a flaky resolver all leave
GPUs billing by the second. This is the external recovery: list what's actually
live (the v1 endpoint -- the cost-safety source of truth) and destroy it, with
retry on transient errors so a network blip can't strand the cleanup the way it
stranded the rental.

Two scopes:
  - default: every live instance on the account (the "clean slate" button).
  - --ledger PATH: only instances this campaign rented but never recorded as
    destroyed (``rented``/``running`` minus ``destroyed``), intersected with
    what's actually still live -- so it won't touch unrelated instances.

SAFE BY DEFAULT: a bare run only LISTS (dry run). Destroying requires --yes.

  python -m jax_solitons.campaign.reap                 # list all live (dry run)
  python -m jax_solitons.campaign.reap --yes           # destroy ALL live
  python -m jax_solitons.campaign.reap --ledger out/vast_ledger.jsonl --yes
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def leaked_ids(ledger_path: str | Path) -> set[int]:
    """Instance ids a ledger rented/saw-running but never recorded destroyed.

    Pure (no network): the suspect set to intersect with what's actually live.
    """
    seen: set[int] = set()
    destroyed: set[int] = set()
    p = Path(ledger_path)
    if not p.exists():
        return set()
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        iid = ev.get("instance_id")
        if iid is None:
            continue
        iid = int(iid)
        if ev.get("event") in ("rented", "running"):
            seen.add(iid)
        elif ev.get("event") == "destroyed":
            destroyed.add(iid)
    return seen - destroyed


def _destroy_with_retry(provider, iid: int, retries: int = 4) -> bool:
    """Destroy, retrying on transient errors (the failure that stranded us)."""
    for attempt in range(retries):
        try:
            provider.destroy(iid)
            return True
        except Exception as e:  # noqa: BLE001 -- transient net/REST; retry then give up
            if attempt == retries - 1:
                print(f"  ! instance {iid}: destroy FAILED after {retries} tries: "
                      f"{type(e).__name__}: {str(e)[:120]}")
                return False
            time.sleep(2 ** attempt)
    return False


def reap(provider, *, ledger: str | Path | None = None, dry_run: bool = True,
         retries: int = 4) -> dict:
    """List live instances, destroy the targeted ones (unless dry_run).

    provider: anything with ``list_instances() -> [Instance(id,status,dph)]`` and
    ``destroy(id)`` (a VastProvider, or a fake in tests). With ``ledger`` set,
    only reaps this campaign's leaked-and-still-live instances. Returns a report
    dict (live / targeted / destroyed / failed / dph_reclaimed).
    """
    live = provider.list_instances()
    live_ids = {int(i.id) for i in live}
    if ledger is not None:
        suspects = leaked_ids(ledger)
        target_ids = live_ids & suspects
    else:
        target_ids = set(live_ids)
    targets = [i for i in live if int(i.id) in target_ids]
    dph = sum(float(getattr(i, "dph", 0) or 0) for i in targets)

    report = dict(live=len(live), targeted=len(targets), destroyed=[], failed=[],
                  dph_reclaimed=dph, dry_run=dry_run)
    if not targets:
        return report
    if dry_run:
        return report
    for i in targets:
        if _destroy_with_retry(provider, int(i.id), retries=retries):
            report["destroyed"].append(int(i.id))
        else:
            report["failed"].append(int(i.id))
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Reap orphaned Vast instances.")
    ap.add_argument("--ledger", default=None,
                    help="only reap instances leaked by this campaign's ledger "
                         "(rented/running minus destroyed), still live")
    ap.add_argument("--yes", action="store_true",
                    help="actually destroy (default is a dry-run listing)")
    ap.add_argument("--retries", type=int, default=4)
    args = ap.parse_args(argv)

    from jax_solitons.campaign.vast import VastProvider
    provider = VastProvider()

    live = provider.list_instances()
    scope = f"ledger {args.ledger}" if args.ledger else "ALL live instances"
    print(f"reap scope: {scope}")
    if not live:
        print("no live instances -- nothing to reap."); return 0
    for i in live:
        print(f"  instance {i.id}  status={i.status}  ${float(i.dph or 0):.4f}/hr")

    rep = reap(provider, ledger=args.ledger, dry_run=not args.yes, retries=args.retries)
    print(f"\nlive={rep['live']} targeted={rep['targeted']} "
          f"(~${rep['dph_reclaimed']:.3f}/hr)")
    if rep["dry_run"]:
        if rep["targeted"]:
            print(f"DRY RUN -- pass --yes to destroy {rep['targeted']} instance(s).")
        return 0
    print(f"destroyed {len(rep['destroyed'])}: {rep['destroyed']}")
    if rep["failed"]:
        print(f"FAILED {len(rep['failed'])}: {rep['failed']} -- re-run to retry.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

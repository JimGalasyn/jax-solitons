"""Reap orphaned Vast instances -- the "clean up my farm" button.

The leak-proof ``rent()`` contract only protects the NORMAL exit path. A SIGKILL,
a crash, or a teardown REST call that itself fails on a flaky resolver all leave
GPUs billing by the second. This is the external recovery: list what's actually
live (the v1 endpoint -- the cost-safety source of truth) and destroy it, with
retry on transient errors so a network blip can't strand the cleanup the way it
stranded the rental. Destroy is idempotent: an already-gone instance counts as
success (that's the desired state), not a failure.

Two scopes:
  - --ledger PATH (recommended): only instances this campaign rented but never
    recorded destroyed (``rented``/``running`` minus ``destroyed``), intersected
    with what's actually still live -- safe when several sessions share one Vast
    account, since it won't touch instances this ledger never created.
  - --all: EVERY live instance on the account (the "clean slate" button). With
    concurrent farming this also kills other sessions' boxes -- hence opt-in.

SAFE BY DEFAULT: a bare run only lists (dry run). Destroying needs --yes, and the
unscoped all-account destroy additionally needs --all.

  python -m jax_solitons.campaign.reap                                 # list (dry run)
  python -m jax_solitons.campaign.reap --ledger out/vast_ledger.jsonl --yes
  python -m jax_solitons.campaign.reap --all --yes                     # nuke everything
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

# substrings that classify a destroy error (VastError msg carries "HTTP <code>")
_GONE = ("404", "410", "not found", "notfound", "no_such", "does not exist")
_AUTH = ("401", "403", "forbidden", "unauthor")


def leaked_ids(ledger_path: str | Path) -> set[int]:
    """Instance ids a ledger rented/saw-running but never CONFIRMED destroyed.

    Pure (no network): the suspect set to intersect with what's actually live.
    A ``destroyed`` event only clears a leak when it is a *confirmed* teardown
    (``verify == "gone"``) -- the leak-proof rent() also logs a ``destroyed``
    event when teardown FAILED (``verify == "present"`` / ``destroyed: false``),
    and counting those as gone would make ledger-scoped reaping miss exactly the
    leaks it exists to catch. Erring toward "still leaked" is safe: reap()
    intersects this set with what's actually live, so a box that really is gone
    just won't be a target. Non-numeric ids (other providers) are skipped.
    """
    seen: set[int] = set()
    confirmed_gone: set[int] = set()
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
        try:
            iid = int(iid)
        except (TypeError, ValueError):
            continue                                    # string ids (other providers)
        evt = ev.get("event")
        if evt in ("rented", "running"):
            seen.add(iid)
        elif evt == "destroyed" and ev.get("verify") == "gone":
            confirmed_gone.add(iid)                     # only a verified teardown clears it
    return seen - confirmed_gone


def _classify(exc: Exception) -> str:
    """transient (retry) | gone (already destroyed -> success) | auth (terminal)."""
    s = str(exc).lower()
    if any(t in s for t in _GONE):
        return "gone"
    if any(t in s for t in _AUTH):
        return "auth"
    return "transient"


def _destroy_with_retry(provider, iid: int, retries: int = 4) -> str:
    """Idempotent destroy. Returns 'destroyed' | 'gone' (already absent, success)
    | 'failed'. Retries only TRANSIENT errors; an already-gone instance is the
    desired state (no wasted backoff), and auth/permission fails fast."""
    for attempt in range(retries):
        try:
            provider.destroy(iid)
            return "destroyed"
        except Exception as e:  # noqa: BLE001 -- classify, then retry/skip/fail
            kind = _classify(e)
            if kind == "gone":
                return "gone"                       # idempotent: already destroyed
            if kind == "auth":
                print(f"  ! instance {iid}: destroy refused (auth/permission): "
                      f"{str(e)[:120]}")
                return "failed"                     # terminal -- don't burn backoff
            if attempt == retries - 1:
                print(f"  ! instance {iid}: destroy FAILED after {retries} tries: "
                      f"{type(e).__name__}: {str(e)[:120]}")
                return "failed"
            time.sleep(2 ** attempt)
    return "failed"


def reap(provider, *, ledger: str | Path | None = None, dry_run: bool = True,
         retries: int = 4, live=None) -> dict:
    """List live instances, destroy the targeted ones (unless dry_run).

    provider: anything with ``list_instances() -> [Instance(id,status,dph)]`` and
    ``destroy(id)`` (a VastProvider, or a fake in tests). Pass ``live`` to reuse
    an already-fetched listing (avoids a second API call + TOCTOU). With
    ``ledger`` set, only reaps this campaign's leaked-and-still-live instances.
    Returns a report dict; ``destroyed`` and ``gone`` are both successes (in the
    desired state), ``failed`` is the only error bucket.
    """
    if live is None:
        live = provider.list_instances()
    live_ids = {int(i.id) for i in live}
    if ledger is not None:
        target_ids = live_ids & leaked_ids(ledger)
    else:
        target_ids = set(live_ids)
    targets = [i for i in live if int(i.id) in target_ids]
    dph = sum(float(getattr(i, "dph", 0) or 0) for i in targets)

    report = dict(live=len(live), targeted=len(targets), destroyed=[], gone=[],
                  failed=[], dph_reclaimed=dph, dry_run=dry_run)
    if dry_run or not targets:
        return report
    for i in targets:
        status = _destroy_with_retry(provider, int(i.id), retries=retries)
        report[status].append(int(i.id))           # destroyed | gone | failed
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Reap orphaned Vast instances.")
    ap.add_argument("--ledger", default=None,
                    help="RECOMMENDED scope: only reap instances this ledger "
                         "leaked (rented/running minus destroyed), still live")
    ap.add_argument("--all", action="store_true", dest="all_scope",
                    help="scope = EVERY live instance on the account (also kills "
                         "other sessions' boxes); required for an unscoped destroy")
    ap.add_argument("--yes", action="store_true",
                    help="actually destroy (default is a dry-run listing)")
    ap.add_argument("--retries", type=int, default=4)
    args = ap.parse_args(argv)

    from jax_solitons.campaign.vast import VastProvider
    provider = VastProvider()

    live = provider.list_instances()                # fetch ONCE, reuse below
    scope = f"ledger {args.ledger}" if args.ledger else "ALL live instances"
    print(f"reap scope: {scope}")
    if not live:
        print("no live instances -- nothing to reap."); return 0
    for i in live:
        print(f"  instance {i.id}  status={i.status}  ${float(i.dph or 0):.4f}/hr")

    # safety gate: an unscoped (all-account) destroy must be explicit
    if args.yes and args.ledger is None and not args.all_scope:
        print("\nREFUSING unscoped destroy: this would target ALL "
              f"{len(live)} live instances, including any concurrent session's "
              "boxes. Re-run with --ledger <path> (safe) or --all (clean slate).")
        return 2

    rep = reap(provider, ledger=args.ledger, dry_run=not args.yes,
               retries=args.retries, live=live)
    print(f"\nlive={rep['live']} targeted={rep['targeted']} "
          f"(~${rep['dph_reclaimed']:.3f}/hr)")
    if rep["dry_run"]:
        if rep["targeted"]:
            scope_note = "" if args.ledger else "  [ALL-ACCOUNT scope]"
            print(f"DRY RUN -- pass --yes to destroy {rep['targeted']} "
                  f"instance(s).{scope_note}")
        return 0
    if args.ledger is None:
        print("!! ALL-ACCOUNT scope: destroying every live instance !!")
    done = rep["destroyed"] + rep["gone"]
    print(f"cleared {len(done)} ({len(rep['destroyed'])} destroyed, "
          f"{len(rep['gone'])} already gone): {done}")
    if rep["failed"]:
        print(f"FAILED {len(rep['failed'])}: {rep['failed']} -- re-run to retry.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

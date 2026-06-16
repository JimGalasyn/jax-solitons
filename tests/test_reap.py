"""Reaper logic: ledger-diff, targeting, idempotent/classified destroy. No network."""
import json


class _Inst:
    def __init__(self, id, status="running", dph=0.1, label=None):
        self.id = id; self.status = status; self.dph = dph
        self.raw = {"label": label} if label else {}    # Vast stores the label here


class _FakeProvider:
    """list_instances + destroy. `errors` maps id -> list of exceptions raised on
    successive destroy calls (then it succeeds). `labels` maps id -> LaunchSpec
    label (for --label scope). Records every attempt."""
    def __init__(self, ids, errors=None, labels=None):
        labels = labels or {}
        self._live = {i: _Inst(i, label=labels.get(i)) for i in ids}
        self.destroyed = []
        self.attempts = []          # every destroy call (incl. failed)
        self.list_calls = 0
        self._errors = {k: list(v) for k, v in (errors or {}).items()}

    def list_instances(self):
        self.list_calls += 1
        return list(self._live.values())

    def destroy(self, iid):
        self.attempts.append(iid)
        q = self._errors.get(iid)
        if q:
            raise q.pop(0)
        self.destroyed.append(iid)
        self._live.pop(iid, None)


def _ledger(tmp_path, events):
    p = tmp_path / "vast_ledger.jsonl"
    p.write_text("\n".join(json.dumps(e) for e in events) + "\n")
    return p


# -- ledger diff --------------------------------------------------------------
def test_leaked_ids_diffs_rented_minus_destroyed(tmp_path):
    from jax_solitons.campaign.reap import leaked_ids
    led = _ledger(tmp_path, [
        {"event": "rented", "instance_id": 1},
        {"event": "running", "instance_id": 1},
        {"event": "rented", "instance_id": 2},
        {"event": "destroyed", "instance_id": 2, "verify": "gone"},
        {"event": "rented", "instance_id": 3},
    ])
    assert leaked_ids(led) == {1, 3}


def test_leaked_ids_missing_file_is_empty(tmp_path):
    from jax_solitons.campaign.reap import leaked_ids
    assert leaked_ids(tmp_path / "nope.jsonl") == set()


def test_leaked_ids_failed_teardown_stays_leaked(tmp_path):
    # a `destroyed` event clears a leak ONLY when verify=="gone" (issue #30/PR#22
    # review): a failed/unverified teardown must remain a suspect.
    from jax_solitons.campaign.reap import leaked_ids
    led = _ledger(tmp_path, [
        {"event": "rented", "instance_id": 5},
        {"event": "destroyed", "instance_id": 5, "destroyed": False, "verify": "present"},
        {"event": "rented", "instance_id": 6},
        {"event": "destroyed", "instance_id": 6, "verify": "gone"},
    ])
    assert leaked_ids(led) == {5}                    # 5 failed teardown; 6 confirmed gone


def test_leaked_ids_skips_nonnumeric_ids(tmp_path):
    from jax_solitons.campaign.reap import leaked_ids
    led = _ledger(tmp_path, [
        {"event": "rented", "instance_id": "i-abc"},  # string id (other provider)
        {"event": "rented", "instance_id": 7},
    ])
    assert leaked_ids(led) == {7}                    # didn't crash; skipped the string id


# -- targeting / scope --------------------------------------------------------
def test_reap_dry_run_destroys_nothing():
    from jax_solitons.campaign.reap import reap
    p = _FakeProvider([10, 11, 12])
    rep = reap(p, dry_run=True)
    assert rep["targeted"] == 3 and rep["destroyed"] == [] and p.destroyed == []


def test_reap_all_live_destroys_all():
    from jax_solitons.campaign.reap import reap
    p = _FakeProvider([10, 11, 12])
    rep = reap(p, dry_run=False)
    assert sorted(rep["destroyed"]) == [10, 11, 12] and sorted(p.destroyed) == [10, 11, 12]


def test_reap_ledger_scope_only_leaked_and_live(tmp_path):
    from jax_solitons.campaign.reap import reap
    led = _ledger(tmp_path, [
        {"event": "rented", "instance_id": 1},
        {"event": "rented", "instance_id": 3},
        {"event": "destroyed", "instance_id": 3, "verify": "gone"},
    ])
    p = _FakeProvider([1, 99])                       # leaked={1,3}, live={1,99}
    rep = reap(p, ledger=led, dry_run=False)
    assert rep["destroyed"] == [1] and p.destroyed == [1]   # 99 untouched, 3 not live


def test_label_of_reads_raw_label():
    from jax_solitons.campaign.reap import _label_of
    assert _label_of(_Inst(1, label="farm-x")) == "farm-x"
    assert _label_of(_Inst(1)) is None                # no label stamped
    assert _label_of(object()) is None                # no `raw` attr -> robust


def test_reap_label_scope_only_matching_label():
    from jax_solitons.campaign.reap import reap
    p = _FakeProvider([10, 11, 12], labels={10: "farmA", 11: "farmA", 12: "farmB"})
    rep = reap(p, label="farmA", dry_run=False)
    assert rep["targeted"] == 2
    assert sorted(rep["destroyed"]) == [10, 11] and sorted(p.destroyed) == [10, 11]


def test_reap_label_and_ledger_intersect(tmp_path):
    # scopes are ANDed: leaked={10,11}, label farmA={10,12} -> only 10 reaped
    from jax_solitons.campaign.reap import reap
    led = _ledger(tmp_path, [
        {"event": "rented", "instance_id": 10},
        {"event": "rented", "instance_id": 11},
    ])
    p = _FakeProvider([10, 11, 12], labels={10: "farmA", 11: "farmB", 12: "farmA"})
    rep = reap(p, ledger=led, label="farmA", dry_run=False)
    assert rep["destroyed"] == [10]


def test_reap_reuses_prefetched_live_no_extra_list_call():
    from jax_solitons.campaign.reap import reap
    p = _FakeProvider([10])
    live = p.list_instances()                        # caller's single fetch
    assert p.list_calls == 1
    reap(p, dry_run=True, live=live)
    assert p.list_calls == 1                         # reap did not re-list (issue #30.2)


# -- classified / idempotent destroy (issue #30.1) ----------------------------
def test_reap_retries_transient_then_succeeds(monkeypatch):
    from jax_solitons.campaign import reap as reapmod
    from jax_solitons.campaign.reap import reap
    monkeypatch.setattr(reapmod.time, "sleep", lambda *_: None)   # no real backoff
    p = _FakeProvider([10], errors={10: [OSError("Temporary failure in name resolution")] * 2})
    rep = reap(p, dry_run=False, retries=4)
    assert rep["destroyed"] == [10] and rep["failed"] == []
    assert p.attempts.count(10) == 3                 # 2 transient fails + 1 success


def test_reap_already_gone_counts_as_success_not_failure():
    from jax_solitons.campaign.reap import reap
    p = _FakeProvider([10], errors={10: [RuntimeError(
        "DELETE /api/v0/instances/10/ -> HTTP 404: {'msg':'not found'}")]})
    rep = reap(p, dry_run=False, retries=4)
    assert rep["gone"] == [10] and rep["destroyed"] == [] and rep["failed"] == []
    assert p.attempts.count(10) == 1                 # no wasted backoff on already-gone


def test_reap_auth_error_fails_fast_no_retry():
    from jax_solitons.campaign.reap import reap
    p = _FakeProvider([10], errors={10: [RuntimeError(
        "DELETE /api/v0/instances/10/ -> HTTP 403: forbidden")] * 9})
    rep = reap(p, dry_run=False, retries=4)
    assert rep["failed"] == [10]
    assert p.attempts.count(10) == 1                 # terminal -> one attempt, no backoff


def test_reap_permanent_transient_failure_reported(monkeypatch):
    from jax_solitons.campaign import reap as reapmod
    from jax_solitons.campaign.reap import reap
    monkeypatch.setattr(reapmod.time, "sleep", lambda *_: None)   # no real backoff
    p = _FakeProvider([10], errors={10: [OSError("conn reset")] * 99})
    rep = reap(p, dry_run=False, retries=3)
    assert rep["failed"] == [10] and rep["destroyed"] == [] and rep["gone"] == []
    assert p.attempts.count(10) == 3                 # exhausted the retries


# -- main() CLI: --label scope + safety gate (issue #24) ----------------------
def _patch_provider(monkeypatch, provider):
    import jax_solitons.campaign.vast as vast
    monkeypatch.setattr(vast, "VastProvider", lambda: provider)


def test_main_label_scope_destroys_only_that_label(monkeypatch, capsys):
    from jax_solitons.campaign.reap import main
    p = _FakeProvider([10, 11], labels={10: "farmA", 11: "farmB"})
    _patch_provider(monkeypatch, p)
    rc = main(["--label", "farmA", "--yes"])
    assert rc == 0 and p.destroyed == [10]           # farmB (11) untouched
    out = capsys.readouterr().out
    assert "label 'farmA'" in out and "label=farmA" in out


def test_main_label_dry_run_lists_only(monkeypatch, capsys):
    from jax_solitons.campaign.reap import main
    p = _FakeProvider([10, 11], labels={10: "farmA", 11: "farmB"})
    _patch_provider(monkeypatch, p)
    rc = main(["--label", "farmA"])                  # no --yes
    assert rc == 0 and p.destroyed == []
    assert "DRY RUN" in capsys.readouterr().out


def test_main_unscoped_destroy_refused(monkeypatch, capsys):
    """--yes with no --ledger/--label/--all must refuse (won't nuke other sessions)."""
    from jax_solitons.campaign.reap import main
    p = _FakeProvider([10, 11])
    _patch_provider(monkeypatch, p)
    rc = main(["--yes"])
    assert rc == 2 and p.destroyed == []
    assert "REFUSING unscoped destroy" in capsys.readouterr().out

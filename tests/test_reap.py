"""Reaper logic: ledger-diff + dry-run/destroy targeting, no network."""
import json

import pytest

from jax_solitons.campaign.reap import leaked_ids, reap


class _Inst:
    def __init__(self, id, status="running", dph=0.1):
        self.id = id; self.status = status; self.dph = dph


class _FakeProvider:
    """list_instances + destroy; destroy fails `flaky` times then succeeds."""
    def __init__(self, ids, flaky=0):
        self._live = {i: _Inst(i) for i in ids}
        self.destroyed = []
        self._flaky = flaky

    def list_instances(self):
        return list(self._live.values())

    def destroy(self, iid):
        if self._flaky > 0:
            self._flaky -= 1
            raise OSError("Temporary failure in name resolution")
        self.destroyed.append(iid)
        self._live.pop(iid, None)


def _write_ledger(tmp_path, events):
    p = tmp_path / "vast_ledger.jsonl"
    p.write_text("\n".join(json.dumps(e) for e in events) + "\n")
    return p


def test_leaked_ids_diffs_rented_minus_destroyed(tmp_path):
    led = _write_ledger(tmp_path, [
        {"event": "rented", "instance_id": 1},
        {"event": "running", "instance_id": 1},
        {"event": "rented", "instance_id": 2},
        {"event": "destroyed", "instance_id": 2},
        {"event": "rented", "instance_id": 3},
    ])
    assert leaked_ids(led) == {1, 3}


def test_leaked_ids_missing_file_is_empty(tmp_path):
    assert leaked_ids(tmp_path / "nope.jsonl") == set()


def test_reap_dry_run_destroys_nothing():
    p = _FakeProvider([10, 11, 12])
    rep = reap(p, dry_run=True)
    assert rep["targeted"] == 3 and rep["destroyed"] == [] and p.destroyed == []
    assert rep["dry_run"] is True


def test_reap_all_live_destroys_all():
    p = _FakeProvider([10, 11, 12])
    rep = reap(p, dry_run=False)
    assert sorted(rep["destroyed"]) == [10, 11, 12]
    assert sorted(p.destroyed) == [10, 11, 12]


def test_reap_ledger_scope_only_leaked_and_live(tmp_path):
    # ledger leaked {1,3}; but only {1,99} are actually live -> reap just {1}
    led = _write_ledger(tmp_path, [
        {"event": "rented", "instance_id": 1},
        {"event": "rented", "instance_id": 3},
        {"event": "destroyed", "instance_id": 3},
    ])
    p = _FakeProvider([1, 99])
    rep = reap(p, ledger=led, dry_run=False)
    assert rep["destroyed"] == [1] and p.destroyed == [1]  # 99 untouched, 3 not live


def test_reap_retries_transient_destroy_failures():
    p = _FakeProvider([10], flaky=2)          # first 2 destroys raise, 3rd works
    rep = reap(p, dry_run=False, retries=4)
    assert rep["destroyed"] == [10] and rep["failed"] == []


def test_reap_reports_permanent_failure():
    p = _FakeProvider([10], flaky=99)         # always fails
    rep = reap(p, dry_run=False, retries=3)
    assert rep["failed"] == [10] and rep["destroyed"] == []

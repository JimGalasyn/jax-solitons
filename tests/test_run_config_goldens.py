"""Frozen run identities: the contract that survives the run-farm extraction.

`RunConfig.to_json()` bytes ARE the permanent names of every run directory ever
written -- `config_hash` names the dir, and the registry's idempotent skip
(mechanism A) resolves a prior run by that name. So a change to the config's
serialization silently renames every run: `is_complete` stops recognizing finished
work, a resumed campaign restarts from zero, and a rented fleet re-bills for
results already on disk. Nothing raises; it just quietly costs money and time.

The rows in tests/data/run_name_goldens.jsonl were generated from the
PRE-EXTRACTION code, and the `campaign_out/*` ones carry run names read off the
real ledger on disk rather than computed. **These tests do not get updated to match
new behavior; they get obeyed.** A failure here means the extraction broke run
identity -- fix the code, not the fixture.

(campaign_out/ is gitignored, so the ledger rows are copied into the fixture; that
copy is the only committed evidence of the pre-extraction naming.)
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from jax_solitons.runs import RunConfig, run_dir

GOLDENS = [json.loads(ln) for ln in
           (Path(__file__).parent / "data/run_name_goldens.jsonl")
           .read_text().splitlines() if ln.strip()]

# The subset with names read off real run dirs / MANIFEST rows rather than computed.
FROM_LEDGER = [g for g in GOLDENS if g["source"].startswith("campaign_out")]


def _ids(goldens):
    return [f"{g['run_name']}" for g in goldens]


def test_fixture_is_present_and_covers_the_real_ledger():
    """Guard the guard: a fixture that silently emptied would pass every test
    below vacuously."""
    assert len(GOLDENS) >= 25
    assert len(FROM_LEDGER) >= 4, "real on-disk ledger rows must be represented"


@pytest.mark.parametrize("g", GOLDENS, ids=_ids(GOLDENS))
def test_run_identity_is_frozen(g):
    """to_json bytes, config_hash, and run_name are all exactly as pinned."""
    c = RunConfig(**g["config"])
    # bytes first: it's the root cause of the other two, so it fails most usefully
    assert c.to_json() == g["to_json"]
    assert c.config_hash() == g["config_hash"]
    assert c.run_name() == g["run_name"]


@pytest.mark.parametrize("g", GOLDENS, ids=_ids(GOLDENS))
def test_from_json_round_trips_to_the_same_identity(g):
    """A worker rebuilds the config from JSON and must land on the same run.

    This is the remote path: the driver serializes, the box deserializes, and both
    sides must agree on the run directory or the box writes results somewhere the
    driver will never look.
    """
    c = RunConfig.from_json(g["to_json"])
    assert c.config_hash() == g["config_hash"]
    assert c.run_name() == g["run_name"]
    assert c.to_json() == g["to_json"], "round-trip must be byte-stable"


@pytest.mark.parametrize("g", FROM_LEDGER, ids=_ids(FROM_LEDGER))
def test_ledger_rows_still_resolve_to_their_directories(g, tmp_path):
    """The full trip the registry does: MANIFEST row -> from_json -> run_dir, and
    the directory it picks must be the one already sitting in campaign_out on disk.

    This is the direct answer to "do existing ledgers still resolve after the
    extraction?" -- and it exercises RunConfig, run_dir, and (post-extraction) the
    re-export from run_farm, in one shot.
    """
    c = RunConfig.from_json(g["to_json"])
    assert run_dir(tmp_path, c).name == g["run_name"]


def test_run_dir_writes_a_manifest_row_matching_the_golden(tmp_path):
    """run_dir's registry side effect (the MANIFEST line) must keep the same shape
    the real ledger has -- it's what a later reader parses to find prior work."""
    g = FROM_LEDGER[0]
    c = RunConfig.from_json(g["to_json"])
    run_dir(tmp_path, c)
    rows = [json.loads(ln) for ln in
            (tmp_path / "MANIFEST.jsonl").read_text().splitlines() if ln.strip()]
    assert rows == [{"run": g["run_name"], "config": g["config"]}]


def test_config_hash_is_insensitive_to_params_key_order():
    """Two configs differing only in dict insertion order are the SAME run.

    Load-bearing for the remote path: a worker rebuilding params from JSON gets
    whatever order the serializer emitted, and it must not fork the identity.
    """
    a = RunConfig(model="m", N=8, L=1.0, params={"z": 1, "a": 2})
    b = RunConfig(model="m", N=8, L=1.0, params={"a": 2, "z": 1})
    assert a.config_hash() == b.config_hash()

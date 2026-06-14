"""Multi-provider partition-and-merge runner: splitting, concurrent merge,
failure isolation, and duplicate detection -- all with fake executors, no clouds."""

import pytest

from jax_solitons.campaign.multi import CampaignReport, run_multi, split_configs
from jax_solitons.runs import RunConfig


def _cfgs(n):
    return [RunConfig(model="faddeev_cp1", N=16, L=12.0, params={"R": 2.0 + i})
            for i in range(n)]


class FakeExec:
    """A remote executor stand-in: .name + run(configs) -> result records."""

    def __init__(self, name, *, fail=False):
        self.name = name
        self.fail = fail
        self.seen = None

    def run(self, configs, *, admission=None):
        self.seen = list(configs)
        if self.fail:
            raise RuntimeError("cloud down")
        return [{"run": c.run_name(), "result": {"R": c.params["R"]},
                 "skipped": False} for c in self.seen]


def test_split_round_robin():
    e1, e2 = FakeExec("a"), FakeExec("b")
    asg = dict((ex, cfgs) for ex, cfgs in split_configs(_cfgs(5), [e1, e2]))
    assert len(asg[e1]) == 3 and len(asg[e2]) == 2        # 5 split 3/2


def test_split_weighted():
    e1, e2 = FakeExec("a"), FakeExec("b")
    asg = dict(split_configs(_cfgs(4), [e1, e2], weights=[3, 1]))
    assert len(asg[e1]) == 3 and len(asg[e2]) == 1
    # largest-remainder keeps the total exact even when it doesn't divide evenly
    asg2 = dict(split_configs(_cfgs(5), [e1, e2], weights=[2, 1]))
    assert sum(len(v) for v in asg2.values()) == 5


def test_run_multi_merges_and_annotates():
    e1, e2 = FakeExec("modal"), FakeExec("vast")
    configs = _cfgs(6)
    report = run_multi(split_configs(configs, [e1, e2]))
    assert isinstance(report, CampaignReport) and report.ok
    # every config harvested exactly once, keyed by its content hash
    assert set(report.results) == {c.run_name() for c in configs}
    # each record is annotated with the provider that produced it
    provs = {r["provider"] for r in report.results.values()}
    assert provs == {"modal", "vast"}
    assert not report.duplicates


def test_run_multi_isolates_a_failing_provider():
    good, bad = FakeExec("modal"), FakeExec("vast", fail=True)
    configs = _cfgs(4)
    report = run_multi(split_configs(configs, [good, bad]))
    assert not report.ok                                   # partial
    # the good provider's results survive the bad one's failure
    assert all(r["provider"] == "modal" for r in report.results.values())
    assert len(report.results) == len(good.seen)
    errs = [p for p in report.by_provider if not p.ok]
    assert len(errs) == 1 and "cloud down" in errs[0].error


def test_run_multi_flags_nondisjoint_partition():
    e1, e2 = FakeExec("modal"), FakeExec("vast")
    (c,) = _cfgs(1)
    report = run_multi([(e1, [c]), (e2, [c])])             # same config to both!
    assert c.run_name() in report.duplicates
    assert set(report.duplicates[c.run_name()]) == {"modal", "vast"}


def test_run_multi_empty_is_noop():
    report = run_multi([])
    assert report.results == {} and report.by_provider == []
    assert "0 runs" in report.summary()

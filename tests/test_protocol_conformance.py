"""jax_solitons.RunConfig must satisfy the run_farm.RunConfig Protocol.

The extraction leaves this relationship STRUCTURAL and unenforced by imports: the
campaign layer (now run-farm) types against a Protocol, and jax_solitons.RunConfig
happens to satisfy it. Nothing makes that true at import time -- a field rename or a
`to_json` "cleanup" in runs.py would compile, pass most tests, and then fail on a
rented box when the worker rebuilds the config from JSON. This test is the cheap
thing standing in that gap. If it fails, the extraction's seam is broken.
"""

from run_farm.protocols import RunConfig as RunConfigProtocol

from jax_solitons.runs import RunConfig


def test_run_config_satisfies_run_farm_protocol():
    c = RunConfig(model="faddeev_cp1", N=16, L=12.0, params={"R": 2.0})
    # isinstance works on a @runtime_checkable Protocol (presence-only).
    # NOTE: issubclass() would raise TypeError on a data-member Protocol -- don't.
    assert isinstance(c, RunConfigProtocol)


def test_run_config_has_the_contract_surface():
    """The six members run-farm actually calls, spelled out so a rename here fails
    LOUDLY at CI rather than silently on a worker."""
    c = RunConfig(model="m", N=8, L=1.0, dtype="float64", params={"k": 1})
    assert c.dtype == "float64"
    assert c.params == {"k": 1}
    assert isinstance(c.to_json(), str)
    assert RunConfig.from_json(c.to_json()) == c
    assert isinstance(c.config_hash(), str)
    assert c.run_name().endswith(c.config_hash())


def test_config_class_ref_rebuilds_remotely():
    """The remote path in miniature: a worker imports the config class by
    'module:ClassName' ref and rebuilds from JSON. Proves the ref run-farm ships
    for jax-solitons resolves and round-trips to the same identity."""
    from run_farm.remote import load_config_class
    cls = load_config_class("jax_solitons.runs:RunConfig")
    assert cls is RunConfig
    c = RunConfig(model="faddeev_cp1", N=16, L=12.0, params={"R": 2.0})
    assert cls.from_json(c.to_json()).config_hash() == c.config_hash()

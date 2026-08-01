# B=2 binding calibration — the evidence

These four manifests are the run outputs behind the resolution ladder in
[`../SKYRME.md`](../SKYRME.md): the claim that a dx → 0 ladder recovers the published
classical massless Skyrme **B=2 binding of ~4.3%**.

| file | N | dx | E(B=1)/bound | E(B=2)/bound | binding |
|---|---|---|---|---|---|
| `N64.json` | 64 | 0.250 | 1.198 | 1.152 | 3.91% |
| `N96.json` | 96 | 0.167 | 1.230 | 1.176 | 4.39% |
| `N128.json` | 128 | 0.125 | 1.240 | 1.184 | 4.52% |
| `N160.json` | 160 | 0.100 | 1.245 | 1.188 | 4.58% |
| *ideal* | | | *1.232* | *1.179* | *~4.3%* |

Every leg is above the Bogomolny bound with B held at +1/+2 (`physical: true`,
`sub_bound_seen: false` in each manifest), which is what makes the ladder a
convergence result rather than a collapse.

## Why they are here

They were produced on 2026-06-27 by a Vast fleet run driven from
`null-worldtube-private`, at `simulations/engine_dogfood/output/skyrme_converge_fleet/`
— and **they were never tracked in git there.** That repo is deprecated and being
dismantled, so the evidence for this module's headline result existed as untracked
files on one machine's disk. Copying them here puts them under version control beside
the model they validate, in the repo that makes the claim.

The drivers came with them, in [`../../scripts/skyrme_converge/`](../../scripts/skyrme_converge/):

- `run_skyrme_converge_fleet.py` — one fleet leg per resolution N
- `skyrme_converge_batch.py` — the on-box driver; relaxes both the B=1 hedgehog and
  the B=2 rational-map torus at the same N, so the pair shares a box and the binding
  is clean
- `skyrme_converge_analyze.py` — collates the manifests into the table above
- `onstart_skyrme.sh` — the Vast worker bootstrap

## Reproducing the table from these files

```
python scripts/skyrme_converge/skyrme_converge_analyze.py \
    docs/skyrme_calibration/N*.json
```

Verified on migration (2026-08-01): that command reproduces the ladder in `SKYRME.md`
row for row, from these manifests, with no network and no GPU.

## What was changed on migration, and what was not

**The manifests are byte-identical to the originals.** Nothing was edited, rounded or
regenerated — a regenerated state would not be the same evidence.

**One driver was repaired.** `run_skyrme_converge_fleet.py` imported
`jax_solitons.campaign`, which was extracted to run-farm in 0.0.8; the import had been
dead since then, so the script could not have run as written. The symbols moved without
being renamed, so the fix is only a change of home:

| symbol | was | now |
|---|---|---|
| `FleetExecutor`, `FleetLeg`, `SentinelReady` | `jax_solitons.campaign` | `run_farm.fleet` |
| `HostSpec`, `LaunchSpec` | `jax_solitons.campaign` | `run_farm.protocols` |
| `VastLedger`, `VastProvider` | `jax_solitons.campaign` | `run_farm.vast` |

A hardcoded `sys.path.insert("/home/jim/.venv/...")` was also dropped, and the
`onstart_skyrme.sh` path updated for the flatter layout here. The driver now imports and
runs `--help`; it did neither before.

## Not migrated

`vast_ledger.jsonl` from the same run — rental provenance (instance and offer ids, GPU
model, region, price per hour). It ties each leg to the hardware that produced it, which
is real provenance, but it is operational spend data rather than physics and this repo is
public. It remains in `null-worldtube-private`; say the word if it should follow.

## Caveat worth keeping

These are *run outputs*, not a test. They record what one fleet produced on one day. The
tests that hold the physics continuously live in `tests/test_skyrme_exact.py`; these
manifests are the provenance for a number quoted in prose, and should be read as such.

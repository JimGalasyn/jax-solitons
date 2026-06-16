# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project aims to follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.0.4] - 2026-06-16

### Added
- **`reap --label NAME`** (#24) — a third reap scope: destroy only instances
  stamped with that `LaunchSpec.label`. Safe under concurrent farming like
  `--ledger` (no `--all` needed) but ledger-free, so a crashed run's boxes can be
  reaped by campaign label from any machine. Scopes AND together (`--ledger` ∩
  `--label`); the label is read from the instance's `raw` record. Completes the
  proactive-attribution half of #24 (the `LaunchSpec.label` stamping shipped in
  #33).
- **`FleetExecutor`** (`campaign.fleet`, #25) — a parallel, one-rented-host-per-leg
  *script* fleet over any `Provider`. A fleet run is now data — a list of
  `FleetLeg(label, command, ship, fetch)` — instead of a forked driver, so the
  three hand-rolled private drivers (`run_eps_fleet` / `run_stability_fleet` /
  `run_eps_kick_fleet`) collapse to thin callers. Robustness from the 2026-06-15
  farming session is built in: per-leg failover on a bad host (`HostProbeFailed`)
  or offer race (`RentUnavailable`); **fast-fail a corpse** via the provider's API
  status instead of ssh-polling for the full deadline (#27); **offer-pool refresh**
  when it drains under failover (#28); **resume/skip** legs whose output already
  exists (#26); **launch jitter** against the thundering-herd (#29); and a
  **signal-safe teardown** backstop that destroys in-flight rentals on
  SIGTERM/SIGINT (#24). Ships `FleetLeg`, `LegResult`, and the `ImportReady` /
  `SentinelReady` readiness probes.
- **`RentUnavailable`** — a `Provider` failover signal distinct from
  `HostProbeFailed`: the offer was taken before an instance was created, so there
  is nothing to tear down and the executor just tries the next offer.
  `VastProvider.rent` raises it on a create-time race; `ProviderExecutor` and
  `FleetExecutor` both fail over on it.
- **`VastProvider.dead_reason`** — reports a visibly-failed instance
  (error/exited or a bad-host `status_msg`) so a readiness loop can fast-fail it
  (#27); keeps the bad-host string matching in the adapter so executors stay
  provider-agnostic.
- **`campaign status`** (`campaign.status`, #29) — `python -m
  jax_solitons.campaign.status --ledger <path>` prints live instances (what is
  billing now) + cumulative spend/outcomes from the ledger, replacing
  log-grepping and shelling `vastai`.
- **Self-healing Vast REST calls** (`campaign.vast._req`, #23) — every Vast API
  call now retries transient transport faults (DNS `EAI_AGAIN`, connection reset,
  read timeout, 5xx) with exponential backoff before failing, so a saturated
  local resolver no longer turns a whole fleet of legs into terminal failures.
  Retry is idempotent-aware: `create` retries only pre-send DNS failures so a
  half-completed rent can never double-rent a GPU. A raw `URLError` no longer
  escapes — every network failure surfaces as `VastError`.
- **`LaunchSpec.label`** (#24) — stamps every rented instance with a campaign/run
  id so a live instance is attributable to its run (identifiable in `vastai` / the
  API), the proactive half of orphan prevention and the hook for a future
  label-scoped `reap`. (`reap` today still scopes by ledger ids or `--all`.)
- **`make_verlet_step` is now exported** from `jax_solitons.steppers` (#29) — the
  factory was previously reachable only via the `.verlet` submodule.

## [0.0.3] - 2026-06-15

### Added
- **Core-curve knot ID (`jax_solitons.knots`)** — the inverse of the carrier
  ladder: given a soliton field, trace its core curve (predictor-corrector on the
  implicit `{n1=0, n2=0}` / `{Re ψ=0, Im ψ=0}` set) and read off the Alexander
  determinant via pyknotid (unknot=1, trefoil=3, cinquefoil=5, …). Includes
  `core_curves_from_n` / `core_curves_from_psi`, `identify_knot` /
  `identify_core_knot`, `curve_energy_scores`, `trace_implicit_curve`, and a
  `with_time_limit` guard (the pure-Python tracer/Alexander can go pathological on
  turbulent evolved fields).
- **Coupled L₂ + L₃ gauged Faddeev–Skyrme–Higgs model** (`models.gauged_faddeev`,
  Paper 16 L_NWT) — an SU(2) Skyrme field slaved to a C² doublet with a U(1)
  gauge + Higgs sector; `gauged_faddeev_model` and `n_from_doublet` recover the
  Skyrme field for the shared relax-then-ID.
- **Torus-knot seeds** — `torus_knot_hopfion` (T(p,q) hopfion, Q_H = p·m) and
  `flux_threaded_knot_seed` (the gauged-model knot IC).
- **Campaign `Provider` seam (F)** — a pluggable cloud-broker Protocol, so a new
  GPU cloud is a ~150-line adapter rather than a fork. Adds the `Provider`
  Protocol and shared `HostSpec` / `Offer` / `RentedHost` / `LaunchSpec` types
  plus `HostProbeFailed`. A **leak-proof `rent()`** — teardown guaranteed on
  every exit and independently verified, raising on a confirmed leak — is the
  contract invariant, not an implementation detail.
- **`RunPodProvider`** — a second reference adapter (RunPod), exercising the seam
  against a different cloud model: offers come from RunPod's GraphQL GPU-type
  catalog (offers are GPU *types*, not hosts), and CUDA-floor / bandwidth
  admission is applied at pod *create* (`allowedCudaVersions`, `minDownloadMbps`)
  rather than at selection. Absorbed with zero changes to the shared contract.
- `FakeProvider` contract test (zero-spend) covering leak-proof teardown on
  success, on exception, and on a bad host, plus the cheapest-first failover
  idiom.
- **Remote campaign execution (the Executor/D seam).** A closure can't cross a
  network, so remote workers receive a config (JSON) + a RunFn *by name*
  (`'module:function'`) + a work dir, and run the shared `execute_config`
  (factored out of the driver) — identical register/skip/resume/finish on every
  machine. Adds `campaign.remote` (`run_one`, `load_run_fn`), a
  `campaign.worker` CLI, and:
  - **`ModalExecutor`** — serverless fan-out via `Function.map`, a Modal
    Volume-backed registry; Modal owns the lifecycle (no host to rent or leak).
    `modal` is an optional dependency (imported only by `campaign.modal_exec`).
  - **`ProviderExecutor`** — runs a campaign over any `Provider` (Vast/RunPod):
    offers → rent with per-host failover → SSH the worker per config → sync
    artifacts back → leak-proof teardown. Generalizes the `run_eps_fleet` driver.
- **`campaign.multi`** — `run_multi` / `split_configs` drive one campaign across
  several executors at once (partition-and-merge). Content-addressed run identity
  (`config_hash`) makes the cross-provider harvest collision-free; a failing
  provider is isolated. `stream_multi` yields each provider's slice as it
  completes (a fast cloud isn't gated on the slowest); `run_multi(on_result=...)`
  observes them live.
- **`InProcessExecutor`** (`campaign.local_exec`) — runs configs on the local
  machine in-process via the shared `run_one`, the same `run(configs)->records`
  shape as the remote executors, so the local GPU joins a `run_multi` partition
  alongside the clouds (`split_configs([local, modal, vast, runpod])`).
- **Shared object-store backend** (`campaign.store`) — `ObjectStoreRunRegistry`
  (A/B) and `ObjectStoreEventSink` (C) over a minimal `BlobStore` (`MemoryBlobStore`
  for tests/single-process; `S3BlobStore` for S3/R2/GCS/MinIO, `boto3` optional).
  One store shared by every executor gives global `is_complete` (dedup +
  cross-provider/-restart resume) and one place to read results and the streamed
  ledger. One blob per record (no append) so concurrent writers never contend.

### Changed
- `VastClient` → **`VastProvider`**, now implementing the `Provider` Protocol:
  `offers()` takes a `HostSpec` and `rent()` takes a `LaunchSpec` (previously
  loose keyword arguments), and `Offer.id` is now a string. A `VastClient` alias
  is kept for import compatibility, but these method signatures changed.
- **`knots.identify_knot` / `identify_core_knot` resample default `600 → 200`.**
  600 still went combinatorial on jittery evolved curves when pyknotid's cython
  chelpers are absent (the common case); 200 resolves every torus knot through
  T(2,9) on noisy traces in <1 s. `identify_core_knot` now also forwards
  `max_points`. Raise it for genuinely high-crossing curves.

### Fixed
- **`knots.core_curves_from_n` now auto-detects the anti-vacuum pole** (new
  default `pole="auto"`: `-1` when `mean(n3) > 0`, else `+1` — a strict test, so
  the `mean(n3) == 0` tie maps deterministically to `+1`). It was hard-coded
  `pole=+1` (assumes
  vacuum at −z), but the library's own `torus_knot_hopfion` + `arrested_flow`
  leave the vacuum at +z — so the tracer seeded on the entire +z vacuum bulk
  (~millions of points) and hung for hours. The library was internally
  inconsistent: the fields it generates tripped the knot-ID it ships. Auto-detect
  is convention-agnostic (vacuum −z → +1, +z → −1), falls back to the opposite
  pole on a degenerate one-pole field, and raises `ValueError` for any pole other
  than `"auto"`/±1. Likely the root of the "census on evolved fields blocked"
  wall. (#19)

### Notes
- The seam is deliberately scoped to **rent-a-box container marketplaces**.
  Serverless backends (e.g. Modal) belong at the `Executor` (D) seam, not as
  Providers; VM marketplaces (e.g. TensorDock) are out of scope for the
  container-based `LaunchSpec`. See `CAMPAIGN.md`.

## [0.0.2]

- Baseline at which this changelog was introduced: the JAX/GPU Faddeev–Skyrme
  soliton engine and the campaign boundary protocols A–E (`RunRegistry`,
  `EventSink`, `Admission`, `Executor`) with local reference implementations.
  Earlier history is in the git log.

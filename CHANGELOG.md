# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project aims to follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
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
  provider is isolated.

### Changed
- `VastClient` → **`VastProvider`**, now implementing the `Provider` Protocol:
  `offers()` takes a `HostSpec` and `rent()` takes a `LaunchSpec` (previously
  loose keyword arguments), and `Offer.id` is now a string. A `VastClient` alias
  is kept for import compatibility, but these method signatures changed.

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

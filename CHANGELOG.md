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

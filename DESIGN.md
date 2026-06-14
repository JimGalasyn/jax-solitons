# Design principles

jax-solitons is built to run **campaigns** — 10^4–10^6 registered,
restartable runs across rented fleets — not demos. Every API and every PR
gets checked against these principles. They are numbered so reviews can
cite them.

## P1. One Model abstraction, composable terms

A field theory is a configuration — `(state, energy terms, constraint,
charges)` — not a new solver. Hybrid theories (gauged Faddeev, GPE+Skyrme)
are term lists. If adding a physics feature means writing a new stepper or
copying a relaxer, the abstraction has failed.

## P2. Batch-first state

The leading batch axis is free (`vmap`), and CI proves batched dynamics ≡
per-sample. Caveat from measurement (2026-06-12, RTX 3090, N=160 fp32):
batching pays **only when single runs underfill the device** — at
bandwidth saturation it is 0.90x. Throughput claims in this repo are
measured, never assumed.

## P3. Stencil-local canonical path

Forward-difference E2 and plaquette E4 shard with halo exchange only;
spectral/split-step paths are single-device conveniences, never
load-bearing for scale. Discretizations are chosen for **exactness
properties first** (see VALIDATION.md — the area form is exactly
quantized; the naive form is not) and shardability second. Speed never
buys a topology-unsafe stencil.

## P4. Registered, restartable runs

Every run is a `RunConfig` (hashed, serialized into every output, one
manifest line). Checkpoints carry FULL integrator state; restart is
bit-identical at fixed dtype/devices. A result that cannot name its
config hash does not exist. This is what makes spot/interruptible fleets
free capacity instead of risk.

## P5. Explicit precision

dtype is a `BoxGrid` parameter, never a global flag. The protocol is
**hunt fp32, certify x64** (measured: fp32 is ~4.4x faster and endpoint-
identical to 4 significant figures on these bandwidth-bound workloads —
but physics claims are certified in x64). The 4-sig-fig figure is for
descent endpoints and short integration; long real-time Hamiltonian /
Kibble–Zurek-length evolutions accumulate fp32 phase error that step-wise
conservation hides, so time-resolved dynamics are certified in x64.

## P6. Event records, not raw fields

At campaign scale the product of a run is a small record: config, charge
and energy ledgers, censuses, core-curve polylines. Full fields are kept
only for triggered events. Observables stream during the run; "hold all
snapshots" patterns are bugs.

## P7. Diagnostics are triaged, and never trusted mid-bath

Cheap classifiers (conservation ledgers, blob counts) run on every event;
expensive identification (tracing, knot ID) runs on flagged events, and
ONLY after a quench — descent cannot create topology, so relax-then-ID
is faithful where in-bath tracing is not (a measured in-run false-decay,
2026-06-12, is the standing case study; see VALIDATION.md known-limits).

## P8. The exactness contract

Tier-1 tests assert mathematical identities (quantization, symmetries,
constraint algebra) on adversarial random inputs. Any tolerance wider
than float rounding on a Tier-1 test is a design smell. New
discretizations ship with their exactness tests or they don't ship.

## P9. Fail loudly at the boundary

Hosts, networks, and devices lie (measured: a 0.996-reliability host with
zero outbound bandwidth). Anything that touches infrastructure probes
first, writes what it measured, and bails early — never "runs anyway" on
unverified capacity.

## P10. Measured costs drive the roadmap

Optimization targets come from profiles of real campaigns, not intuition.
Current measured cost center: identification (CPU tracing), not GPU
dynamics. The roadmap (TODO.md) cites measurements next to every
scaling item.

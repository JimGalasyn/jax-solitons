"""Skyrme B=2 binding convergence fleet: the resolution ladder dx -> 0.

Thin caller over jax_solitons.campaign.FleetExecutor (cf. run_soft_link_fleet.py):
one leg per resolution N, each shipping skyrme_converge_batch.py and relaxing BOTH
the B=1 hedgehog and the B=2 rational-map torus at that N (so the two legs share
the box/resolution -> a clean per-N binding). There is NO IC file -- the seeds are
generated on-box. Unlike the soft-link farm, there is NO local knot-ID step: the
baryon number is the exact on-box charge, so each manifest carries the full per-N
result; collate the ladder with skyrme_converge_analyze.py.

Uses vast/onstart_skyrme.sh (installs jax-solitons from the skyrme-module branch,
which carries models/skyrme.py) instead of the default onstart.sh (main, no skyrme).

Why the farm: the local calibration (docs/SKYRME.md) showed the known ~4.3%
binding sits below the dx~0.18 resolution floor (both legs land at E/bound~1.16
vs ideal 1.232/1.179). The ladder N=64..160 at L=8 pushes dx 0.125 -> 0.05 to ask
whether BOTH legs converge to E/bound -> 1.232/1.179 (above the Bogomolny bound)
and the binding -> ~4.3%. Skyrme fp64 fields are small (4*N^3*8 B; ~0.7 GB at
N=160), so a 24 GB box is ample and cheap.

  # validate the path on one box (smallest N):
  python run_skyrme_converge_fleet.py --N 64 --max-parallel 1
  # the resolution ladder (does binding converge to ~4% as dx -> 0?):
  python run_skyrme_converge_fleet.py --N 64 96 128 160 --max-parallel 4
  python run_skyrme_converge_fleet.py --N 64 96 128 160 --dry-run   # legs, no spend

Then locally: python skyrme_converge_analyze.py output/skyrme_converge_fleet/N*/out_sk
"""
import argparse
from pathlib import Path

# Ported on migration (2026-08-01): this read `from jax_solitons.campaign import ...`,
# but that subpackage was extracted to run-farm in 0.0.8, so the import had been dead
# since then. The symbols moved without renaming; only their home changed.
from run_farm.fleet import FleetExecutor, FleetLeg, SentinelReady
from run_farm.protocols import HostSpec, LaunchSpec
from run_farm.vast import VastLedger, VastProvider

HERE = Path(__file__).resolve().parent
IMG = "nvidia/cuda:12.2.2-runtime-ubuntu22.04"
ONSTART = (HERE / "onstart_skyrme.sh").read_text()
DRIVER = HERE / "skyrme_converge_batch.py"
PYBOX = "/workspace/jaxenv/bin/python"            # built by vast/onstart_skyrme.sh


def build_legs(args) -> list[FleetLeg]:
    """One leg per N: ship the driver, relax B=1+B=2, fetch out_sk."""
    legs = []
    for N in args.N:
        command = (
            f"{PYBOX} skyrme_converge_batch.py --N {N} --L {args.L} "
            f"--c2 {args.c2} --c4 {args.c4} --dt {args.dt} --block {args.block} "
            f"--max-steps {args.max_steps} --r0-1 {args.r0_1} --r0-2 {args.r0_2} "
            f"--only both --out out_sk")
        legs.append(FleetLeg(
            label=f"N{N}", command=command, ship=(str(DRIVER),),
            fetch="out_sk", done_when="out_sk/manifest.json"))
    return legs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, nargs="+", required=True,
                    help="resolution(s); one rented GPU per N")
    ap.add_argument("--L", type=float, default=16.0, help="L=16 fits r0*~2 (c4=1)")
    ap.add_argument("--c2", type=float, default=1.0)
    ap.add_argument("--c4", type=float, default=1.0, help="c4=1 -> r0*~2, fits L=16")
    ap.add_argument("--dt", type=float, default=1e-3)
    ap.add_argument("--block", type=int, default=600)
    ap.add_argument("--max-steps", type=int, default=12000)
    ap.add_argument("--r0-1", type=float, default=2.0)
    ap.add_argument("--r0-2", type=float, default=2.6)
    ap.add_argument("--max-parallel", type=int, default=4)
    ap.add_argument("--run-timeout", type=int, default=3600)
    ap.add_argument("--max-dph", type=float, default=0.45)
    ap.add_argument("--gpu", default="RTX_3090", help="24 GB-class is ample for fp64")
    ap.add_argument("--out", default="output/skyrme_converge_fleet")
    ap.add_argument("--dry-run", action="store_true",
                    help="build + print the legs (resume status too) without renting")
    args = ap.parse_args()

    legs = build_legs(args)
    outdir = HERE / args.out

    if args.dry_run:                                  # fully offline: no key, no API
        print(f"{len(legs)} legs (N), {args.max_parallel} parallel -> {outdir}; "
              f"gpu={args.gpu} max_dph={args.max_dph} L={args.L} c4={args.c4} "
              f"dt={args.dt} max_steps={args.max_steps}")
        for leg in legs:
            done = (outdir / leg.label / leg.marker()).exists()
            print(f"  {leg.label}: {'COMPLETE (would skip)' if done else 'pending'}")
            print(f"      {leg.command}")
        return

    ledger = VastLedger(outdir / "vast_ledger.jsonl")
    provider = VastProvider(ledger=ledger)
    launch = LaunchSpec(image=IMG, onstart=ONSTART, disk_gb=24, label="skyrme-converge")
    spec = HostSpec(gpu_name=args.gpu, max_dph=args.max_dph, min_reliability=0.97,
                    min_inet_mbps=300, min_cuda=12.2)
    ex = FleetExecutor(provider, launch, local_out_dir=str(outdir), host_spec=spec,
                       ready=SentinelReady(), max_parallel=args.max_parallel,
                       run_timeout=args.run_timeout, ledger=ledger)
    results = ex.run(legs)
    print("=" * 60)
    for r in sorted(results, key=lambda r: r.label):
        print(f"  {r.label}: {r.status}" + (f"  ({r.detail})" if r.detail else ""))
    print(f"ledger: {ledger.path}")
    print(f"\nNext (LOCAL): python skyrme_converge_analyze.py {outdir}/N*/out_sk")


if __name__ == "__main__":
    main()

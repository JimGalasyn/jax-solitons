#!/usr/bin/env python3
"""LOCAL collation for the Skyrme convergence farm legs.

Unlike the soft-link farm there is no field-tracing step: the baryon number is
the exact on-box charge, so each leg's manifest.json already carries the full
per-N result. This just reads the fetched manifests and prints the resolution
ladder + the convergence read:

    N   dx     E1/bound  E2/bound   binding%   B1  B2  phys

The decisive read: as dx -> 0 do BOTH legs converge to the ideal E/bound
(1.232 for B=1, 1.179 for B=2) from ABOVE the Bogomolny bound (physical=True,
no sub-bound collapse), and does the binding converge to the published ~4.3%?
A leg flagged sub-bound / physical=False is under-resolved at that dx -- its
number is not yet trustworthy.

  python skyrme_converge_analyze.py output/skyrme_converge_fleet/N*/out_sk
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

IDEAL_B1 = 1.232   # E(B=1)/bound, massless Skyrme
IDEAL_B2 = 1.179   # E(B=2)/(2*bound(1)) = 1.232*0.957; bound(2)=2*bound(1)
IDEAL_BINDING = 4.3  # percent


def load(path: Path) -> dict | None:
    m = path / "manifest.json" if path.is_dir() else path
    if not m.exists():
        return None
    return json.loads(m.read_text())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+", help="leg out_sk dirs (or manifest.json paths)")
    args = ap.parse_args()

    rows = []
    for d in args.dirs:
        m = load(Path(d))
        if m is None:
            print(f"  (skip, no manifest: {d})")
            continue
        rows.append(m)
    rows.sort(key=lambda m: m["N"])

    print(f"\n{'N':>5} {'dx':>7} {'E1/bnd':>8} {'r1':>6} {'E2/bnd':>8} {'r2':>6} "
          f"{'bind%':>8} {'B1':>3} {'B2':>3} {'phys':>5}  notes")
    print("-" * 86)
    for m in rows:
        r = m.get("results", {})
        b1 = r.get("b1", {}); b2 = r.get("b2", {})
        e1b = b1.get("E_over_bound", float("nan"))
        e2b = b2.get("E_over_bound", float("nan"))
        r1 = b1.get("ratio", float("nan")); r2 = b2.get("ratio", float("nan"))
        bind = 100.0 * m.get("binding_frac", float("nan"))
        B1 = b1.get("B", float("nan")); B2 = b2.get("B", float("nan"))
        phys = b1.get("physical", False) and b2.get("physical", False)
        notes = []
        if b1.get("sub_bound_seen") or b2.get("sub_bound_seen"):
            notes.append("collapse-reachable")
        if max(abs(r1 - 1), abs(r2 - 1)) > 0.25:
            notes.append("poor-virial-capture")
        if not phys:
            notes.append("UNDER-RESOLVED")
        print(f"{m['N']:>5} {m['dx']:>7.4f} {e1b:>8.3f} {r1:>6.2f} {e2b:>8.3f} "
              f"{r2:>6.2f} {bind:>8.2f} {B1:>+3.0f} {B2:>+3.0f} {str(phys):>5}  "
              f"{' '.join(notes)}")

    print("-" * 72)
    print(f"ideal (massless Skyrme):  E1/bnd={IDEAL_B1}  E2/bnd={IDEAL_B2}  "
          f"binding={IDEAL_BINDING}%")
    if len(rows) >= 2:
        finest = rows[-1]
        bind = 100.0 * finest.get("binding_frac", float("nan"))
        print(f"finest N={finest['N']} (dx={finest['dx']:.4f}): "
              f"binding={bind:+.2f}%  "
              f"(toward {IDEAL_BINDING}%? watch the trend as dx -> 0)")


if __name__ == "__main__":
    main()

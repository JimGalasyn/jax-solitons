"""EHN two-scalar cross-linking: Lk(φ₁-string, φ₂-string) — the baryon linking number.

The ⟨Lk⟩ tracer (gpe_vortex_topology) does INTER-loop linking within ONE complex field.
EHN's baryon number is the linking BETWEEN the two different strings (φ₁ gauged, φ₂ global).
This extends the tracer: extract both skeletons (plaquette phase-winding), Gauss-link them.

Used to disambiguate the R-scan: does the low-E small-R config still carry the link
(Lk≈−4, a valid compact knot) or has it unlinked (Lk≈0, link% was right)?

  python ehn_cross_linking.py output/ehn_rscan/*/out_ehn_relax/field.npz
"""
import numpy as np

# The two sys.path inserts that used to sit here ("on-box: everything shipped
# flat" / "local: gpe_vortex_topology in parent") are gone: this is a package
# module now, so the import resolves the same way in every layout. Those inserts
# were the flat-shipping workaround, and their absence from standard_box.py is
# what killed the first rented B2 attempt before it ran a single leg.
from ..vortex_topology import vortex_skeleton, _gauss, _label_lines


def cross_linking(psi1, psi2, dx):
    """Total Gauss linking between the φ₁ and φ₂ vortex-core sets (= EHN baryon #)."""
    P1, T1, C1 = vortex_skeleton(psi1)
    P2, T2, C2 = vortex_skeleton(psi2)
    if not len(P1) or not len(P2):
        return float("nan"), len(P1), len(P2)
    sh = np.array(psi1.shape) / 2.0
    lk = _gauss((P1 - sh) * dx, T1, (P2 - sh) * dx, T2, dx)
    return lk, len(P1), len(P2)


def main(paths):
    print(f"{'field':>28} {'dx':>5} {'nseg₁':>6} {'nseg₂':>6} {'Lk(φ₁,φ₂)':>10}")
    for p in paths:
        d = np.load(p)
        u = d["u"]                        # (10,N,N,N): p1r,p1i,p2r,p2i,...
        N = u.shape[1]
        L = N * 0.8                        # dx=0.8 fixed (EHN); L=N·dx
        dx = 0.8
        psi1 = u[0] + 1j * u[1]
        psi2 = u[2] + 1j * u[3]
        lk, n1, n2 = cross_linking(psi1, psi2, dx)
        tag = p.split("/")[-3][:28] if "/" in p else p
        print(f"{tag:>28} {dx:>5.2f} {n1:>6} {n2:>6} {lk:>+10.3f}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])

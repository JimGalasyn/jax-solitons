"""The EHN real-time engine: is it actually conservative, and is the lock real?

`relax.py` descends; `quench.py` evolves. A descent engine has an easy correctness
story -- the energy must go down -- and a real-time engine has a hard one, because
its energy must go NOWHERE. These tests are built around the two ways that check
gets faked:

  1. measuring the wrong energy, which is how a conservative engine can look
     dissipative (and, worse, look it CONSISTENTLY -- see the dt-independence
     test, which is the trap that actually bit during the port);
  2. shipping a lock that changes nothing, which passes every conservation test
     precisely because it does not participate.

The grids are small (N = 24) and the horizons short. These are contract tests on
the integrator and the wiring, not physics runs.
"""
import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

from jax_solitons.ehn import quench as Q                        # noqa: E402
from jax_solitons.ehn.knot_batch import build_ic                # noqa: E402

N, L = 24, 6.0
LAM, KAPPA = 50.0, 0.5


@pytest.fixture(scope="module")
def ic():
    """A built EHN link plus a zero gauge sector, in x64."""
    dx = L / N
    kv = Q.kvecs(N, L)
    phi1, phi2 = build_ic(N, L, nlink=2, R=1.4, core=0.5)
    z = jnp.zeros((N, N, N))
    return dict(kv=kv, dx=dx, phi1=phi1, phi2=phi2, z=z)


def _run(ic, *, dt, steps, **kw):
    z = ic["z"]
    return Q.evolve(ic["phi1"], ic["phi2"], z, z, z, z, z, z, ic["kv"],
                    steps=steps, dt=dt, dx=ic["dx"], lam=LAM, kappa=KAPPA, **kw)


def _dH_over_H(ic, dt, T=0.02, **kw):
    steps = int(round(T / dt))
    _, s = _run(ic, dt=dt, steps=steps, sample_every=steps, **kw)
    return (s[-1]["E_total"] - s[0]["E_total"]) / s[0]["E_total"]


# -- 1. the integrator ------------------------------------------------------
def test_energy_drift_is_second_order_in_dt(ic):
    """With gamma = eta = C_l3 = 0 the engine is Hamiltonian, so the only energy
    change is truncation error -- and halving dt must quarter it.

    Asserting merely "dH is small" would pass for a dissipative engine at small
    dt. The ORDER is what distinguishes integrator error from physics."""
    coarse = abs(_dH_over_H(ic, 2e-3, C_l3=0.0))
    fine = abs(_dH_over_H(ic, 1e-3, C_l3=0.0))
    assert fine < coarse, f"drift did not shrink with dt: {coarse:.2e} -> {fine:.2e}"
    ratio = coarse / fine
    assert 3.0 < ratio < 5.0, f"expected ~4x (2nd order), got {ratio:.2f}"
    assert fine < 1e-3, f"drift too large even at dt=1e-3: {fine:.2e}"


def test_the_ehn_normalised_energy_is_not_the_invariant(ic):
    """A regression guard on the trap, not on the fix.

    `knot_batch.two_scalar_energy` is EHN-normalised -- no 1/2 on the covariant
    gradients, matching EHN's STATIC Eq.10 -- while this engine integrates
    `i d_t phi = -1/2 D^2 phi + ...`. Measuring the run with the un-halved
    functional reports a ~3% loss that does NOT shrink with dt, so it survives a
    convergence check and reads as physical dissipation.

    This pins the distinction so that "simplifying" total_energy to reuse
    two_scalar_energy fails here instead of in someone's results."""
    def unhalved(p1, p2, Ax, Ay, Az, Ex, Ey, Ez, kv, dx):
        c = Q.total_energy(p1, p2, Ax, Ay, Az, Ex, Ey, Ez, kv, dx, LAM, KAPPA,
                           components=True)
        return c["total"] + c["grad1"] + c["grad2"]      # i.e. gradients un-halved

    drifts = []
    for dt in (2e-3, 1e-3):
        steps = int(round(0.02 / dt))
        (p1, p2, Ax, Ay, Az, Ex, Ey, Ez, _), _ = _run(ic, dt=dt, steps=steps,
                                                      C_l3=0.0)
        z = ic["z"]
        h0 = unhalved(ic["phi1"], ic["phi2"], z, z, z, z, z, z, ic["kv"], ic["dx"])
        h1 = unhalved(p1, p2, Ax, Ay, Az, Ex, Ey, Ez, ic["kv"], ic["dx"])
        drifts.append((h1 - h0) / h0)

    assert all(d < -1e-3 for d in drifts), \
        f"the wrong functional should look lossy; got {drifts}"
    # and the damning part: refining dt does not help it
    assert abs(drifts[0] - drifts[1]) < 0.2 * abs(drifts[0]), \
        f"wrong-functional drift should be dt-INdependent, got {drifts}"


def test_skyrmion_charge_drift_is_a_resolution_artifact(ic):
    """Q = pi_3(S^3) degree is topological, so the integrator must not leak it --
    but on a lattice Q is only approximately an integer, and the honest test is
    about which of the two you are looking at.

    Measured at fixed dt over fixed T, the initial error |Q0 + 2| and the drift
    over the run BOTH fall by ~an order of magnitude per resolution step
    (N = 24, 32, 48 -> err 6.3e-3, 6.8e-4, 2.6e-5; drift 3.1e-3, 3.4e-4,
    6.7e-6). That is the degree ESTIMATOR converging, not charge leaking: a real
    leak would survive refinement. Asserting a fixed small drift at one N would
    have encoded the artifact as the specification."""
    drifts = {}
    for n in (24, 32):
        ll = 6.0
        dx = ll / n
        kv = Q.kvecs(n, ll)
        p1, p2 = build_ic(n, ll, nlink=2, R=1.4, core=0.5)
        z = jnp.zeros((n, n, n))
        _, s = Q.evolve(p1, p2, z, z, z, z, z, z, kv, steps=20, dt=1e-3, dx=dx,
                        lam=LAM, kappa=KAPPA, C_l3=0.0, sample_every=20)
        drifts[n] = (abs(s[0]["Q"] + 2.0), abs(s[-1]["Q"] - s[0]["Q"]))

    assert drifts[24][0] < 0.05, f"IC should carry Q = -2, got err {drifts[24][0]:.2e}"
    assert drifts[32][0] < 0.3 * drifts[24][0], \
        f"|Q+2| did not converge with N: {drifts}"
    assert drifts[32][1] < 0.3 * drifts[24][1], \
        f"Q drift did not shrink with N -- that is a leak, not an artifact: {drifts}"


# -- 2. the lock ------------------------------------------------------------
def test_locked_step_reduces_to_the_bare_step_when_the_lock_is_off(ic):
    """C_l3 = 0 must make `step_locked` bit-for-bit `step`.

    The bare census path is the control arm of the R-C-LC-1 triptych: if turning
    the lock off does not recover exactly the bare engine, the control and the
    treatment differ by something nobody chose."""
    z, p1, p2, kv, dx = ic["z"], ic["phi1"], ic["phi2"], ic["kv"], ic["dx"]
    s0 = jnp.zeros_like(jnp.real(p1))
    #             fields..................  kv  dt    g    mu5  eta  agrad
    bare = Q.step(p1, p2, z, z, z, z, z, z, kv, 1e-3, 1.0, 0.0, 0.0, "wrapped",
                  0.0, LAM, KAPPA, dx, 0.0, 1e-3)          # gamma lam kap dx C eps
    locked = Q.step_locked(p1, p2, z, z, z, z, z, z, s0, kv, 1e-3, 1.0, 0.0,
                           0.0, "wrapped", 0.0, LAM, KAPPA, 1, dx, 0.0, 1e-3)
    for a, b, name in zip(bare, locked[:8], "p1 p2 Ax Ay Az Ex Ey Ez".split()):
        assert jnp.allclose(a, b, rtol=0, atol=0), f"{name} differs with C_l3=0"


def test_the_lock_actually_moves_phi2(ic):
    """A lock that changes nothing passes every other test in this file.

    With C_l3 > 0 the L3 force is `-alpha_l3 * dE_L3/dphi2*`, so phi2 must differ
    from the C_l3 = 0 run. Uses a deliberately large alpha_l3 so one step is
    visible -- this asks whether the term is WIRED, not whether it is tuned."""
    z = ic["z"]
    s0 = jnp.zeros_like(jnp.real(ic["phi1"]))
    common = dict(kv=ic["kv"], dt=1e-3, dx=ic["dx"])

    def one(C_l3):
        return Q.step_locked(ic["phi1"], ic["phi2"], z, z, z, z, z, z, s0,
                             ic["kv"], 1e-3, 1.0, 0.0, 0.0, "wrapped", 0.0,
                             LAM, KAPPA, 4, ic["dx"], C_l3, 1e-3, 2e-3, 1.0)
    off, on = one(0.0), one(400.0)
    d = float(jnp.max(jnp.abs(on[1] - off[1])))
    assert d > 1e-12, "C_l3 > 0 left phi2 untouched -- the L3 force is not wired"
    # ...and it is the phi2 sector specifically: phi1 carries no L3 force.
    assert jnp.allclose(on[0], off[0], rtol=0, atol=0), \
        "phi1 changed; the L3 force should reach phi2 only"


def test_wrapped_and_bilinear_are_different_locks(ic):
    """`agrad` selects THE LOCK vs the modulus-suppressed form. The triptych
    turned on these differing, so a flag that silently aliased them would void
    the comparison the gate is built from."""
    z = ic["z"]
    s0 = jnp.zeros_like(jnp.real(ic["phi1"]))

    def one(agrad):
        return Q.step_locked(ic["phi1"], ic["phi2"], z, z, z, z, z, z, s0,
                             ic["kv"], 1e-3, 1.0, 0.0, 0.0, agrad, 0.0,
                             LAM, KAPPA, 4, ic["dx"], 400.0, 1e-3, 2e-3, 1.0)
    w, b = one("wrapped"), one("bilinear")
    assert float(jnp.max(jnp.abs(w[1] - b[1]))) > 1e-12, \
        "wrapped and bilinear produced identical phi2 -- agrad is not threaded"


# -- 3. dissipation is opt-in ----------------------------------------------
def test_damping_is_off_by_default_and_costs_energy_when_on(ic):
    """gamma defaults to 0 (this module's whole correctness gate depends on it).
    Turning it on must actually remove energy -- otherwise the flag is inert."""
    free = _dH_over_H(ic, 1e-3, C_l3=0.0)
    damped = _dH_over_H(ic, 1e-3, C_l3=0.0, gamma=0.5)
    assert damped < free - 1e-6, \
        f"gamma=0.5 did not dissipate: free={free:.2e} damped={damped:.2e}"

"""Split-step Fourier stepper for the GPE (Strang splitting).

Imaginary time (relaxation):
    psi <- exp(-dt/2 g (|psi|^2 - 1)) psi      (half nonlinear)
    psi <- ifft( exp(-dt k^2 / 2) fft(psi) )   (full kinetic)
    psi <- exp(-dt/2 g (|psi|^2 - 1)) psi      (half nonlinear)

Real time uses the same splitting with -i dt. Spectral kinetics make this a
single-device stepper; it rides jaxDecomp's distributed FFT when sharded.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from jax_solitons.grid import BoxGrid


def make_splitstep(grid: BoxGrid, dt: float, *, g: float = 1.0,
                   imaginary_time: bool = True):
    """Build a jitted GPE split-step psi -> psi."""
    _, _, _, K2 = grid.k_vectors()
    if imaginary_time:
        kin = jnp.exp(-0.5 * K2 * dt)

        @jax.jit
        def step(psi):
            psi = psi * jnp.exp(-0.5 * dt * g * (jnp.abs(psi) ** 2 - 1.0))
            psi = jnp.fft.ifftn(jnp.fft.fftn(psi) * kin)
            psi = psi * jnp.exp(-0.5 * dt * g * (jnp.abs(psi) ** 2 - 1.0))
            return psi
    else:
        kin = jnp.exp(-0.5j * K2 * dt)

        @jax.jit
        def step(psi):
            psi = psi * jnp.exp(-0.5j * dt * g * (jnp.abs(psi) ** 2 - 1.0))
            psi = jnp.fft.ifftn(jnp.fft.fftn(psi) * kin)
            psi = psi * jnp.exp(-0.5j * dt * g * (jnp.abs(psi) ** 2 - 1.0))
            return psi

    return step


def splitstep_evolve(grid: BoxGrid, psi, *, dt: float, steps: int,
                     g: float = 1.0, imaginary_time: bool = True,
                     observe_every: int = 0, observer=None):
    """Evolve psi for `steps` split-steps; collect observer returns."""
    step = make_splitstep(grid, dt, g=g, imaginary_time=imaginary_time)
    obs = []
    for i in range(steps):
        if observer and observe_every and (i % observe_every == 0):
            obs.append(observer(i, psi))
        psi = step(psi)
    if observer:
        obs.append(observer(steps, psi))
    return psi, obs

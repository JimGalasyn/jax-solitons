#!/bin/bash
# Vast.ai worker bootstrap for jax-solitons fleet runs.
#
# Rebuilt 2026-06-13: the old path (probe GitHub speed; `pip install jax[cuda12]`
# from PyPI) bit the RUNBOOK's documented #1 gotcha -- PyPI/Fastly AND GitHub are
# throttled on Vast datacenter IPs (~KB/s), while Cloudflare/CDN run at MB/s on the
# SAME hosts. So the GitHub probe FALSE-REJECTED good hosts and the PyPI install
# HUNG. This version uses the delivery paths that actually work:
#   - JAX+CUDA via conda-forge (anaconda.org rides Cloudflare, tens of MB/s)
#   - miniconda installer from repo.anaconda.com (CDN), since the nvidia/cuda
#     image has no conda
#   - probe the CONDA CDN (the path we use), not GitHub
# The engine (jax-solitons) is small, so pip-from-GitHub is tolerable.
# Env lands at /workspace/jaxenv -> driver runs /workspace/jaxenv/bin/python.
set -uo pipefail
mkdir -p /workspace
L=/workspace/onstart.log
log(){ echo "$(date +%T) $*" >> "$L"; }
log "onstart start"

MC=https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

# 1. Probe the conda CDN (what we actually pull from). Bail on a throttled host.
SPD=$(curl -s -o /dev/null -w "%{speed_download}" --max-time 15 -L "$MC")
log "conda-cdn ${SPD}B/s"
if [ "${SPD%.*}" -lt 2000000 ]; then
    touch /tmp/worker-bad-network; log "BAD NET (conda cdn < 2MB/s)"; exit 1
fi

# 2. Miniconda + mamba (fast solver)
curl -s -L "$MC" -o /tmp/mc.sh
bash /tmp/mc.sh -b -p /workspace/conda >> "$L" 2>&1
source /workspace/conda/etc/profile.d/conda.sh
# Accept Anaconda channel ToS. Since 2024 conda HARD-FAILS any solve that even
# references the defaults channels (pkgs/main, pkgs/r) until their ToS is
# accepted (CondaToSNonInteractiveError). We only pull from conda-forge, but the
# Miniconda base env has defaults configured, so `conda install ... mamba` trips
# the gate and aborts -> jaxenv never builds. Accept once, non-interactively.
# (Measured 2026-06-14: this was the real onstart failure, NOT the IP throttle --
#  the same host clocked 109 MB/s to the conda CDN.)
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main >> "$L" 2>&1 || log "tos main warn"
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r    >> "$L" 2>&1 || log "tos r warn"
conda install -n base -y -c conda-forge mamba >> "$L" 2>&1 || log "mamba install warn"

# 3. JAX + CUDA from conda-forge (Cloudflare), into a prefix env
SOLVER=mamba; command -v mamba >/dev/null || SOLVER=conda
# NB: `pip` MUST be in this list -- step 4 installs the engine via
# /workspace/jaxenv/bin/pip, which conda does NOT provide unless asked.
$SOLVER create -y -p /workspace/jaxenv -c conda-forge \
    "jaxlib=*=*cuda*" jax scipy numpy pip >> "$L" 2>&1

# 4. The engine (small; GitHub pip is tolerable)
/workspace/jaxenv/bin/pip install -q \
    "git+https://github.com/JimGalasyn/jax-solitons@skyrme-module" >> "$L" 2>&1

# 5. Verify JAX sees the GPU, then signal ready (else bail loudly)
if /workspace/jaxenv/bin/python -c \
    "import jax; d=jax.devices(); print(d); assert d and d[0].platform=='gpu'" >> "$L" 2>&1; then
    touch /tmp/worker-ready; log "worker ready (GPU ok)"
else
    touch /tmp/worker-bad-network; log "jax/GPU verify FAILED"; exit 1
fi

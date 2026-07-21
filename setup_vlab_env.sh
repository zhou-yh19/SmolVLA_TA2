#!/usr/bin/env bash

set -Eeuo pipefail

ENV_NAME="${ENV_NAME:-vlab}"
CONDA_CHANNEL="${CONDA_CHANNEL:-https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/}"
PYPI_INDEX_URL="${PYPI_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
PIP_TIMEOUT="${PIP_TIMEOUT:-300}"

on_error() {
    local exit_code=$?
    echo "[setup] Failed at line ${BASH_LINENO[0]} (exit ${exit_code})." >&2
    exit "${exit_code}"
}
trap on_error ERR

if ! command -v conda >/dev/null 2>&1; then
    echo "[setup] conda was not found in PATH. Install Miniconda/Miniforge first." >&2
    exit 1
fi

CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"

echo "[setup] Conda: $(command -v conda)"
echo "[setup] Environment: ${ENV_NAME}"

if conda env list | awk -v name="${ENV_NAME}" '$1 == name { found = 1 } END { exit !found }'; then
    echo "[setup] Conda environment '${ENV_NAME}' already exists; reusing it."
else
    echo "[setup] Creating Conda environment '${ENV_NAME}'..."
    conda create -y -n "${ENV_NAME}" \
        python=3.11 \
        pip \
        ffmpeg \
        av=14.4 \
        --solver=libmamba \
        --override-channels \
        -c "${CONDA_CHANNEL}"
fi

echo "[setup] Installing PyTorch CUDA 12.8 wheels..."
conda run --no-capture-output -n "${ENV_NAME}" \
    python -m pip install \
    torch==2.7.1 \
    torchvision==0.22.1 \
    torchaudio==2.7.1 \
    --index-url "${PYTORCH_INDEX_URL}" \
    --progress-bar on \
    --timeout "${PIP_TIMEOUT}"

echo "[setup] Installing VLAb Python dependencies..."
conda run --no-capture-output -n "${ENV_NAME}" \
    python -m pip install \
    "transformers==4.53.3" \
    "accelerate==1.7.0" \
    "datasets==3.6.0" \
    "peft==0.15.2" \
    "num2words>=0.5.14,<0.6" \
    "numpy==1.26.4" \
    "safetensors>=0.4.3,<1" \
    "draccus==0.10.0" \
    "huggingface-hub[hf-transfer,cli]>=0.30.0,<1" \
    "termcolor>=2.4.0,<4" \
    "einops>=0.8.0,<1" \
    "imageio[ffmpeg]>=2.34.0,<3" \
    "jsonlines>=4.0.0,<5" \
    "deepdiff>=7.0.1,<9" \
    "packaging>=24.2" \
    "wandb>=0.16.3,<1" \
    "trackio>=0.7.0,<1" \
    "pyyaml>=6,<7" \
    "requests>=2.32.2,<3" \
    "brotli>=1,<2" \
    "sympy==1.14.0" \
    "mpmath==1.3.0" \
    --index-url "${PYPI_INDEX_URL}" \
    --progress-bar on \
    --timeout "${PIP_TIMEOUT}"

echo "[setup] Verifying imports and versions..."
conda run --no-capture-output -n "${ENV_NAME}" python - <<'PY'
import sys

import accelerate
import av
import datasets
import numpy
import peft
import torch
import torchaudio
import torchvision
import transformers

expected = {
    "torch": "2.7.1",
    "torchvision": "0.22.1",
    "torchaudio": "2.7.1",
    "transformers": "4.53.3",
    "accelerate": "1.7.0",
    "datasets": "3.6.0",
    "peft": "0.15.2",
    "numpy": "1.26.4",
}
actual = {
    "torch": torch.__version__.split("+")[0],
    "torchvision": torchvision.__version__.split("+")[0],
    "torchaudio": torchaudio.__version__.split("+")[0],
    "transformers": transformers.__version__,
    "accelerate": accelerate.__version__,
    "datasets": datasets.__version__,
    "peft": peft.__version__,
    "numpy": numpy.__version__,
}
for package, expected_version in expected.items():
    actual_version = actual[package]
    if actual_version != expected_version:
        raise RuntimeError(f"{package}: expected {expected_version}, got {actual_version}")

print(f"Python: {sys.version.split()[0]}")
print(f"PyAV: {av.__version__}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA runtime: {torch.version.cuda}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
PY

cat <<EOF

[setup] Environment '${ENV_NAME}' is ready.

Activate it in the current shell with:
  conda activate ${ENV_NAME}

Then expose this repository's source package with:
  export PYTHONPATH="\${PWD}/src:\${PYTHONPATH:-}"

Verify the repository with:
  python tests/test_installation.py
EOF

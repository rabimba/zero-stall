#!/usr/bin/env bash
# Phase 0 Environment Bootstrapper for Cloud TPU v7 (Ironwood / tpu7x) VMs.
# Inspects currently installed packages FIRST (Execution Contract #2) and only clones/installs
# missing upstream components into third_party/ when needed.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
THIRD_PARTY="${REPO_ROOT}/third_party"
mkdir -p "${THIRD_PARTY}"

echo "=== [Bootstrap] Inspecting existing Python environment ($(python3 --version)) ==="
python3 -m pip install -e "${REPO_ROOT}[tpu]" --quiet || python3 -m pip install -e "${REPO_ROOT}"

check_mod() {
    python3 -c "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('$1') else 1)"
}

if check_mod torch_tpu; then
    echo "[Bootstrap] Found installed torch_tpu."
else
    echo "[Bootstrap] WARNING: torch_tpu not found in active Python env. Ensure TPU VM base virtualenv/conda env is active."
fi

if check_mod verl; then
    echo "[Bootstrap] Found installed verl."
else
    echo "[Bootstrap] Cloning and installing verl-project/verl..."
    if [[ ! -d "${THIRD_PARTY}/verl" ]]; then
        git clone --depth 1 https://github.com/verl-project/verl.git "${THIRD_PARTY}/verl"
    fi
    python3 -m pip install --no-deps -e "${THIRD_PARTY}/verl"
fi

if check_mod torchtitan; then
    echo "[Bootstrap] Found installed torchtitan."
else
    echo "[Bootstrap] Cloning and installing google-pytorch/torchtitan (Ironwood recipes + SimpleFSDP)..."
    if [[ ! -d "${THIRD_PARTY}/torchtitan" ]]; then
        git clone --depth 1 https://github.com/google-pytorch/torchtitan.git "${THIRD_PARTY}/torchtitan"
    fi
    python3 -m pip install --no-deps -e "${THIRD_PARTY}/torchtitan"
fi

if check_mod vllm_torchtpu; then
    echo "[Bootstrap] Found installed vllm_torchtpu."
else
    echo "[Bootstrap] Cloning and installing vllm-project/vllm-torchtpu..."
    if [[ ! -d "${THIRD_PARTY}/vllm-torchtpu" ]]; then
        git clone --depth 1 https://github.com/vllm-project/vllm-torchtpu.git "${THIRD_PARTY}/vllm-torchtpu"
    fi
    python3 -m pip install --no-deps -e "${THIRD_PARTY}/vllm-torchtpu"
fi

if check_mod tpu_sync; then
    echo "[Bootstrap] Found installed google/tpu-sync."
else
    echo "[Bootstrap] Attempting pip install of tpu-sync-torch, falling back to google/tpu-sync source clone..."
    if ! python3 -m pip install tpu-sync-torch 2>/dev/null; then
        if [[ ! -d "${THIRD_PARTY}/tpu-sync" ]]; then
            git clone --depth 1 https://github.com/google/tpu-sync.git "${THIRD_PARTY}/tpu-sync"
        fi
        echo "[Bootstrap] Cloned google/tpu-sync at ${THIRD_PARTY}/tpu-sync (run ./build.sh on TPU VM if wheel unavailable)."
    fi
fi

echo "=== [Bootstrap] Preparing deterministic GSM8K & MATH-500 datasets ==="
python3 "${REPO_ROOT}/scripts/prepare_datasets.py" --output-dir "${REPO_ROOT}/data"

echo "=== [Bootstrap] Running Phase 0 TPU v7 Probe ==="
python3 "${REPO_ROOT}/scripts/probe_tpu_v7.py" --update-doc

#!/usr/bin/env bash
# Phase 1 Gate: 10-Step Baseline VeRL + TorchTitan + vLLM-TorchTPU Smoke Run & Validation.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUT_DIR="${REPO_ROOT}/results/phase1_smoke"

echo "=== [Phase 1] Running TPU & Ray Hygiene Reset ==="
bash "${SCRIPT_DIR}/tpu_hygiene.sh"

echo "=== [Phase 1] Verifying Phase 0 Hardware Gate ==="
python3 "${SCRIPT_DIR}/probe_tpu_v7.py" --require-tpu-v7

echo "=== [Phase 1] Launching 10-Step Baseline VeRL TPU GRPO Smoke Run ==="
rm -rf "${OUT_DIR}"
python3 "${SCRIPT_DIR}/run_grpo_experiment.py" \
    --profile phase1_smoke \
    --output-dir "${OUT_DIR}" \
    --seed 42

echo "=== [Phase 1] Validating Numerical & Telemetry Gate via Pytest ==="
PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}" pytest "${REPO_ROOT}/tests/test_baseline_grpo.py" \
    --run-dir "${OUT_DIR}" \
    --require-tpu-v7

echo "=== [Phase 1] PASSED ==="

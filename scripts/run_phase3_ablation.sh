#!/usr/bin/env bash
# Phase 3 Gate: Controlled 3x60-Step GRPO Convergence Ablation + KL Sweep + XProf Trace Analysis.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ABLATION_DIR="${REPO_ROOT}/results/phase3_ablation"
SWEEP_DIR="${REPO_ROOT}/results/kl_sweep"
PLOTS_DIR="${REPO_ROOT}/results/plots"
SEED=42

python3 "${SCRIPT_DIR}/probe_tpu_v7.py" --require-tpu-v7

for PROFILE in run1_sync_bf16_baseline run2_async_fp8_tpu_sync run3_zero_stall_kv_grpo; do
    echo "=================================================================="
    echo "[Phase 3] Preparing ${PROFILE} (seed=${SEED})"
    echo "=================================================================="
    bash "${SCRIPT_DIR}/tpu_hygiene.sh"
    RUN_OUT="${ABLATION_DIR}/${PROFILE}"
    rm -rf "${RUN_OUT}"
    python3 "${SCRIPT_DIR}/run_grpo_experiment.py" \
        --profile "${PROFILE}" \
        --output-dir "${RUN_OUT}" \
        --seed "${SEED}"
done

echo "=== [Phase 3] Running KL Staleness Threshold Sensitivity Sweep ==="
for DELTA in 0.0 0.005 0.01 0.02 0.05 0.10; do
    bash "${SCRIPT_DIR}/tpu_hygiene.sh"
    SWEEP_OUT="${SWEEP_DIR}/delta_${DELTA}"
    rm -rf "${SWEEP_OUT}"
    python3 "${SCRIPT_DIR}/run_grpo_experiment.py" \
        --profile run3_zero_stall_kv_grpo \
        --steps 30 \
        --kv-delta "${DELTA}" \
        --output-dir "${SWEEP_OUT}" \
        --seed "${SEED}"
done

echo "=== [Phase 3] Analyzing XProf (.xplane.pb) Steady-State Traces (Steps 5-8) ==="
python3 "${SCRIPT_DIR}/analyze_grpo_traces.py" \
    "${ABLATION_DIR}/run1_sync_bf16_baseline" \
    "${ABLATION_DIR}/run2_async_fp8_tpu_sync" \
    "${ABLATION_DIR}/run3_zero_stall_kv_grpo" \
    --output-json "${ABLATION_DIR}/xprof_summary.json"

echo "=== [Phase 3] Generating Publication Plots & Updating Paper Telemetry Tables ==="
python3 "${SCRIPT_DIR}/generate_plots.py" \
    --ablation-dir "${ABLATION_DIR}" \
    --sweep-dir "${SWEEP_DIR}" \
    --output-dir "${PLOTS_DIR}"

python3 "${SCRIPT_DIR}/populate_paper_tables.py" --ablation-dir "${ABLATION_DIR}"
echo "=== [Phase 3] PASSED ==="

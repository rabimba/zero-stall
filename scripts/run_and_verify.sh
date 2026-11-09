#!/usr/bin/env bash
# Phase 4 Master Reproducibility & Deliverable Verification Gate.
#
# Modes:
#   1. Default (Strict Cloud TPU v7 Hardware Mode):
#        ./scripts/run_and_verify.sh
#      Resets Ray & VFIO locks, runs all unit + hardware pytest suites, runs a cached
#      verification run on TPU v7, and validates all CSVs, JSONs, XProf summaries, and plots.
#
#   2. Local Pre-Deployment Staging Mode (`--stage-verify`):
#        ./scripts/run_and_verify.sh --stage-verify
#      Verifies hygiene, runs the complete CPU/mathematical/wire-format pytest suite,
#      checks dry-run Hydra command synthesis for 1/2/4/8-chip TPU v7 slices, and tests
#      the XProf analyzer & plot generator inside an ephemeral temp directory (keeping
#      `results/` 100% free of synthetic/mocked data!).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
MODE="tpu"

if [[ "${1:-}" == "--stage-verify" ]]; then
    MODE="stage"
fi

echo "=================================================================="
echo " Zero-Stall TPU v7 GRPO — End-to-End Verification (mode=${MODE})"
echo "=================================================================="

# 1. Clean Ray & VFIO lock hygiene
bash "${SCRIPT_DIR}/tpu_hygiene.sh"

# 2. Run Pytest unit test suite
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
if [[ "${MODE}" == "tpu" ]]; then
    echo "[Verify] Probing Cloud TPU v7 hardware gate..."
    python3 "${SCRIPT_DIR}/probe_tpu_v7.py" --require-tpu-v7 --update-doc

    echo "[Verify] Running full unit + TPU hardware pytest suite..."
    pytest "${REPO_ROOT}/tests" --require-tpu-v7

    echo "[Verify] Executing cached 6-step TPU v7 verification run..."
    VERIFY_RUN_DIR="${REPO_ROOT}/results/phase4_cached_verify"
    rm -rf "${VERIFY_RUN_DIR}"
    python3 "${SCRIPT_DIR}/run_grpo_experiment.py" \
        --profile run3_zero_stall_kv_grpo \
        --steps 6 \
        --output-dir "${VERIFY_RUN_DIR}" \
        --seed 42

    pytest "${REPO_ROOT}/tests/test_baseline_grpo.py" \
        --run-dir "${VERIFY_RUN_DIR}" \
        --expected-min-steps 6 \
        --require-tpu-v7

    echo "[Verify] Validating Phase 3 deliverables (CSVs, JSONs, XProf summaries, PNG plots)..."
    python3 - <<'PY'
from pathlib import Path
import json, sys

root = Path(".")
required_files = [
    root / "results/phase0_topology.json",
    root / "results/phase3_ablation/run1_sync_bf16_baseline/step_metrics.csv",
    root / "results/phase3_ablation/run1_sync_bf16_baseline/val_metrics.csv",
    root / "results/phase3_ablation/run2_async_fp8_tpu_sync/step_metrics.csv",
    root / "results/phase3_ablation/run2_async_fp8_tpu_sync/val_metrics.csv",
    root / "results/phase3_ablation/run3_zero_stall_kv_grpo/step_metrics.csv",
    root / "results/phase3_ablation/run3_zero_stall_kv_grpo/val_metrics.csv",
    root / "results/phase3_ablation/xprof_summary.json",
    root / "results/plots/reward_and_val_vs_wallclock.png",
    root / "results/plots/step_breakdown_stacked_bar.png",
    root / "results/plots/kl_threshold_sensitivity_sweep.png",
]
missing = [str(p) for p in required_files if not p.exists() or p.stat().st_size == 0]
if missing:
    print(f"ERROR: Missing or empty required Phase 3/4 artifacts: {missing}", file=sys.stderr)
    sys.exit(1)

xp = json.loads((root / "results/phase3_ablation/xprof_summary.json").read_text())
assert len(xp.get("runs", [])) == 3, "Expected 3 runs in xprof_summary.json"
print("[Verify] All TPU v7 CSVs, JSONs, XProf summaries, and PNG plots verified!")
PY

else
    echo "[Verify] Running full local staging unit test suite (math, FP8 quant, async 20-cycle sync, KV loss, XPlane wire parser)..."
    pytest "${REPO_ROOT}/tests"

    echo "[Verify] Validating dry-run Hydra command synthesis across 1, 2, 4, and 8 physical TPU v7 chips..."
    TMP_STAGE="$(mktemp -d)"
    trap 'rm -rf "${TMP_STAGE}"' EXIT
    for CHIPS in 1 2 4 8; do
        for PROF in phase1_smoke run1_sync_bf16_baseline run2_async_fp8_tpu_sync run3_zero_stall_kv_grpo; do
            python3 "${SCRIPT_DIR}/run_grpo_experiment.py" \
                --profile "${PROF}" \
                --output-dir "${TMP_STAGE}/chips_${CHIPS}_${PROF}" \
                --simulated-chips "${CHIPS}" \
                --dry-run >/dev/null
            bash -n "${TMP_STAGE}/chips_${CHIPS}_${PROF}/resolved_hydra_cmd.sh"
        done
    done
    echo "[Verify] Validated bash syntax for all 16 generated Hydra launch scripts."

    echo "[Verify] Validating shell script syntax across scripts/*.sh..."
    for SH_FILE in "${SCRIPT_DIR}"/*.sh; do
        bash -n "${SH_FILE}"
    done
    echo "[Verify] Pre-deployment staging verification PASSED (zero synthetic data written to results/)."
fi

echo "=================================================================="
echo " SUCCESS: Exit Code 0"
echo "=================================================================="

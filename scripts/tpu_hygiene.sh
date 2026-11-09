#!/usr/bin/env bash
# Enforces Contract Rule #3: Process, Ray, VFIO Lock & Persistent Compilation Cache Hygiene.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CACHE_ROOT="${ZERO_STALL_CACHE_ROOT:-/var/tmp/zero_stall_tpu_cache}"

echo "[Hygiene] 1/4 Stopping any active Ray cluster processes..."
if command -v ray >/dev/null 2>&1; then
    ray stop --force >/dev/null 2>&1 || true
else
    echo "[Hygiene]     'ray' CLI not in PATH (skipping ray stop)."
fi

echo "[Hygiene] 2/4 Clearing stale libtpu lockfiles in /tmp..."
rm -f /tmp/libtpu_lockfile* /tmp/tpu_logs/libtpu_lockfile* 2>/dev/null || true

echo "[Hygiene] 3/4 Verifying /dev/vfio/* IOMMU device groups are completely unlocked..."
PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}" python3 - <<'PY'
import sys
from src.tpu_topology import check_vfio_and_libtpu_locks

report = check_vfio_and_libtpu_locks()
print(f"[Hygiene]     VFIO control node present: {report.vfio_control_present}")
print(f"[Hygiene]     VFIO IOMMU device groups:  {len(report.iommu_group_devices)} {report.iommu_group_devices}")
if not report.is_clean:
    print(f"[Hygiene] ERROR: Active VFIO locks or stale lockfiles detected!", file=sys.stderr)
    print(f"[Hygiene]        Locked devices: {report.locked_devices}", file=sys.stderr)
    print(f"[Hygiene]        Stale locks:    {report.stale_lockfiles}", file=sys.stderr)
    sys.exit(1)
print("[Hygiene]     VFIO & libtpu lock state is CLEAN.")
PY

echo "[Hygiene] 4/4 Ensuring persistent local disk compilation cache directories exist at ${CACHE_ROOT}..."
mkdir -p \
    "${CACHE_ROOT}/torch_tpu_tier3" \
    "${CACHE_ROOT}/vllm_cache" \
    "${CACHE_ROOT}/xla_cache" \
    "${CACHE_ROOT}/hf_home"

export TORCH_TPU_TIER2_COMPILATION_CACHE="zero_stall_grpo"
export TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT="${CACHE_ROOT}/torch_tpu_tier3"
export VLLM_CACHE_ROOT="${CACHE_ROOT}/vllm_cache"
export VLLM_XLA_CACHE_PATH="${CACHE_ROOT}/xla_cache"
export XLA_CACHE_DIR="${CACHE_ROOT}/xla_cache"

echo "[Hygiene] Ready. Persistent caches configured under ${CACHE_ROOT}."

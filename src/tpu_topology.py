"""Cloud TPU v7 (Ironwood / tpu7x) Topology Probing, Hygiene & Dynamic Partitioning.

Grounded in upstream contracts from:
- `google-pytorch/torch_tpu` (`TORCH_TPU_TIER2_COMPILATION_CACHE`, `TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT`)
- `google-pytorch/torchtitan` (`torchtitan/experiments/tpu/recipes/ironwood/*/tpu7x-*`)
- `verl-project/verl` (`examples/tpu/grpo/run_qwen3_4b_torchtitan.sh`, `run_qwen3_5_torchtitan.sh`)
- `vllm-project/vllm-torchtpu` (`VLLM_CACHE_ROOT`, `TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL`, `VLLM_TPU_BLOCK_MAJOR_KV`)
- `google/tpu-sync` (`tpu_sync` DMA & `WeightSynchronizer` slice coordination)
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
import glob
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any


# Cloud TPU v7 (Ironwood / tpu7x) hardware constants per Google Cloud / TorchTitan specs:
# - Each physical Ironwood chip exposes 2 TensorCores (2 logical PJRT devices in 1-core/process mode)
# - 192 GiB HBM3E per physical chip (~96 GiB per logical TensorCore) at ~7.37 TB/s HBM bandwidth
# - Peak dense compute per physical chip: ~4,614 TFLOPS FP8 / ~2,307 TFLOPS BF16
IRONWOOD_HBM_GIB_PER_CHIP: float = 192.0
IRONWOOD_LOGICAL_CORES_PER_CHIP: int = 2
IRONWOOD_PEAK_FP8_TFLOPS_PER_CHIP: float = 4614.0
IRONWOOD_PEAK_BF16_TFLOPS_PER_CHIP: float = 2307.0

SPARSECORE_TRAINER_XLA_FLAGS: tuple[str, ...] = (
    "--xla_enable_hlo_sharding_v3=false",
    "--xla_tpu_enable_sparse_core_collective_aggregator=true",
    "--xla_tpu_enable_latency_hiding_layer_scheduler=true",
    "--xla_tpu_enable_sparse_core_offload_queuing_in_lhs=true",
    "--xla_tpu_enable_sparse_core_collective_offload_all_gather=true",
    "--xla_tpu_enable_sparse_core_collective_offload_2d_all_gather=true",
    "--xla_tpu_enable_sparse_core_collective_offload_3d_all_gather=true",
    "--xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=true",
    "--xla_tpu_enable_sparse_core_collective_offload_nd_reduce_scatter=true",
    "--xla_tpu_enable_sparse_core_hierarchical_reduce_scatter=true",
    "--xla_tpu_use_single_sparse_core_for_reduce_scatter_offload=false",
    "--xla_enable_async_all_gather=true",
    "--xla_tpu_prefer_async_allgather_to_allreduce=true",
    "--xla_tpu_enable_ici_ag_pipelining=true",
    "--xla_tpu_enable_ici_rs_pipelining=true",
    "--xla_tpu_ici_rs_pipelining_threshold_kib=1024",
)


@dataclass(frozen=True)
class PackageStatus:
    """Installed status of a Python package in the active environment."""

    package_name: str
    import_name: str
    installed: bool
    version: str | None = None
    origin: str | None = None


@dataclass(frozen=True)
class VfioLockReport:
    """Status of `/dev/vfio/*` IOMMU device groups and libtpu lock files."""

    vfio_control_present: bool
    iommu_group_devices: list[str]
    locked_devices: dict[str, list[int]]
    stale_lockfiles: list[str]

    @property
    def is_clean(self) -> bool:
        return len(self.locked_devices) == 0 and len(self.stale_lockfiles) == 0


@dataclass(frozen=True)
class PartitionPlan:
    """Dynamic chip/core partitioning plan for TorchTitan trainer + vLLM-TorchTPU rollout."""

    mode: str
    physical_chips: int
    logical_devices: int
    num_hosts: int
    hbm_per_chip_gib: float
    total_hbm_gib: float
    trainer_nnodes: int
    trainer_devices_per_node: int
    trainer_dp_shard: int
    trainer_tp: int
    trainer_cp: int
    rollout_nnodes: int
    rollout_devices_per_node: int
    rollout_tp: int
    rollout_dp: int
    vllm_gpu_memory_utilization: float
    xla_mem_fraction: float
    enable_sparsecore_offload: bool
    recommended_smoke_model: str
    recommended_scale_model: str
    ici_topology: str
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class EnvironmentProbeResult:
    """Complete Phase 0 hardware and software stack discovery snapshot."""

    hostname: str
    kernel: str
    arch: str
    cpu_model: str
    cpu_count_logical: int
    host_ram_total_gib: float
    host_ram_available_gib: float
    python_version: str
    python_executable: str
    accelerator_type: str
    is_tpu_v7: bool
    physical_tpu_chips: int
    logical_tpu_devices: int
    ici_topology: str
    vfio_report: VfioLockReport
    packages: dict[str, PackageStatus]
    partition_plan: PartitionPlan | None

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def check_vfio_and_libtpu_locks() -> VfioLockReport:
    """Inspect `/dev/vfio/*` and `/tmp/libtpu_lockfile*` without acquiring any device lock."""
    vfio_ctrl = os.path.exists("/dev/vfio/vfio")
    group_devs = sorted(
        p for p in glob.glob("/dev/vfio/*") if os.path.basename(p).isdigit()
    )

    locked: dict[str, list[int]] = {}
    target_set = set(group_devs)
    if target_set and os.path.isdir("/proc"):
        for pid_entry in os.listdir("/proc"):
            if not pid_entry.isdigit():
                continue
            pid = int(pid_entry)
            if pid == os.getpid():
                continue
            fd_dir = f"/proc/{pid}/fd"
            try:
                for fd_name in os.listdir(fd_dir):
                    try:
                        link_dst = os.readlink(f"{fd_dir}/{fd_name}")
                    except OSError:
                        continue
                    if link_dst in target_set:
                        locked.setdefault(link_dst, []).append(pid)
            except OSError:
                continue

    for dev in locked:
        locked[dev] = sorted(set(locked[dev]))

    stale_locks = sorted(glob.glob("/tmp/libtpu_lockfile*"))
    return VfioLockReport(
        vfio_control_present=vfio_ctrl,
        iommu_group_devices=group_devs,
        locked_devices=locked,
        stale_lockfiles=stale_locks,
    )


def _probe_package(package_name: str, import_name: str) -> PackageStatus:
    """Check package metadata and module spec without triggering PJRT initialization."""
    version: str | None = None
    try:
        version = importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        pass

    spec = None
    try:
        spec = importlib.util.find_spec(import_name)
    except (ImportError, ValueError, AttributeError):
        spec = None

    installed = (version is not None) or (spec is not None)
    origin = getattr(spec, "origin", None) if spec is not None else None
    return PackageStatus(
        package_name=package_name,
        import_name=import_name,
        installed=installed,
        version=version,
        origin=origin,
    )


def _probe_tpu_subprocess(timeout_s: float = 25.0) -> dict[str, Any]:
    """Probe `tpu_info` and `torch.tpu` in an isolated child process so parent never locks VFIO."""
    probe_code = r"""
import json, os, sys
out = {
    "tpu_info_chips": 0,
    "tpu_info_type": None,
    "tpu_info_topology": None,
    "torch_tpu_available": False,
    "torch_tpu_device_count": 0,
    "torch_tpu_device_name": None,
}
try:
    from tpu_info import device as tpu_device
    chips = tpu_device.get_local_chips()
    if chips:
        chip_type, count = chips[0], chips[1] if isinstance(chips, tuple) else (chips[0].chip_type, len(chips))
        out["tpu_info_chips"] = int(count)
        out["tpu_info_type"] = str(chip_type)
except Exception:
    pass

try:
    import torch
    import torch_tpu  # noqa: F401
    if hasattr(torch, "tpu") and torch.tpu.is_available():
        out["torch_tpu_available"] = True
        out["torch_tpu_device_count"] = int(torch.tpu.device_count())
        try:
            out["torch_tpu_device_name"] = str(torch.tpu.get_device_name(0))
        except Exception:
            out["torch_tpu_device_name"] = "tpu"
except Exception:
    pass

print(json.dumps(out))
"""
    try:
        proc = subprocess.run(
            [sys.executable, "-c", probe_code],
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            last_line = proc.stdout.strip().splitlines()[-1]
            return json.loads(last_line)
    except Exception:
        pass
    return {
        "tpu_info_chips": 0,
        "tpu_info_type": None,
        "tpu_info_topology": None,
        "torch_tpu_available": False,
        "torch_tpu_device_count": 0,
        "torch_tpu_device_name": None,
    }


def _read_cpu_and_mem() -> tuple[str, float, float]:
    cpu_model = platform.processor() or "unknown"
    if os.path.exists("/proc/cpuinfo"):
        with open("/proc/cpuinfo", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if line.lower().startswith("model name"):
                    cpu_model = line.split(":", 1)[1].strip()
                    break

    total_gib = 0.0
    avail_gib = 0.0
    if os.path.exists("/proc/meminfo"):
        with open("/proc/meminfo", encoding="utf-8", errors="ignore") as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 2 and parts[0] == "MemTotal:":
                    total_gib = int(parts[1]) / (1024.0 * 1024.0)
                elif len(parts) >= 2 and parts[0] == "MemAvailable:":
                    avail_gib = int(parts[1]) / (1024.0 * 1024.0)
    return cpu_model, round(total_gib, 2), round(avail_gib, 2)


def _infer_ici_topology(physical_chips: int, env_topo: str | None = None) -> str:
    if env_topo:
        return env_topo
    topo_map = {
        1: "1x1x1",
        2: "1x2x1",
        4: "2x2x1",
        8: "2x2x2",
        16: "2x4x2",
        32: "4x4x2",
        64: "4x4x4",
    }
    return topo_map.get(physical_chips, f"{physical_chips}x1x1")


def compute_partition_plan(
    physical_chips: int,
    logical_devices: int | None = None,
    num_hosts: int = 1,
    ici_topology: str | None = None,
    is_ironwood: bool = True,
) -> PartitionPlan:
    """Compute optimal TorchTitan + vLLM-TorchTPU slice partitioning for TPU v7 (`tpu7x`).

    Implements Phase 0 dynamic partitioning rules:
    - If `physical_chips >= 4`: Dedicate 50% of the slice to TorchTitan (`SimpleFSDP` +
      `SplashAttention` + `fuse_qkv` + SparseCore collective offload) and 50% to standalone
      `vLLM-TorchTPU` rollout workers orchestrated by Ray (`separate_async`, 1-step overlap).
    - If `physical_chips in (1, 2)`: Configure split-chip (2 chips: 1 chip trainer + 1 chip
      rollout) or dual-TensorCore single-chip orchestration sized cleanly within 192 GiB
      HBM3E per physical Ironwood chip.
    """
    if physical_chips <= 0:
        raise ValueError(f"physical_chips must be >= 1, got {physical_chips}")

    cores_per_chip = IRONWOOD_LOGICAL_CORES_PER_CHIP if is_ironwood else 1
    if logical_devices is None or logical_devices <= 0:
        logical_devices = physical_chips * cores_per_chip

    hbm_per_chip = IRONWOOD_HBM_GIB_PER_CHIP if is_ironwood else 32.0
    total_hbm = physical_chips * hbm_per_chip
    topo = _infer_ici_topology(physical_chips, ici_topology)

    if physical_chips >= 4:
        # Half slice to TorchTitan, half slice to vLLM-TorchTPU
        if num_hosts >= 2 and num_hosts % 2 == 0:
            trainer_nnodes = num_hosts // 2
            rollout_nnodes = num_hosts // 2
            devices_per_host = logical_devices // num_hosts
            trainer_dev_per_node = devices_per_host
            rollout_dev_per_node = devices_per_host
        else:
            trainer_nnodes = 1
            rollout_nnodes = 1
            trainer_dev_per_node = logical_devices // 2
            rollout_dev_per_node = logical_devices - trainer_dev_per_node

        total_trainer_devs = trainer_nnodes * trainer_dev_per_node
        total_rollout_devs = rollout_nnodes * rollout_dev_per_node
        rollout_tp = min(4, total_rollout_devs)
        while total_rollout_devs % rollout_tp != 0 and rollout_tp > 1:
            rollout_tp //= 2
        rollout_dp = max(1, total_rollout_devs // rollout_tp)

        return PartitionPlan(
            mode="disaggregated_50_50_slice",
            physical_chips=physical_chips,
            logical_devices=logical_devices,
            num_hosts=num_hosts,
            hbm_per_chip_gib=hbm_per_chip,
            total_hbm_gib=total_hbm,
            trainer_nnodes=trainer_nnodes,
            trainer_devices_per_node=trainer_dev_per_node,
            trainer_dp_shard=total_trainer_devs,
            trainer_tp=1,
            trainer_cp=1,
            rollout_nnodes=rollout_nnodes,
            rollout_devices_per_node=rollout_dev_per_node,
            rollout_tp=rollout_tp,
            rollout_dp=rollout_dp,
            vllm_gpu_memory_utilization=0.65,
            xla_mem_fraction=0.85,
            enable_sparsecore_offload=is_ironwood,
            recommended_smoke_model="Qwen/Qwen3-4B-Base",
            recommended_scale_model="Qwen/Qwen3-8B-Base",
            ici_topology=topo,
            notes=[
                "50/50 disaggregated slice split: TorchTitan (Actor+Ref colocated in single WorkerDict PJRT process/core) + standalone vLLM-TorchTPU rollout pool.",
                "SparseCore collective offload enabled ONLY on trainer/ref ranks via VERL_TPU_TRAINER_XLA_FLAGS.",
                "CheckpointEngineWorker runs as 0-TPU CPU sidecar streaming FP8 block-scaled weights via google/tpu-sync.",
            ],
        )

    if physical_chips == 2:
        # 2 physical Ironwood chips (384 GiB HBM3E total, 4 logical TensorCores):
        # Chip 0 (2 TensorCores, 192 GiB) -> TorchTitan SimpleFSDP
        # Chip 1 (2 TensorCores, 192 GiB) -> vLLM-TorchTPU TP=2
        trainer_devs = max(1, logical_devices // 2)
        rollout_devs = max(1, logical_devices - trainer_devs)
        return PartitionPlan(
            mode="split_2chip_disaggregated",
            physical_chips=2,
            logical_devices=logical_devices,
            num_hosts=1,
            hbm_per_chip_gib=hbm_per_chip,
            total_hbm_gib=total_hbm,
            trainer_nnodes=1,
            trainer_devices_per_node=trainer_devs,
            trainer_dp_shard=trainer_devs,
            trainer_tp=1,
            trainer_cp=1,
            rollout_nnodes=1,
            rollout_devices_per_node=rollout_devs,
            rollout_tp=rollout_devs,
            rollout_dp=1,
            vllm_gpu_memory_utilization=0.65,
            xla_mem_fraction=0.80,
            enable_sparsecore_offload=is_ironwood,
            recommended_smoke_model="Qwen/Qwen3-4B-Base",
            recommended_scale_model="Qwen/Qwen3-8B-Base",
            ici_topology=topo,
            notes=[
                "Chip 0 dedicated to TorchTitan (Actor+Ref, 192 GiB HBM3E); Chip 1 dedicated to vLLM-TorchTPU (192 GiB HBM3E).",
                "Separate deepsea_slice_builder_worker_port ranges (8471+ trainer, 8070+ rollout) with --xla_tpu_use_enhanced_launch_barrier=false.",
            ],
        )

    # 1 physical Ironwood chip (192 GiB HBM3E, 2 logical TensorCores of ~96 GiB each)
    if logical_devices >= 2:
        return PartitionPlan(
            mode="single_chip_dual_core_split",
            physical_chips=1,
            logical_devices=logical_devices,
            num_hosts=1,
            hbm_per_chip_gib=hbm_per_chip,
            total_hbm_gib=total_hbm,
            trainer_nnodes=1,
            trainer_devices_per_node=1,
            trainer_dp_shard=1,
            trainer_tp=1,
            trainer_cp=1,
            rollout_nnodes=1,
            rollout_devices_per_node=1,
            rollout_tp=1,
            rollout_dp=1,
            vllm_gpu_memory_utilization=0.45,
            xla_mem_fraction=0.45,
            enable_sparsecore_offload=False,
            recommended_smoke_model="Qwen/Qwen3-4B-Base",
            recommended_scale_model="Qwen/Qwen3-4B-Base",
            ici_topology=topo,
            notes=[
                "1 physical Ironwood chip (192 GiB HBM3E) exposes 2 TensorCores (~96 GiB/core): TensorCore 0 runs TorchTitan Actor+Ref, TensorCore 1 runs vLLM-TorchTPU.",
                "Uses XLA_PYTHON_CLIENT_PREALLOCATE=false and TPU_CHIPS_PER_PROCESS_BOUNDS=1,1,1 to stay cleanly inside 192 GiB HBM3E.",
            ],
        )

    return PartitionPlan(
        mode="single_device_time_sliced_sync",
        physical_chips=1,
        logical_devices=1,
        num_hosts=1,
        hbm_per_chip_gib=hbm_per_chip,
        total_hbm_gib=total_hbm,
        trainer_nnodes=1,
        trainer_devices_per_node=1,
        trainer_dp_shard=1,
        trainer_tp=1,
        trainer_cp=1,
        rollout_nnodes=1,
        rollout_devices_per_node=1,
        rollout_tp=1,
        rollout_dp=1,
        vllm_gpu_memory_utilization=0.42,
        xla_mem_fraction=0.45,
        enable_sparsecore_offload=False,
        recommended_smoke_model="Qwen/Qwen3-0.6B",
        recommended_scale_model="Qwen/Qwen3-4B-Base",
        ici_topology=topo,
        notes=[
            "Single logical TPU core detected: sized with XLA_PYTHON_CLIENT_MEM_FRACTION=0.45 and tpu-sync host DRAM staging.",
        ],
    )


def build_cache_and_runtime_env(
    cache_root: str | os.PathLike[str],
    partition_plan: PartitionPlan | None = None,
    enable_timeline: bool = True,
) -> dict[str, str]:
    """Construct persistent compilation cache & TorchTPU/vLLM/VeRL environment variables."""
    root = Path(cache_root).expanduser().resolve()
    tier3_dir = root / "torch_tpu_tier3"
    vllm_dir = root / "vllm_cache"
    xla_dir = root / "xla_cache"
    for d in (tier3_dir, vllm_dir, xla_dir):
        d.mkdir(parents=True, exist_ok=True)

    libtpu_flags = [
        "--xla_tpu_use_enhanced_launch_barrier=false",
        "--xla_tpu_bf16_emission_mode=PROMOTE_F32",
        "--xla_tpu_enable_bitcast_reduce_fusion_graduation=false",
        "--xla_tpu_enable_sublane_major_scaling_bitcast_fusion=false",
    ]

    env = {
        "VERL_PLATFORM": "tpu",
        "VERL_USE_EXTERNAL_MODULES": "verl_hardware_plugin,src.verl_tpu_plugin_ext",
        "VERL_TPU_VLLM_ATTN_FP32_STATS": "1",
        "VERL_TPU_TIMELINE": "1" if enable_timeline else "0",
        "TPU_ACCELERATOR_TYPE": "tpu7x",
        "VLLM_TARGET_DEVICE": "tpu",
        "VLLM_PLUGINS": "torchtpu,torchtpu_layers",
        "VLLM_TPU_BLOCK_MAJOR_KV": "1",
        "TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL": "1",
        "TORCH_TPU_TIER2_COMPILATION_CACHE": "zero_stall_grpo",
        "TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT": str(tier3_dir),
        "VLLM_CACHE_ROOT": str(vllm_dir),
        "VLLM_XLA_CACHE_PATH": str(xla_dir),
        "XLA_CACHE_DIR": str(xla_dir),
        "RAY_OVERRIDE_JOB_RUNTIME_ENV": "1",
        "RAY_memory_monitor_refresh_ms": "0",
        "RAY_memory_usage_threshold": "0.99",
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        "LIBTPU_INIT_ARGS": " ".join(libtpu_flags),
        "PYTHONHASHSEED": "42",
    }

    if partition_plan is not None:
        env["XLA_PYTHON_CLIENT_MEM_FRACTION"] = f"{partition_plan.xla_mem_fraction:.2f}"
        env["JAX_MEM_FRACTION"] = f"{partition_plan.xla_mem_fraction:.2f}"
        if partition_plan.enable_sparsecore_offload:
            env["VERL_TPU_TRAINER_LIBTPU_INIT_ARGS"] = "--xla_tpu_scoped_vmem_limit_kib=65472"
            env["VERL_TPU_TRAINER_XLA_FLAGS"] = " ".join(SPARSECORE_TRAINER_XLA_FLAGS)
        if partition_plan.mode == "single_chip_dual_core_split":
            env["TPU_CHIPS_PER_PROCESS_BOUNDS"] = "1,1,1"

    return env


def probe_environment(run_subprocess_tpu_probe: bool = True) -> EnvironmentProbeResult:
    """Run complete Phase 0 topology and software stack probe."""
    cpu_model, ram_total, ram_avail = _read_cpu_and_mem()
    vfio_report = check_vfio_and_libtpu_locks()

    target_packages = [
        ("torch", "torch"),
        ("torch-tpu", "torch_tpu"),
        ("libtpu", "libtpu"),
        ("verl", "verl"),
        ("verl-hardware-plugin", "verl_hardware_plugin"),
        ("torchtitan", "torchtitan"),
        ("vllm", "vllm"),
        ("vllm-torchtpu-core", "vllm_torchtpu"),
        ("ray", "ray"),
        ("tpu-sync-torch", "tpu_sync"),
        ("tpu-info", "tpu_info"),
        ("xprof", "xprof"),
    ]
    pkg_map: dict[str, PackageStatus] = {}
    for dist_name, mod_name in target_packages:
        pkg_map[mod_name] = _probe_package(dist_name, mod_name)

    sub_probe = (
        _probe_tpu_subprocess()
        if run_subprocess_tpu_probe and (len(vfio_report.iommu_group_devices) > 0 or pkg_map["torch_tpu"].installed)
        else {
            "tpu_info_chips": 0,
            "tpu_info_type": None,
            "tpu_info_topology": None,
            "torch_tpu_available": False,
            "torch_tpu_device_count": 0,
            "torch_tpu_device_name": None,
        }
    )

    vfio_chips = len(vfio_report.iommu_group_devices)
    physical_chips = int(sub_probe.get("tpu_info_chips") or vfio_chips)
    logical_devs = int(sub_probe.get("torch_tpu_device_count") or 0)

    env_accel = os.environ.get("TPU_ACCELERATOR_TYPE", "")
    dev_name = str(sub_probe.get("torch_tpu_device_name") or sub_probe.get("tpu_info_type") or env_accel or "none")
    is_v7 = any(tok in dev_name.lower() for tok in ("v7", "7x", "ironwood")) or (
        physical_chips > 0 and logical_devs == physical_chips * 2
    )

    if physical_chips > 0 and logical_devs == 0:
        logical_devs = physical_chips * (IRONWOOD_LOGICAL_CORES_PER_CHIP if is_v7 else 1)

    accel_label = dev_name if physical_chips > 0 else "none (no TPU VFIO devices attached)"
    topo = _infer_ici_topology(
        physical_chips,
        os.environ.get("TORCH_TPU_TOPOLOGY") or sub_probe.get("tpu_info_topology"),
    ) if physical_chips > 0 else "none"

    plan = (
        compute_partition_plan(
            physical_chips=physical_chips,
            logical_devices=logical_devs,
            num_hosts=int(os.environ.get("TPU_NUM_HOSTS", "1")),
            ici_topology=topo,
            is_ironwood=is_v7 or True,
        )
        if physical_chips > 0
        else None
    )

    return EnvironmentProbeResult(
        hostname=platform.node(),
        kernel=f"{platform.system()} {platform.release()}",
        arch=platform.machine(),
        cpu_model=cpu_model,
        cpu_count_logical=os.cpu_count() or 1,
        host_ram_total_gib=ram_total,
        host_ram_available_gib=ram_avail,
        python_version=platform.python_version(),
        python_executable=sys.executable,
        accelerator_type=accel_label,
        is_tpu_v7=is_v7 and physical_chips > 0,
        physical_tpu_chips=physical_chips,
        logical_tpu_devices=logical_devs,
        ici_topology=topo,
        vfio_report=vfio_report,
        packages=pkg_map,
        partition_plan=plan,
    )

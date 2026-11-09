#!/usr/bin/env python3
"""Phase 0 Gate: Cloud TPU v7 (`tpu7x` / Ironwood) Topology & Software Stack Discovery."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.tpu_topology import EnvironmentProbeResult, probe_environment  # noqa: E402


def render_live_probe_markdown(probe: EnvironmentProbeResult) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    lines = [
        "<!-- BEGIN_AUTO_PHASE0_PROBE -->",
        f"### Latest Automated Hardware Probe (`{ts}`)",
        "",
        "| Attribute | Detected Value |",
        "| :--- | :--- |",
        f"| **Hostname / OS** | `{probe.hostname}` (`{probe.kernel}`, `{probe.arch}`) |",
        f"| **CPU / Host RAM** | `{probe.cpu_model}` ({probe.cpu_count_logical} logical CPUs), {probe.host_ram_total_gib:.2f} GiB total ({probe.host_ram_available_gib:.2f} GiB avail) |",
        f"| **Python** | `{probe.python_version}` (`{probe.python_executable}`) |",
        f"| **Accelerator** | `{probe.accelerator_type}` (`is_tpu_v7={probe.is_tpu_v7}`) |",
        f"| **Physical TPU Chips** | `{probe.physical_tpu_chips}` |",
        f"| **Logical PJRT Devices** | `{probe.logical_tpu_devices}` |",
        f"| **ICI Topology** | `{probe.ici_topology}` |",
        f"| **VFIO Hygiene** | Clean={probe.vfio_report.is_clean}, Groups={len(probe.vfio_report.iommu_group_devices)} |",
        "",
        "#### Installed Package Matrix",
        "",
        "| Import Module | Distribution | Installed | Version |",
        "| :--- | :--- | :--- | :--- |",
    ]
    for mod_name, status in probe.packages.items():
        lines.append(
            f"| `{mod_name}` | `{status.package_name}` | `{'YES' if status.installed else 'NO'}` | `{status.version or '-'}` |"
        )

    if probe.partition_plan is not None:
        p = probe.partition_plan
        lines.extend(
            [
                "",
                "#### Computed Dynamic Slice Partition Plan",
                "",
                f"- **Mode**: `{p.mode}` ({p.physical_chips} physical chips / {p.logical_devices} logical TensorCores, {p.total_hbm_gib:.0f} GiB HBM3E total)",
                f"- **TorchTitan Trainer**: `nnodes={p.trainer_nnodes}`, `devices_per_node={p.trainer_devices_per_node}`, `dp_shard={p.trainer_dp_shard}`, `tp={p.trainer_tp}`, `sparsecore_offload={p.enable_sparsecore_offload}`",
                f"- **vLLM-TorchTPU Rollout**: `nnodes={p.rollout_nnodes}`, `devices_per_node={p.rollout_devices_per_node}`, `tp={p.rollout_tp}`, `dp={p.rollout_dp}`, `gpu_memory_utilization={p.vllm_gpu_memory_utilization}`",
                f"- **Recommended Models**: Smoke=`{p.recommended_smoke_model}`, Scale=`{p.recommended_scale_model}`",
            ]
        )
    else:
        lines.extend(
            [
                "",
                "> [!WARNING]",
                "> Probe ran on a host without attached Cloud TPU v7 VFIO devices. Re-run `python3 scripts/probe_tpu_v7.py --require-tpu-v7 --update-doc` on the Cloud TPU v7 VM to lock in live hardware topology.",
            ]
        )
    lines.append("<!-- END_AUTO_PHASE0_PROBE -->")
    return "\n".join(lines)


def update_architecture_doc(doc_path: Path, snippet: str) -> None:
    if not doc_path.exists():
        return
    content = doc_path.read_text(encoding="utf-8")
    start_marker = "<!-- BEGIN_AUTO_PHASE0_PROBE -->"
    end_marker = "<!-- END_AUTO_PHASE0_PROBE -->"
    if start_marker in content and end_marker in content:
        before = content.split(start_marker, 1)[0]
        after = content.split(end_marker, 1)[1]
        updated = before + snippet + after
    else:
        updated = content.rstrip() + "\n\n" + snippet + "\n"
    doc_path.write_text(updated, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe Cloud TPU v7 topology and software stack.")
    parser.add_argument(
        "--output-json",
        type=Path,
        default=REPO_ROOT / "results" / "phase0_topology.json",
        help="Path to write JSON probe snapshot.",
    )
    parser.add_argument(
        "--update-doc",
        action="store_true",
        help="Update auto-probe block inside docs/architecture_setup.md.",
    )
    parser.add_argument(
        "--require-tpu-v7",
        action="store_true",
        help="Exit non-zero if real Cloud TPU v7 hardware or clean VFIO state is missing.",
    )
    args = parser.parse_args()

    probe = probe_environment(run_subprocess_tpu_probe=True)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(probe.to_dict(), indent=2), encoding="utf-8")

    md = render_live_probe_markdown(probe)
    print(md)

    if args.update_doc:
        update_architecture_doc(REPO_ROOT / "docs" / "architecture_setup.md", md)

    if args.require_tpu_v7:
        if probe.physical_tpu_chips <= 0:
            print(
                "ERROR: Phase 0 Gate Failed — 0 TPU chips detected on this host (--require-tpu-v7).",
                file=sys.stderr,
            )
            return 2
        if not probe.vfio_report.is_clean:
            print(
                f"ERROR: Phase 0 Gate Failed — /dev/vfio/* locks not clean: {probe.vfio_report}",
                file=sys.stderr,
            )
            return 3
        required_mods = ("torch_tpu", "verl", "torchtitan", "vllm_torchtpu", "ray", "tpu_sync")
        missing = [m for m in required_mods if not probe.packages[m].installed]
        if missing:
            print(
                f"ERROR: Phase 0 Gate Failed — Missing required TPU stack modules: {missing}. "
                "Run `bash scripts/bootstrap_tpu_vm.sh` first.",
                file=sys.stderr,
            )
            return 4

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

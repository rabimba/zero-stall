"""Unit tests for binary XProf `.xplane.pb` wire-format trace analysis and plot generation."""

from __future__ import annotations

import math
from pathlib import Path
import struct

from scripts.analyze_grpo_traces import analyze_run_directory, parse_xspace_file
from scripts.generate_plots import (
    plot_kl_sensitivity_sweep,
    plot_reward_and_validation_vs_wallclock,
    plot_step_breakdown_stacked_bar,
)
from src.telemetry import StepTelemetryRecord, TelemetryLogger, ValidationTelemetryRecord


def _varint(val: int) -> bytes:
    if val < 0:
        val += 1 << 64
    out = bytearray()
    while True:
        b = val & 0x7F
        val >>= 7
        if val:
            out.append(b | 0x80)
        else:
            out.append(b)
            return bytes(out)


def _field_varint(fnum: int, val: int) -> bytes:
    return _varint((fnum << 3) | 0) + _varint(val)


def _field_double(fnum: int, val: float) -> bytes:
    return _varint((fnum << 3) | 1) + struct.pack("<d", val)


def _field_bytes(fnum: int, payload: bytes) -> bytes:
    return _varint((fnum << 3) | 2) + _varint(len(payload)) + payload


def _build_test_xspace_bytes() -> bytes:
    """Construct a valid `tensorflow.profiler.XSpace` binary protobuf matching `xplane.proto`:
    `/device:TPU:0` window `[0, 1_000_000_000]` ps (`1.0` ms):
      - Active spans: `[0, 350_000_000]` ps + `[500_000_000, 900_000_000]` ps = `750_000_000` ps (75% active -> 25% bubble!)
      - Sync event `tpu_sync_d2h`: `120_000_000` ps (`0.12` ms)
      - MXU hardware counter stat `mxu_utilization = 64.5`
    """
    ev_meta_compute = _field_varint(1, 1) + _field_bytes(2, _field_varint(1, 1) + _field_bytes(2, b"torchtitan_fwd_bwd"))
    ev_meta_sync = _field_varint(1, 2) + _field_bytes(2, _field_varint(1, 2) + _field_bytes(2, b"tpu_sync_d2h"))
    st_meta_mxu = _field_varint(1, 10) + _field_bytes(2, _field_varint(1, 10) + _field_bytes(2, b"mxu_utilization"))

    stat_mxu = _field_varint(1, 10) + _field_double(2, 64.5)
    ev1 = _field_varint(1, 1) + _field_varint(2, 0) + _field_varint(3, 350_000_000) + _field_bytes(4, stat_mxu)
    ev2 = _field_varint(1, 2) + _field_varint(2, 500_000_000) + _field_varint(3, 120_000_000)
    ev3 = _field_varint(1, 1) + _field_varint(2, 580_000_000) + _field_varint(3, 320_000_000) + _field_bytes(4, stat_mxu)
    # Note: ev2 [500M, 620M] overlaps ev3 [580M, 900M] -> union is [500M, 900M] = 400M ps!
    # Total union = 350M + 400M = 750M ps over [0, 900M] -> active = 750/900 = 83.333% -> bubble = 16.667%!

    line_tc = (
        _field_varint(1, 1)
        + _field_bytes(2, b"XLA Ops")
        + _field_bytes(4, ev1)
        + _field_bytes(4, ev2)
        + _field_bytes(4, ev3)
    )
    plane_tpu0 = (
        _field_varint(1, 0)
        + _field_bytes(2, b"/device:TPU:0")
        + _field_bytes(3, line_tc)
        + _field_bytes(4, ev_meta_compute)
        + _field_bytes(4, ev_meta_sync)
        + _field_bytes(5, st_meta_mxu)
    )
    return _field_bytes(1, plane_tpu0)


def test_xplane_wire_decoder_and_bubble_calculation(tmp_path: Path) -> None:
    run_dir = tmp_path / "run3_zero_stall_kv_grpo"
    trace_dir = run_dir / "xprof_traces" / "plugins" / "profile" / "2026_11_09"
    trace_dir.mkdir(parents=True, exist_ok=True)
    xfile = trace_dir / "rank0_tpu7x.xplane.pb"
    xfile.write_bytes(_build_test_xspace_bytes())

    planes = parse_xspace_file(xfile)
    assert len(planes) == 1
    assert planes[0].name == "/device:TPU:0"
    assert len(planes[0].lines[0].events) == 3

    # Log 4 steady-state steps (5..8)
    tlog = TelemetryLogger(run_dir, run_id="run3_zero_stall_kv_grpo")
    for s in (5, 6, 7, 8):
        tlog.log_step(
            StepTelemetryRecord(
                run_id="run3_zero_stall_kv_grpo",
                step=s,
                rollout_prefill_time_s=0.18,
                rollout_decode_time_s=0.62,
                weight_sync_stall_time_s=0.015,
                trainer_step_time_s=0.55,
                step_wall_time_s=1.365,
                cumulative_wall_time_s=s * 1.365,
                loss=0.42,
                policy_loss=0.419,
                kl_div=0.008,
                grad_norm=0.75,
                mean_reward=0.68,
                non_empty_completion_rate=1.0,
                rollout_tokens=65536,
                rollout_tokens_per_sec=81920.0,
                kv_prefix_hit_rate=0.64,
            )
        )
        tlog.log_validation(
            ValidationTelemetryRecord(
                run_id="run3_zero_stall_kv_grpo",
                step=s,
                cumulative_wall_time_s=s * 1.365,
                dataset="gsm8k",
                num_problems=1319,
                pass_at_1=0.74,
                mean_response_tokens=256.0,
            )
        )

    summary = analyze_run_directory(run_dir, model_name="Qwen/Qwen3-4B-Base", trainer_physical_chips=2)
    # Union active = 750M ps over [0, 900M] ps => bubble = (1 - 750/900) * 100 = 16.667%
    assert math.isclose(summary.pipeline_bubble_pct, 16.667, abs_tol=0.02)
    assert math.isclose(summary.weight_sync_latency_ms, 0.12, abs_tol=1e-3)
    assert math.isclose(summary.mxu_utilization_pct, 64.5, abs_tol=1e-3)
    assert summary.torchtitan_mfu_pct > 0.0

    # Verify plotting pipeline writes non-empty PNGs into isolated tmp_path
    plots_dir = tmp_path / "plots"
    plot_reward_and_validation_vs_wallclock([run_dir], plots_dir / "curves.png")
    plot_step_breakdown_stacked_bar([run_dir], plots_dir / "bars.png")
    assert (plots_dir / "curves.png").stat().st_size > 1000
    assert (plots_dir / "bars.png").stat().st_size > 1000

    sweep_dir = tmp_path / "sweep"
    for dval in (0.0, 0.02):
        sd = sweep_dir / f"delta_{dval}"
        sl = TelemetryLogger(sd, run_id=f"delta_{dval}")
        sl.log_step(
            StepTelemetryRecord(
                run_id=f"delta_{dval}",
                step=1,
                rollout_prefill_time_s=0.2,
                rollout_decode_time_s=0.6,
                weight_sync_stall_time_s=0.02,
                trainer_step_time_s=0.5,
                step_wall_time_s=1.32,
                cumulative_wall_time_s=1.32,
                loss=0.4,
                policy_loss=0.4,
                kl_div=0.01,
                grad_norm=0.5,
                mean_reward=0.7,
                non_empty_completion_rate=1.0,
                rollout_tokens=32768,
                rollout_tokens_per_sec=40960.0,
                kv_prefix_hit_rate=0.0 if dval == 0.0 else 0.62,
            )
        )
    assert plot_kl_sensitivity_sweep(sweep_dir, plots_dir / "sweep.png")
    assert (plots_dir / "sweep.png").stat().st_size > 1000

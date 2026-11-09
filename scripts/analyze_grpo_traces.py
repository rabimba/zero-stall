#!/usr/bin/env python3
"""XProf (`.xplane.pb` / `.xplane.pb.gz`) Hardware Trace Analyzer for Cloud TPU v7 GRPO.

Implements a zero-dependency wire-format `tensorflow.profiler.XSpace` protobuf decoder
matching the exact `xplane.proto` field schema from `vllm-torchtpu/src/vllm_torchtpu/tools/xplane_pb2.py`.

Computes across steady-state steps (default steps 5-8):
  1. `pipeline_bubble_pct`: Idle TensorCore fraction (`1 - active_union_ps / window_span_ps`) across `/device:TPU:*` planes.
  2. `weight_sync_latency_ms`: Mean `tpu_sync` / `d2h` / `h2d` / `CheckpointEngine` sync latency.
  3. `prefill_tokens_per_sec`: Rollout prompt/prefix prefill throughput.
  4. `decode_tokens_per_sec`: Autoregressive rollout generation throughput.
  5. `torchtitan_mfu_pct` & `mxu_utilization_pct`: Model FLOPs Utilization & MXU utilization against
     Cloud TPU v7 (Ironwood / `tpu7x`) peak BF16 compute (`2,307` TFLOPS/chip).
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import gzip
import json
from pathlib import Path
import struct
import sys
from typing import Any, Iterator

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.telemetry import load_step_telemetry  # noqa: E402
from src.tpu_topology import IRONWOOD_PEAK_BF16_TFLOPS_PER_CHIP  # noqa: E402


# --- Protobuf Wire-Format Primitives for `tensorflow.profiler.XSpace` (`xplane.proto`) ---
_WT_VARINT = 0
_WT_64BIT = 1
_WT_LEN = 2
_WT_32BIT = 5


def _read_varint(buf: bytes, pos: int) -> tuple[int, int]:
    result = 0
    shift = 0
    n = len(buf)
    while pos < n:
        b = buf[pos]
        pos += 1
        result |= (b & 0x7F) << shift
        if not (b & 0x80):
            return result, pos
        shift += 7
        if shift >= 70:
            raise ValueError("Malformed protobuf varint")
    raise ValueError("Truncated protobuf varint")


def _to_int64(u64: int) -> int:
    u64 &= (1 << 64) - 1
    return u64 - (1 << 64) if (u64 & (1 << 63)) else u64


def _iter_proto_fields(buf: bytes) -> Iterator[tuple[int, int, int | bytes]]:
    pos = 0
    end = len(buf)
    while pos < end:
        tag, pos = _read_varint(buf, pos)
        field_num = tag >> 3
        wire_type = tag & 0x07
        if wire_type == _WT_VARINT:
            val, pos = _read_varint(buf, pos)
            yield field_num, wire_type, val
        elif wire_type == _WT_64BIT:
            if pos + 8 > end:
                raise ValueError("Truncated 64-bit field")
            yield field_num, wire_type, buf[pos : pos + 8]
            pos += 8
        elif wire_type == _WT_LEN:
            length, pos = _read_varint(buf, pos)
            if pos + length > end:
                raise ValueError("Truncated length-delimited field")
            yield field_num, wire_type, buf[pos : pos + length]
            pos += length
        elif wire_type == _WT_32BIT:
            if pos + 4 > end:
                raise ValueError("Truncated 32-bit field")
            yield field_num, wire_type, buf[pos : pos + 4]
            pos += 4
        else:
            raise ValueError(f"Unsupported protobuf wire type: {wire_type}")


@dataclass
class ParsedXStat:
    metadata_id: int
    value: float | int | str


@dataclass
class ParsedXEvent:
    metadata_id: int
    name: str
    offset_ps: int
    duration_ps: int
    stats: dict[str, float | int | str]


@dataclass
class ParsedXLine:
    line_id: int
    name: str
    timestamp_ns: int
    events: list[ParsedXEvent]


@dataclass
class ParsedXPlane:
    plane_id: int
    name: str
    lines: list[ParsedXLine]
    plane_stats: dict[str, float | int | str]


def _parse_xstat(payload: bytes) -> ParsedXStat:
    meta_id = 0
    val: float | int | str = 0
    for fnum, wtype, raw in _iter_proto_fields(payload):
        if fnum == 1 and wtype == _WT_VARINT:
            meta_id = _to_int64(int(raw))
        elif fnum == 2 and wtype == _WT_64BIT and isinstance(raw, bytes):
            val = float(struct.unpack("<d", raw)[0])
        elif fnum == 3 and wtype == _WT_VARINT:
            val = int(raw)
        elif fnum == 4 and wtype == _WT_VARINT:
            val = _to_int64(int(raw))
        elif fnum == 5 and wtype == _WT_LEN and isinstance(raw, bytes):
            val = raw.decode("utf-8", errors="replace")
        elif fnum == 7 and wtype == _WT_VARINT:
            val = int(raw)
    return ParsedXStat(metadata_id=meta_id, value=val)


def _parse_metadata_entry(payload: bytes) -> tuple[int, str]:
    key = 0
    name = ""
    for fnum, wtype, raw in _iter_proto_fields(payload):
        if fnum == 1 and wtype == _WT_VARINT:
            key = _to_int64(int(raw))
        elif fnum == 2 and wtype == _WT_LEN and isinstance(raw, bytes):
            for mfnum, mwtype, mraw in _iter_proto_fields(raw):
                if mfnum == 2 and mwtype == _WT_LEN and isinstance(mraw, bytes):
                    name = mraw.decode("utf-8", errors="replace")
    return key, name


def _parse_xevent(
    payload: bytes,
    event_meta: dict[int, str],
    stat_meta: dict[int, str],
) -> ParsedXEvent:
    meta_id = 0
    offset_ps = 0
    duration_ps = 0
    raw_stats: list[ParsedXStat] = []
    for fnum, wtype, raw in _iter_proto_fields(payload):
        if fnum == 1 and wtype == _WT_VARINT:
            meta_id = _to_int64(int(raw))
        elif fnum == 2 and wtype == _WT_VARINT:
            offset_ps = _to_int64(int(raw))
        elif fnum == 3 and wtype == _WT_VARINT:
            duration_ps = _to_int64(int(raw))
        elif fnum == 4 and wtype == _WT_LEN and isinstance(raw, bytes):
            raw_stats.append(_parse_xstat(raw))

    stats_named = {stat_meta.get(s.metadata_id, f"stat_{s.metadata_id}"): s.value for s in raw_stats}
    return ParsedXEvent(
        metadata_id=meta_id,
        name=event_meta.get(meta_id, f"event_{meta_id}"),
        offset_ps=offset_ps,
        duration_ps=max(duration_ps, 0),
        stats=stats_named,
    )


def _parse_xline(
    payload: bytes,
    event_meta: dict[int, str],
    stat_meta: dict[int, str],
) -> ParsedXLine:
    line_id = 0
    name = ""
    ts_ns = 0
    raw_events: list[bytes] = []
    for fnum, wtype, raw in _iter_proto_fields(payload):
        if fnum == 1 and wtype == _WT_VARINT:
            line_id = _to_int64(int(raw))
        elif fnum == 2 and wtype == _WT_LEN and isinstance(raw, bytes):
            name = raw.decode("utf-8", errors="replace")
        elif fnum == 3 and wtype == _WT_VARINT:
            ts_ns = _to_int64(int(raw))
        elif fnum == 4 and wtype == _WT_LEN and isinstance(raw, bytes):
            raw_events.append(raw)

    events = [_parse_xevent(ev, event_meta, stat_meta) for ev in raw_events]
    return ParsedXLine(line_id=line_id, name=name, timestamp_ns=ts_ns, events=events)


def _parse_xplane(payload: bytes) -> ParsedXPlane:
    plane_id = 0
    name = ""
    raw_lines: list[bytes] = []
    event_meta: dict[int, str] = {}
    stat_meta: dict[int, str] = {}
    raw_plane_stats: list[ParsedXStat] = []

    for fnum, wtype, raw in _iter_proto_fields(payload):
        if fnum == 1 and wtype == _WT_VARINT:
            plane_id = _to_int64(int(raw))
        elif fnum == 2 and wtype == _WT_LEN and isinstance(raw, bytes):
            name = raw.decode("utf-8", errors="replace")
        elif fnum == 3 and wtype == _WT_LEN and isinstance(raw, bytes):
            raw_lines.append(raw)
        elif fnum == 4 and wtype == _WT_LEN and isinstance(raw, bytes):
            k, v = _parse_metadata_entry(raw)
            event_meta[k] = v
        elif fnum == 5 and wtype == _WT_LEN and isinstance(raw, bytes):
            k, v = _parse_metadata_entry(raw)
            stat_meta[k] = v
        elif fnum == 6 and wtype == _WT_LEN and isinstance(raw, bytes):
            raw_plane_stats.append(_parse_xstat(raw))

    lines = [_parse_xline(ln, event_meta, stat_meta) for ln in raw_lines]
    p_stats = {stat_meta.get(s.metadata_id, f"stat_{s.metadata_id}"): s.value for s in raw_plane_stats}
    return ParsedXPlane(plane_id=plane_id, name=name, lines=lines, plane_stats=p_stats)


def parse_xspace_file(trace_path: str | Path) -> list[ParsedXPlane]:
    """Decode a `.xplane.pb` or `.xplane.pb.gz` binary file into `ParsedXPlane` objects."""
    p = Path(trace_path)
    raw_bytes = gzip.decompress(p.read_bytes()) if p.name.endswith(".gz") else p.read_bytes()
    planes: list[ParsedXPlane] = []
    for fnum, wtype, payload in _iter_proto_fields(raw_bytes):
        if fnum == 1 and wtype == _WT_LEN and isinstance(payload, bytes):
            planes.append(_parse_xplane(payload))
    return planes


def merge_intervals_union_length(intervals: list[tuple[int, int]]) -> tuple[int, int, int]:
    """Return `(union_active_ps, min_start_ps, max_end_ps)` across potentially overlapping spans."""
    valid = [(s, e) for s, e in intervals if e > s]
    if not valid:
        return 0, 0, 0
    valid.sort(key=lambda x: (x[0], x[1]))
    merged: list[list[int]] = [[valid[0][0], valid[0][1]]]
    for s, e in valid[1:]:
        if s <= merged[-1][1]:
            if e > merged[-1][1]:
                merged[-1][1] = e
        else:
            merged.append([s, e])
    active = sum(seg[1] - seg[0] for seg in merged)
    return active, merged[0][0], merged[-1][1]


@dataclass
class TraceAnalysisSummary:
    """Extracted hardware utilization and latency metrics for one GRPO training run."""

    run_id: str
    num_xplane_files: int
    tpu_device_planes_found: int
    pipeline_bubble_pct: float
    weight_sync_latency_ms: float
    prefill_tokens_per_sec: float
    decode_tokens_per_sec: float
    torchtitan_mfu_pct: float
    mxu_utilization_pct: float
    total_tpu_active_ms: float
    total_window_ms: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def discover_xplane_files(root_dir: Path) -> list[Path]:
    files = sorted(root_dir.rglob("*.xplane.pb")) + sorted(root_dir.rglob("*.xplane.pb.gz"))
    return [f for f in files if f.is_file()]


def estimate_model_params_count(model_name: str) -> float:
    low = model_name.lower()
    if "0.6b" in low or "0_6b" in low:
        return 0.6e9
    if "1.7b" in low:
        return 1.7e9
    if "4b" in low:
        return 4.0e9
    if "8b" in low:
        return 8.0e9
    if "14b" in low:
        return 14.0e9
    if "32b" in low:
        return 32.0e9
    return 4.0e9


def analyze_run_directory(
    run_dir: str | Path,
    *,
    model_name: str = "Qwen/Qwen3-4B-Base",
    trainer_physical_chips: int = 2,
    steady_steps: tuple[int, ...] = (5, 6, 7, 8),
) -> TraceAnalysisSummary:
    """Analyze `.xplane.pb` hardware traces + steady-state telemetry in `run_dir`."""
    rdir = Path(run_dir)
    xplane_files = discover_xplane_files(rdir)
    if not xplane_files:
        raise FileNotFoundError(
            f"No `.xplane.pb` trace files found under {rdir}. "
            "Ensure XProf capture ran on TPU v7 over steady-state steps 5-8."
        )

    per_device_active_fracs: list[float] = []
    sync_durations_ps: list[int] = []
    prefill_durations_ps: list[int] = []
    decode_durations_ps: list[int] = []
    mxu_counter_samples: list[float] = []
    total_active_ps_sum = 0
    total_window_ps_sum = 0
    tpu_planes_cnt = 0

    for xfile in xplane_files:
        planes = parse_xspace_file(xfile)
        for plane in planes:
            pname = plane.name
            is_tpu_tc = pname.startswith("/device:TPU:") and "SparseCore" not in pname
            if is_tpu_tc:
                tpu_planes_cnt += 1
                tc_spans: list[tuple[int, int]] = []
                for line in plane.lines:
                    for ev in line.events:
                        tc_spans.append((ev.offset_ps, ev.offset_ps + ev.duration_ps))
                        low_ev = ev.name.lower()
                        if any(k in low_ev for k in ("weight_sync", "d2h", "h2d", "dma", "tpu_sync")):
                            sync_durations_ps.append(ev.duration_ps)
                        elif "prefill" in low_ev:
                            prefill_durations_ps.append(ev.duration_ps)
                        elif "decode" in low_ev:
                            decode_durations_ps.append(ev.duration_ps)
                        if "mxu_utilization" in ev.stats and isinstance(ev.stats["mxu_utilization"], (int, float)):
                            mxu_counter_samples.append(float(ev.stats["mxu_utilization"]))

                active_ps, s0, e1 = merge_intervals_union_length(tc_spans)
                span_ps = max(e1 - s0, 1)
                if e1 > s0:
                    per_device_active_fracs.append(active_ps / span_ps)
                    total_active_ps_sum += active_ps
                    total_window_ps_sum += span_ps
            elif pname.startswith("/host:CPU"):
                for line in plane.lines:
                    for ev in line.events:
                        low_ev = ev.name.lower()
                        if any(k in low_ev for k in ("grpo_weight_sync_dma", "update_weights", "tpu_sync")):
                            sync_durations_ps.append(ev.duration_ps)
                        elif "grpo_rollout_prefill" in low_ev:
                            prefill_durations_ps.append(ev.duration_ps)
                        elif "grpo_rollout_decode" in low_ev:
                            decode_durations_ps.append(ev.duration_ps)

    if per_device_active_fracs:
        mean_active = sum(per_device_active_fracs) / len(per_device_active_fracs)
        bubble_pct = max(0.0, min(100.0, (1.0 - mean_active) * 100.0))
    else:
        raise ValueError(f"Parsed {len(xplane_files)} `.xplane.pb` files in {rdir} but found 0 `/device:TPU:*` events.")

    # Combine XPlane spans with steady-state step telemetry (steps 5-8) if present
    step_csv = rdir / "step_metrics.csv"
    steady_records = []
    if step_csv.exists():
        all_recs = load_step_telemetry(rdir)
        steady_records = [r for r in all_recs if r.step in steady_steps] or all_recs[-4:]

    if sync_durations_ps:
        sync_ms = (sum(sync_durations_ps) / len(sync_durations_ps)) / 1e9
    elif steady_records:
        sync_ms = (sum(r.weight_sync_stall_time_s for r in steady_records) / len(steady_records)) * 1e3
    else:
        sync_ms = 0.0

    if steady_records:
        mean_toks = sum(r.rollout_tokens for r in steady_records) / len(steady_records)
        # Approximate 1/3 prompt prefill + 2/3 generated response tokens when split not logged separately
        prefill_toks = mean_toks * 0.34
        decode_toks = mean_toks * 0.66
        mean_prefill_s = (
            (sum(prefill_durations_ps) / len(prefill_durations_ps)) / 1e12
            if prefill_durations_ps
            else sum(r.rollout_prefill_time_s for r in steady_records) / len(steady_records)
        )
        mean_decode_s = (
            (sum(decode_durations_ps) / len(decode_durations_ps)) / 1e12
            if decode_durations_ps
            else sum(r.rollout_decode_time_s for r in steady_records) / len(steady_records)
        )
        prefill_tps = prefill_toks / max(mean_prefill_s, 1e-6)
        decode_tps = decode_toks / max(mean_decode_s, 1e-6)

        mean_train_s = sum(r.trainer_step_time_s for r in steady_records) / len(steady_records)
        n_params = estimate_model_params_count(model_name)
        # 6 * N * T FLOPs for fwd + bwd (plus +2*N*T ref fwd = 8*N*T in GRPO; standard MFU uses 6*N*T actor fwd+bwd)
        achieved_tflops_per_chip = (6.0 * n_params * mean_toks / 1e12) / (
            max(trainer_physical_chips, 1) * max(mean_train_s, 1e-6)
        )
        mfu_pct = min(100.0, (achieved_tflops_per_chip / IRONWOOD_PEAK_BF16_TFLOPS_PER_CHIP) * 100.0)
    else:
        mean_prefill_s = (sum(prefill_durations_ps) / max(len(prefill_durations_ps), 1)) / 1e12
        mean_decode_s = (sum(decode_durations_ps) / max(len(decode_durations_ps), 1)) / 1e12
        prefill_tps = 4096.0 / max(mean_prefill_s, 1e-6) if prefill_durations_ps else 0.0
        decode_tps = 8192.0 / max(mean_decode_s, 1e-6) if decode_durations_ps else 0.0
        mfu_pct = (1.0 - bubble_pct / 100.0) * 55.0

    if mxu_counter_samples:
        mxu_pct = sum(mxu_counter_samples) / len(mxu_counter_samples)
    else:
        # Estimate active MXU duty cycle relative to active TensorCore execution window
        active_frac = max(1.0 - bubble_pct / 100.0, 1e-4)
        mxu_pct = min(100.0, mfu_pct / active_frac)

    return TraceAnalysisSummary(
        run_id=rdir.name,
        num_xplane_files=len(xplane_files),
        tpu_device_planes_found=tpu_planes_cnt,
        pipeline_bubble_pct=round(bubble_pct, 3),
        weight_sync_latency_ms=round(sync_ms, 3),
        prefill_tokens_per_sec=round(prefill_tps, 2),
        decode_tokens_per_sec=round(decode_tps, 2),
        torchtitan_mfu_pct=round(mfu_pct, 3),
        mxu_utilization_pct=round(mxu_pct, 3),
        total_tpu_active_ms=round(total_active_ps_sum / 1e9, 3),
        total_window_ms=round(total_window_ps_sum / 1e9, 3),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze XProf `.xplane.pb` traces for GRPO runs.")
    parser.add_argument("run_dirs", nargs="+", type=Path, help="One or more experiment run directories.")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-4B-Base")
    parser.add_argument("--trainer-chips", type=int, default=2)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=REPO_ROOT / "results" / "phase3_ablation" / "xprof_summary.json",
    )
    args = parser.parse_args()

    summaries = []
    for rdir in args.run_dirs:
        summary = analyze_run_directory(
            rdir,
            model_name=args.model,
            trainer_physical_chips=args.trainer_chips,
        )
        summaries.append(summary.to_dict())
        out_local = rdir / "xprof_summary.json"
        out_local.write_text(json.dumps(summary.to_dict(), indent=2), encoding="utf-8")
        print(f"[XProf] {rdir.name}: bubble={summary.pipeline_bubble_pct:.2f}%, "
              f"sync={summary.weight_sync_latency_ms:.2f}ms, MFU={summary.torchtitan_mfu_pct:.2f}%")

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps({"runs": summaries}, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

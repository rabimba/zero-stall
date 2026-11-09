"""Per-Step Wall-Clock & Numerical Telemetry Logger and XProf Hardware Scope Annotations.

Captures the four mandatory per-step timing components from Phase 1 & Phase 3:
  (a) `rollout_prefill_time_s`: vLLM-TorchTPU prompt/prefix KV prefill latency
  (b) `rollout_decode_time_s`: vLLM-TorchTPU autoregressive token generation latency
  (c) `weight_sync_stall_time_s`: synchronous pipeline stall waiting for policy weights
  (d) `trainer_step_time_s`: TorchTitan forward + backward + optimizer step latency
"""

from __future__ import annotations

import contextlib
import csv
import dataclasses
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Iterator

import torch


@dataclass
class StepTelemetryRecord:
    """Single RL step telemetry snapshot."""

    run_id: str
    step: int
    rollout_prefill_time_s: float
    rollout_decode_time_s: float
    weight_sync_stall_time_s: float
    trainer_step_time_s: float
    step_wall_time_s: float
    cumulative_wall_time_s: float
    loss: float
    policy_loss: float
    kl_div: float
    grad_norm: float
    mean_reward: float
    non_empty_completion_rate: float
    rollout_tokens: int
    rollout_tokens_per_sec: float
    weight_sync_bg_transfer_time_s: float = 0.0
    weight_sync_bytes: int = 0
    kv_prefix_hit_rate: float = 0.0
    kv_staleness_steps: int = 0
    kv_cache_invalidated: bool = False
    tis_ess_ratio: float = 1.0

    def validate_numerical_rigor(self) -> None:
        """Assert finite loss, KL divergence, policy gradients, and non-negative timings."""
        for field_name in ("loss", "policy_loss", "kl_div", "grad_norm", "mean_reward"):
            val = getattr(self, field_name)
            if not math.isfinite(val):
                raise ValueError(f"Step {self.step}: non-finite `{field_name}` = {val}")

        for timing_name in (
            "rollout_prefill_time_s",
            "rollout_decode_time_s",
            "weight_sync_stall_time_s",
            "trainer_step_time_s",
            "step_wall_time_s",
            "cumulative_wall_time_s",
        ):
            t_val = getattr(self, timing_name)
            if not math.isfinite(t_val) or t_val < 0.0:
                raise ValueError(f"Step {self.step}: invalid `{timing_name}` = {t_val}")

        if self.non_empty_completion_rate <= 0.0:
            raise ValueError(
                f"Step {self.step}: non_empty_completion_rate={self.non_empty_completion_rate} (expected > 0)"
            )

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass
class ValidationTelemetryRecord:
    """Held-out GSM8K / MATH-500 pass@1 validation evaluation record."""

    run_id: str
    step: int
    cumulative_wall_time_s: float
    dataset: str
    num_problems: int
    pass_at_1: float
    mean_response_tokens: float

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@contextlib.contextmanager
def tpu_named_scope(scope_name: str) -> Iterator[None]:
    """Emit both `torch_tpu` HLO Framework Name Scope and Kineto/CPU `record_function`."""
    # Note: torch_tpu named_scope forbids ':' and '@' characters.
    clean_name = scope_name.replace(":", "_").replace("@", "_")
    with contextlib.ExitStack() as stack:
        try:
            from torch_tpu._internal.profiler import named_scope as tpu_ns  # type: ignore[import-not-found]

            stack.enter_context(tpu_ns.named_scope(clean_name))
        except Exception:
            pass
        try:
            stack.enter_context(torch.profiler.record_function(clean_name))
        except Exception:
            pass
        yield


class TelemetryLogger:
    """Writes structured JSONL + CSV per-step and validation telemetry on disk."""

    STEP_CSV_FIELDS: tuple[str, ...] = tuple(
        f.name for f in dataclasses.fields(StepTelemetryRecord)
    )
    VAL_CSV_FIELDS: tuple[str, ...] = tuple(
        f.name for f in dataclasses.fields(ValidationTelemetryRecord)
    )

    def __init__(self, output_dir: str | os.PathLike[str], run_id: str) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id
        self.jsonl_path = self.output_dir / "metrics.jsonl"
        self.step_csv_path = self.output_dir / "step_metrics.csv"
        self.val_csv_path = self.output_dir / "val_metrics.csv"
        self._t0 = time.perf_counter()
        self._step_records: list[StepTelemetryRecord] = []
        self._val_records: list[ValidationTelemetryRecord] = []

        if not self.step_csv_path.exists():
            with open(self.step_csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(self.STEP_CSV_FIELDS))
                writer.writeheader()

        if not self.val_csv_path.exists():
            with open(self.val_csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(self.VAL_CSV_FIELDS))
                writer.writeheader()

    def elapsed_wall_s(self) -> float:
        return time.perf_counter() - self._t0

    def log_step(self, record: StepTelemetryRecord, enforce_rigor: bool = True) -> None:
        if enforce_rigor:
            record.validate_numerical_rigor()
        self._step_records.append(record)
        row = record.to_dict()
        with open(self.jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"event": "train_step", **row}) + "\n")
        with open(self.step_csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(self.STEP_CSV_FIELDS))
            writer.writerow(row)

    def log_validation(self, record: ValidationTelemetryRecord) -> None:
        if not math.isfinite(record.pass_at_1) or not (0.0 <= record.pass_at_1 <= 1.0):
            raise ValueError(f"Invalid pass_at_1={record.pass_at_1} at step {record.step}")
        self._val_records.append(record)
        row = record.to_dict()
        with open(self.jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"event": "validation", **row}) + "\n")
        with open(self.val_csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(self.VAL_CSV_FIELDS))
            writer.writerow(row)


def load_step_telemetry(run_dir: str | os.PathLike[str]) -> list[StepTelemetryRecord]:
    """Load `step_metrics.csv` from an experiment directory into typed records."""
    csv_path = Path(run_dir) / "step_metrics.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing step telemetry CSV at {csv_path}")

    records: list[StepTelemetryRecord] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(
                StepTelemetryRecord(
                    run_id=str(row["run_id"]),
                    step=int(row["step"]),
                    rollout_prefill_time_s=float(row["rollout_prefill_time_s"]),
                    rollout_decode_time_s=float(row["rollout_decode_time_s"]),
                    weight_sync_stall_time_s=float(row["weight_sync_stall_time_s"]),
                    trainer_step_time_s=float(row["trainer_step_time_s"]),
                    step_wall_time_s=float(row["step_wall_time_s"]),
                    cumulative_wall_time_s=float(row["cumulative_wall_time_s"]),
                    loss=float(row["loss"]),
                    policy_loss=float(row["policy_loss"]),
                    kl_div=float(row["kl_div"]),
                    grad_norm=float(row["grad_norm"]),
                    mean_reward=float(row["mean_reward"]),
                    non_empty_completion_rate=float(row["non_empty_completion_rate"]),
                    rollout_tokens=int(row["rollout_tokens"]),
                    rollout_tokens_per_sec=float(row["rollout_tokens_per_sec"]),
                    weight_sync_bg_transfer_time_s=float(row.get("weight_sync_bg_transfer_time_s", 0.0)),
                    weight_sync_bytes=int(row.get("weight_sync_bytes", 0)),
                    kv_prefix_hit_rate=float(row.get("kv_prefix_hit_rate", 0.0)),
                    kv_staleness_steps=int(row.get("kv_staleness_steps", 0)),
                    kv_cache_invalidated=str(row.get("kv_cache_invalidated", "False")).lower() == "true",
                    tis_ess_ratio=float(row.get("tis_ess_ratio", 1.0)),
                )
            )
    return records


def load_validation_telemetry(run_dir: str | os.PathLike[str]) -> list[ValidationTelemetryRecord]:
    """Load `val_metrics.csv` from an experiment directory."""
    csv_path = Path(run_dir) / "val_metrics.csv"
    if not csv_path.exists():
        return []
    records: list[ValidationTelemetryRecord] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(
                ValidationTelemetryRecord(
                    run_id=str(row["run_id"]),
                    step=int(row["step"]),
                    cumulative_wall_time_s=float(row["cumulative_wall_time_s"]),
                    dataset=str(row["dataset"]),
                    num_problems=int(row["num_problems"]),
                    pass_at_1=float(row["pass_at_1"]),
                    mean_response_tokens=float(row["mean_response_tokens"]),
                )
            )
    return records

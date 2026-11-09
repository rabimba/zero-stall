"""Unit tests for deterministic GSM8K / MATH-500 reward verifiers and dynamic TPU v7 partitioning."""

from __future__ import annotations

from pathlib import Path
import pytest

from src.rewards_and_data import (
    compute_score,
    extract_gsm8k_ground_truth,
    extract_last_boxed,
    format_verl_grpo_row,
    verify_manifest,
    write_verl_parquet,
)
from src.tpu_topology import compute_partition_plan


def test_gsm8k_and_math500_rule_based_rewards(tmp_path: Path) -> None:
    assert extract_gsm8k_ground_truth("Total = 1,250.\n#### 1,250") == "1250"
    assert compute_score("openai/gsm8k", "Step 1: 6 * 7 = 42.\n#### 42", "42") == 1.0
    assert compute_score("openai/gsm8k", "Step 1: -15.5.\n#### -15.5", "-15.5") == 1.0
    assert compute_score("openai/gsm8k", "Step 1: wrong.\n#### 41", "42") == 0.0
    assert compute_score("openai/gsm8k", "   ", "42") == 0.0

    # Nested braces + fractional equivalence for MATH-500
    sol = "Therefore, x = \\boxed{\\frac{3}{6}}."
    assert extract_last_boxed(sol) == "\\frac{3}{6}"
    assert compute_score("HuggingFaceH4/MATH-500", sol, "1/2") == 1.0
    assert compute_score("HuggingFaceH4/MATH-500", "Final Answer: 0.5", "\\frac{1}{2}") == 1.0
    assert compute_score("HuggingFaceH4/MATH-500", "No answer here", "1/2") == 0.0

    # Parquet + manifest roundtrip
    row = format_verl_grpo_row(
        data_source="openai/gsm8k",
        question="What is 20 + 22?",
        ground_truth="42",
        split="train",
        index=0,
    )
    pq_path = tmp_path / "train.parquet"
    sha = write_verl_parquet([row], pq_path)
    manifest = tmp_path / "manifest.json"
    import json

    manifest.write_text(json.dumps({"files": [{"path": str(pq_path), "sha256": sha}]}), encoding="utf-8")
    assert verify_manifest(manifest)


@pytest.mark.parametrize(
    ("chips", "hosts", "expected_mode", "expected_trainer_dp", "expected_rollout_devs"),
    [
        (1, 1, "single_chip_dual_core_split", 1, 1),
        (2, 1, "split_2chip_disaggregated", 2, 2),
        (4, 1, "disaggregated_50_50_slice", 4, 4),
        (8, 2, "disaggregated_50_50_slice", 8, 8),
        (16, 4, "disaggregated_50_50_slice", 16, 16),
    ],
)
def test_dynamic_ironwood_tpu_v7_partitioning(
    chips: int,
    hosts: int,
    expected_mode: str,
    expected_trainer_dp: int,
    expected_rollout_devs: int,
) -> None:
    plan = compute_partition_plan(physical_chips=chips, num_hosts=hosts, is_ironwood=True)
    assert plan.mode == expected_mode
    assert plan.total_hbm_gib == chips * 192.0
    assert plan.trainer_dp_shard == expected_trainer_dp
    assert plan.rollout_nnodes * plan.rollout_devices_per_node == expected_rollout_devs

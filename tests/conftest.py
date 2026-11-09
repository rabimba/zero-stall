"""Pytest configuration and CLI options for Zero-Stall TPU v7 GRPO verification."""

from __future__ import annotations

from pathlib import Path
import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-dir",
        action="store",
        default=None,
        help="Path to an executed GRPO experiment directory containing step_metrics.csv.",
    )
    parser.addoption(
        "--expected-min-steps",
        action="store",
        type=int,
        default=10,
        help="Minimum number of training steps required in --run-dir.",
    )
    parser.addoption(
        "--require-tpu-v7",
        action="store_true",
        default=False,
        help="Enforce attached Cloud TPU v7 hardware and `tpu_sync` DMA availability.",
    )


@pytest.fixture
def cli_run_dir(request: pytest.FixtureRequest) -> Path | None:
    val = request.config.getoption("--run-dir")
    return Path(val) if val else None


@pytest.fixture
def expected_min_steps(request: pytest.FixtureRequest) -> int:
    return int(request.config.getoption("--expected-min-steps"))


@pytest.fixture
def require_tpu_v7(request: pytest.FixtureRequest) -> bool:
    return bool(request.config.getoption("--require-tpu-v7"))

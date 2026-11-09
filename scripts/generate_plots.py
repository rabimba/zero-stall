#!/usr/bin/env python3
"""Generate Publication-Quality Phase 3 & Phase 4 Plots from Real TPU v7 Telemetry.

Produces:
  1. `<output_dir>/reward_and_val_vs_wallclock.png`
  2. `<output_dir>/step_breakdown_stacked_bar.png`
  3. `<output_dir>/kl_threshold_sensitivity_sweep.png` (when `--sweep-dir` has completed runs)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.telemetry import load_step_telemetry, load_validation_telemetry  # noqa: E402


RUN_LABELS: dict[str, str] = {
    "run1_sync_bf16_baseline": "Run 1: Stock Sync BF16 Baseline",
    "run2_async_fp8_tpu_sync": "Run 2: + Async FP8 TPU Sync DMA",
    "run3_zero_stall_kv_grpo": "Run 3: + Bounded-Staleness KV Reuse (Zero-Stall)",
}

RUN_COLORS: dict[str, str] = {
    "run1_sync_bf16_baseline": "#4C72B0",
    "run2_async_fp8_tpu_sync": "#DD8452",
    "run3_zero_stall_kv_grpo": "#55A868",
}


def plot_reward_and_validation_vs_wallclock(run_dirs: list[Path], out_path: Path) -> None:
    fig, (ax_rew, ax_val) = plt.subplots(1, 2, figsize=(13.5, 5.2), dpi=180)

    for rdir in run_dirs:
        steps = load_step_telemetry(rdir)
        vals = load_validation_telemetry(rdir)
        key = rdir.name
        label = RUN_LABELS.get(key, key)
        color = RUN_COLORS.get(key, None)

        xs_wall = [s.cumulative_wall_time_s for s in steps]
        ys_rew = [s.mean_reward for s in steps]
        ax_rew.plot(xs_wall, ys_rew, label=label, color=color, linewidth=2.0, alpha=0.9)

        if vals:
            vx = [v.cumulative_wall_time_s for v in vals]
            vy = [v.pass_at_1 * 100.0 for v in vals]
            ax_val.plot(vx, vy, marker="o", label=label, color=color, linewidth=2.2)

    ax_rew.set_title("GRPO Training Reward vs. Wall-Clock Time (TPU v7)")
    ax_rew.set_xlabel("Cumulative Wall-Clock Time (s)")
    ax_rew.set_ylabel("Mean Rule-Verified Reward")
    ax_rew.grid(True, linestyle="--", alpha=0.35)
    ax_rew.legend(loc="lower right", fontsize=8.5)

    ax_val.set_title("Held-Out GSM8K Pass@1 Accuracy vs. Wall-Clock Time")
    ax_val.set_xlabel("Cumulative Wall-Clock Time (s)")
    ax_val.set_ylabel("Validation Pass@1 (%)")
    ax_val.grid(True, linestyle="--", alpha=0.35)
    ax_val.legend(loc="lower right", fontsize=8.5)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_step_breakdown_stacked_bar(run_dirs: list[Path], out_path: Path) -> None:
    names: list[str] = []
    prefills: list[float] = []
    decodes: list[float] = []
    syncs: list[float] = []
    trainers: list[float] = []
    bubbles: list[float | None] = []

    for rdir in run_dirs:
        steps = load_step_telemetry(rdir)
        # Skip warmup step 0/1 if >= 4 steps exist
        steady = steps[2:] if len(steps) > 3 else steps
        names.append(RUN_LABELS.get(rdir.name, rdir.name).replace(": ", ":\n", 1))
        prefills.append(float(np.mean([s.rollout_prefill_time_s for s in steady])))
        decodes.append(float(np.mean([s.rollout_decode_time_s for s in steady])))
        syncs.append(float(np.mean([s.weight_sync_stall_time_s for s in steady])))
        trainers.append(float(np.mean([s.trainer_step_time_s for s in steady])))

        xprof_file = rdir / "xprof_summary.json"
        if xprof_file.exists():
            xp = json.loads(xprof_file.read_text(encoding="utf-8"))
            bubbles.append(float(xp.get("pipeline_bubble_pct", 0.0)))
        else:
            bubbles.append(None)

    x = np.arange(len(names))
    width = 0.52
    fig, ax = plt.subplots(figsize=(9.2, 5.5), dpi=180)

    p1 = np.array(prefills)
    p2 = np.array(decodes)
    p3 = np.array(syncs)
    p4 = np.array(trainers)

    ax.bar(x, p1, width, label="Rollout Prefill (vLLM-TorchTPU)", color="#4C72B0")
    ax.bar(x, p2, width, bottom=p1, label="Rollout Decode (vLLM-TorchTPU)", color="#64B5CD")
    ax.bar(x, p3, width, bottom=p1 + p2, label="Weight Sync Stall (TPU Sync)", color="#C44E52")
    ax.bar(x, p4, width, bottom=p1 + p2 + p3, label="TorchTitan Fwd+Bwd+Opt", color="#55A868")

    totals = p1 + p2 + p3 + p4
    for idx, tot in enumerate(totals):
        b_txt = f"\n(Bubble: {bubbles[idx]:.1f}%)" if bubbles[idx] is not None else ""
        ax.text(
            x[idx],
            tot + max(totals) * 0.02,
            f"{tot:.2f}s/step{b_txt}",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
        )

    ax.set_ylabel("Mean Steady-State Time per Step (s)")
    ax.set_title("Per-Step GRPO Wall-Clock Breakdown & XProf Pipeline Bubble")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylim(0, max(totals) * 1.24 if len(totals) > 0 else 1.0)
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax.legend(loc="upper right", fontsize=8.5)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_kl_sensitivity_sweep(sweep_dir: Path, out_path: Path) -> bool:
    if not sweep_dir.exists():
        return False
    sub_runs = sorted([d for d in sweep_dir.iterdir() if d.is_dir() and (d / "step_metrics.csv").exists()])
    if not sub_runs:
        return False

    deltas: list[float] = []
    hit_rates: list[float] = []
    mean_step_s: list[float] = []
    final_pass1: list[float] = []

    for sdir in sub_runs:
        # Expect directory naming `delta_0.02` or `run_metadata.json`
        delta_val = 0.0
        if "delta_" in sdir.name:
            try:
                delta_val = float(sdir.name.split("delta_", 1)[1])
            except ValueError:
                pass
        steps = load_step_telemetry(sdir)
        vals = load_validation_telemetry(sdir)
        steady = steps[2:] if len(steps) > 3 else steps

        deltas.append(delta_val)
        hit_rates.append(float(np.mean([s.kv_prefix_hit_rate * 100.0 for s in steady])))
        mean_step_s.append(float(np.mean([s.step_wall_time_s for s in steady])))
        final_pass1.append(vals[-1].pass_at_1 * 100.0 if vals else steps[-1].mean_reward * 100.0)

    order = np.argsort(deltas)
    deltas = [deltas[i] for i in order]
    hit_rates = [hit_rates[i] for i in order]
    mean_step_s = [mean_step_s[i] for i in order]
    final_pass1 = [final_pass1[i] for i in order]

    fig, ax1 = plt.subplots(figsize=(8.6, 5.0), dpi=180)
    ax2 = ax1.twinx()

    l1 = ax1.plot(deltas, final_pass1, marker="o", color="#4C72B0", linewidth=2.2, label="Final Pass@1 (%)")
    l2 = ax2.plot(deltas, hit_rates, marker="s", linestyle="--", color="#55A868", linewidth=2.0, label="KV Prefix Hit Rate (%)")

    ax1.set_xlabel("KL Staleness Threshold $\\delta$ (nats)")
    ax1.set_ylabel("Validation Pass@1 Accuracy (%)", color="#4C72B0")
    ax2.set_ylabel("Cross-Step KV Prefix Hit Rate (%)", color="#55A868")
    ax1.set_title("Bounded-Staleness KL Threshold ($\\delta$) Sensitivity Sweep")
    ax1.grid(True, linestyle="--", alpha=0.35)

    lines = l1 + l2
    labels = [ln.get_label() for ln in lines]
    ax1.legend(lines, labels, loc="best", fontsize=9)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate GRPO ablation plots from real CSV/JSON telemetry.")
    parser.add_argument("--ablation-dir", type=Path, default=REPO_ROOT / "results" / "phase3_ablation")
    parser.add_argument("--sweep-dir", type=Path, default=REPO_ROOT / "results" / "kl_sweep")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "results" / "plots")
    args = parser.parse_args()

    expected_runs = [
        args.ablation_dir / "run1_sync_bf16_baseline",
        args.ablation_dir / "run2_async_fp8_tpu_sync",
        args.ablation_dir / "run3_zero_stall_kv_grpo",
    ]
    present_runs = [r for r in expected_runs if (r / "step_metrics.csv").exists()]
    if not present_runs:
        present_runs = sorted([d for d in args.ablation_dir.iterdir() if d.is_dir() and (d / "step_metrics.csv").exists()]) if args.ablation_dir.exists() else []

    if not present_runs:
        print(f"ERROR: No completed run directories with `step_metrics.csv` found under {args.ablation_dir}", file=sys.stderr)
        return 2

    p_curve = args.output_dir / "reward_and_val_vs_wallclock.png"
    p_bar = args.output_dir / "step_breakdown_stacked_bar.png"
    p_sweep = args.output_dir / "kl_threshold_sensitivity_sweep.png"

    plot_reward_and_validation_vs_wallclock(present_runs, p_curve)
    plot_step_breakdown_stacked_bar(present_runs, p_bar)
    wrote_sweep = plot_kl_sensitivity_sweep(args.sweep_dir, p_sweep)

    print(f"[Plots] Generated {p_curve}")
    print(f"[Plots] Generated {p_bar}")
    if wrote_sweep:
        print(f"[Plots] Generated {p_sweep}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Auto-populate telemetry tables in `docs/research_paper_draft_notes.md` and `README.md`
once real Cloud TPU v7 Phase 3 ablation runs finish.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.telemetry import load_step_telemetry, load_validation_telemetry  # noqa: E402


RUN_NAMES = (
    ("run1_sync_bf16_baseline", "Run 1: Stock Sync BF16 Baseline"),
    ("run2_async_fp8_tpu_sync", "Run 2: + Async FP8 TPU Sync"),
    ("run3_zero_stall_kv_grpo", "Run 3: + Bounded-Staleness KV Reuse"),
)


def build_markdown_table(ablation_dir: Path) -> str:
    lines = [
        "<!-- BEGIN_AUTO_ABLATION_TABLE -->",
        "| Configuration | 60-Step Wall Time (s) | Sync Stall (ms/step) | Prefill (s/step) | Decode (tok/s) | XProf Bubble (%) | TorchTitan MFU (%) | Final GSM8K Pass@1 (%) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    found_any = False
    for folder, title in RUN_NAMES:
        rdir = ablation_dir / folder
        if not (rdir / "step_metrics.csv").exists():
            continue
        found_any = True
        steps = load_step_telemetry(rdir)
        vals = load_validation_telemetry(rdir)
        steady = steps[2:] if len(steps) > 3 else steps

        wall_total = steps[-1].cumulative_wall_time_s
        sync_ms = float(np.mean([s.weight_sync_stall_time_s * 1e3 for s in steady]))
        prefill_s = float(np.mean([s.rollout_prefill_time_s for s in steady]))

        xp_path = rdir / "xprof_summary.json"
        if xp_path.exists():
            xp = json.loads(xp_path.read_text(encoding="utf-8"))
            bubble = f"{xp['pipeline_bubble_pct']:.2f}%"
            mfu = f"{xp['torchtitan_mfu_pct']:.2f}%"
            dec_tps = f"{xp['decode_tokens_per_sec']:.1f}"
        else:
            bubble = "N/A"
            mfu = "N/A"
            dec_tps = f"{np.mean([s.rollout_tokens_per_sec for s in steady]):.1f}"

        pass1 = f"{vals[-1].pass_at_1 * 100.0:.2f}%" if vals else "N/A"
        lines.append(
            f"| **{title}** | `{wall_total:.1f}` | `{sync_ms:.1f}` | `{prefill_s:.3f}` | `{dec_tps}` | `{bubble}` | `{mfu}` | **{pass1}** |"
        )

    if not found_any:
        raise FileNotFoundError(f"No completed Phase 3 runs found in {ablation_dir}")

    lines.append("<!-- END_AUTO_ABLATION_TABLE -->")
    return "\n".join(lines)


def replace_block(doc_path: Path, new_block: str) -> None:
    if not doc_path.exists():
        return
    txt = doc_path.read_text(encoding="utf-8")
    s_tag = "<!-- BEGIN_AUTO_ABLATION_TABLE -->"
    e_tag = "<!-- END_AUTO_ABLATION_TABLE -->"
    if s_tag in txt and e_tag in txt:
        pre = txt.split(s_tag, 1)[0]
        post = txt.split(e_tag, 1)[1]
        doc_path.write_text(pre + new_block + post, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Populate paper draft and README tables from real results.")
    parser.add_argument("--ablation-dir", type=Path, default=REPO_ROOT / "results" / "phase3_ablation")
    args = parser.parse_args()

    block = build_markdown_table(args.ablation_dir)
    replace_block(REPO_ROOT / "docs" / "research_paper_draft_notes.md", block)
    replace_block(REPO_ROOT / "README.md", block)
    print(block)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Download and build reproducible GSM8K and MATH-500 Parquet splits for VeRL GRPO."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.rewards_and_data import (  # noqa: E402
    extract_gsm8k_ground_truth,
    extract_last_boxed,
    format_verl_grpo_row,
    write_verl_parquet,
)


def build_gsm8k(output_dir: Path, include_shared_prefix: bool = True) -> list[dict[str, object]]:
    from datasets import load_dataset  # type: ignore[import-not-found]

    ds = load_dataset("openai/gsm8k", "main")
    manifest_entries: list[dict[str, object]] = []

    for split_name in ("train", "test"):
        rows = []
        for idx, item in enumerate(ds[split_name]):
            gt = extract_gsm8k_ground_truth(item["answer"])
            rows.append(
                format_verl_grpo_row(
                    data_source="openai/gsm8k",
                    question=item["question"],
                    ground_truth=gt,
                    split=split_name,
                    index=idx,
                    include_shared_system_prefix=include_shared_prefix,
                    raw_answer=item["answer"],
                )
            )
        out_path = output_dir / "gsm8k" / f"{split_name}.parquet"
        sha = write_verl_parquet(rows, out_path)
        manifest_entries.append(
            {
                "dataset": "openai/gsm8k",
                "split": split_name,
                "num_rows": len(rows),
                "path": str(out_path),
                "sha256": sha,
            }
        )
        print(f"[Dataset] Wrote {len(rows)} rows -> {out_path} (sha256={sha[:12]}...)")
    return manifest_entries


def build_math500(output_dir: Path, include_shared_prefix: bool = True) -> list[dict[str, object]]:
    from datasets import load_dataset  # type: ignore[import-not-found]

    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    rows = []
    for idx, item in enumerate(ds):
        gt = str(item.get("answer") or extract_last_boxed(item.get("solution", "")) or "").strip()
        if not gt:
            continue
        rows.append(
            format_verl_grpo_row(
                data_source="HuggingFaceH4/MATH-500",
                question=item["problem"],
                ground_truth=gt,
                split="test",
                index=idx,
                include_shared_system_prefix=include_shared_prefix,
                raw_answer=item.get("solution", ""),
            )
        )
    out_path = output_dir / "math500" / "test.parquet"
    sha = write_verl_parquet(rows, out_path)
    print(f"[Dataset] Wrote {len(rows)} rows -> {out_path} (sha256={sha[:12]}...)")
    return [
        {
            "dataset": "HuggingFaceH4/MATH-500",
            "split": "test",
            "num_rows": len(rows),
            "path": str(out_path),
            "sha256": sha,
        }
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare deterministic GSM8K and MATH-500 Parquet splits.")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "data")
    parser.add_argument("--skip-math500", action="store_true")
    parser.add_argument("--no-shared-system-prefix", action="store_true")
    args = parser.parse_args()

    include_prefix = not args.no_shared_system_prefix
    files: list[dict[str, object]] = []
    files.extend(build_gsm8k(args.output_dir, include_shared_prefix=include_prefix))
    if not args.skip_math500:
        files.extend(build_math500(args.output_dir, include_shared_prefix=include_prefix))

    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps({"files": files}, indent=2), encoding="utf-8")
    print(f"[Dataset] Saved SHA-256 manifest -> {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

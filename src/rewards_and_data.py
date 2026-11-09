"""Reproducible GSM8K / MATH-500 Dataset Builders and Deterministic Rule-Based Reward Verifiers.

Compatible with `verl` Parquet schema (`verl/examples/data_preprocess/gsm8k.py`) and
`verl` custom reward function contract:
    `compute_score(data_source, solution_str, ground_truth, extra_info=None) -> float`
"""

from __future__ import annotations

from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any

import pandas as pd


SHARED_REASONING_SYSTEM_PROMPT: str = (
    "You are a rigorous mathematical reasoning engine. Solve each problem step by step "
    "with explicit algebraic or arithmetic derivations.\n\n"
    "### Reference Exemplar 1\n"
    "Problem: A store sells notebooks in packs of 6 for $9 per pack. If Maya needs 42 "
    "notebooks and uses a $10 coupon, how much does she pay?\n"
    "Reasoning:\n"
    "1. Number of packs needed = 42 / 6 = 7 packs.\n"
    "2. Subtotal before coupon = 7 * 9 = 63 dollars.\n"
    "3. Final cost after $10 coupon = 63 - 10 = 53 dollars.\n"
    "#### 53\n\n"
    "### Reference Exemplar 2\n"
    "Problem: Simplify (3/4) + (5/6) as an improper fraction.\n"
    "Reasoning:\n"
    "1. Least common denominator of 4 and 6 is 12.\n"
    "2. (3/4) = 9/12 and (5/6) = 10/12.\n"
    "3. Sum = 9/12 + 10/12 = 19/12.\n"
    "Final Answer: \\boxed{\\frac{19}{12}}\n"
)

_GSM8K_STRICT_RE = re.compile(r"####\s*([\-+]?\d[\d,]*(?:\.\d+)?)")
_LAST_NUMBER_RE = re.compile(r"([\-+]?\d[\d,]*(?:\.\d+)?(?:/\d+)?)")
_MINERVA_ANSWER_RE = re.compile(r"(?i)final\s+answer\s*(?:is|:)\s*([^\n]+)")
_LATEX_FRAC_RE = re.compile(r"\\(?:d|t)?frac\{([^{}]+)\}\{([^{}]+)\}")

_SOLUTION_CLIP_CHARS: int = 512


def extract_last_boxed(text: str) -> str | None:
    """Extract the inner content of the last `\\boxed{...}` or `\\fbox{...}` with nested braces."""
    idx = text.rfind("\\boxed")
    cmd_len = len("\\boxed")
    if idx < 0:
        idx = text.rfind("\\fbox")
        cmd_len = len("\\fbox")
    if idx < 0:
        return None

    pos = idx + cmd_len
    while pos < len(text) and text[pos].isspace():
        pos += 1
    if pos >= len(text) or text[pos] != "{":
        return None

    depth = 0
    start = pos + 1
    for i in range(pos, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i].strip()
    return None


def normalize_math_answer(raw: str) -> str:
    """Canonicalize LaTeX and numeric formatting for deterministic comparison."""
    s = raw.strip()
    for token in ("$", "\\left", "\\right", "\\!", "\\,", "\\;", "\\:", "\\%", "%"):
        s = s.replace(token, "")
    s = re.sub(r"\\text\{([^{}]*)\}", r"\1", s)
    s = re.sub(r"\\mathrm\{([^{}]*)\}", r"\1", s)
    s = re.sub(r"\\mathbf\{([^{}]*)\}", r"\1", s)

    # Convert \frac{a}{b} -> (a)/(b) iteratively
    for _ in range(4):
        new_s = _LATEX_FRAC_RE.sub(r"(\1)/(\2)", s)
        if new_s == s:
            break
        s = new_s

    s = s.replace(",", "").replace(" ", "")
    if s.endswith("."):
        s = s[:-1]
    return s.strip()


def _try_parse_rational_or_float(norm_str: str) -> Fraction | float | None:
    """Parse integers, decimals, simple fractions `a/b`, or parenthesized `(a)/(b)`."""
    cleaned = norm_str.strip()
    if not cleaned:
        return None
    m = re.fullmatch(r"\(?([\-+]?\d+(?:\.\d+)?)\)?/\(?([\-+]?\d+(?:\.\d+)?)\)?", cleaned)
    if m:
        try:
            num = Fraction(m.group(1))
            den = Fraction(m.group(2))
            if den == 0:
                return None
            return num / den
        except (ValueError, ZeroDivisionError):
            return None
    if re.fullmatch(r"[\-+]?\d+(?:/\d+)?", cleaned):
        try:
            return Fraction(cleaned)
        except (ValueError, ZeroDivisionError):
            return None
    if re.fullmatch(r"[\-+]?\d+\.\d+", cleaned):
        try:
            return Fraction(cleaned)
        except ValueError:
            try:
                val = float(cleaned)
                return val if math.isfinite(val) else None
            except ValueError:
                return None
    return None


def answers_equivalent(pred_raw: str | None, truth_raw: str) -> bool:
    """Deterministic mathematical equivalence check between extracted prediction and ground truth."""
    if pred_raw is None or not pred_raw.strip():
        return False
    norm_pred = normalize_math_answer(pred_raw)
    norm_truth = normalize_math_answer(truth_raw)
    if not norm_pred or not norm_truth:
        return False
    if norm_pred == norm_truth:
        return True

    val_p = _try_parse_rational_or_float(norm_pred)
    val_t = _try_parse_rational_or_float(norm_truth)
    if val_p is not None and val_t is not None:
        if isinstance(val_p, Fraction) and isinstance(val_t, Fraction):
            return val_p == val_t
        return math.isclose(float(val_p), float(val_t), rel_tol=1e-7, abs_tol=1e-9)
    return False


def extract_gsm8k_ground_truth(answer_field: str) -> str:
    """Extract canonical numeric ground truth after `####` in GSM8K dataset solutions."""
    matches = _GSM8K_STRICT_RE.findall(answer_field)
    if not matches:
        raise ValueError(f"Could not locate '#### <number>' in GSM8K solution: {answer_field!r}")
    return matches[-1].replace(",", "").strip()


def score_gsm8k(
    solution_str: str,
    ground_truth: str,
    method: str = "strict",
    format_score: float = 0.0,
    correct_score: float = 1.0,
) -> float:
    """Score a GSM8K completion deterministically (matches `verl.utils.reward_score.gsm8k`)."""
    if not solution_str or not solution_str.strip():
        return 0.0
    tail = solution_str[-_SOLUTION_CLIP_CHARS:]
    extracted: str | None = None

    if method == "strict":
        matches = _GSM8K_STRICT_RE.findall(tail)
        if matches:
            extracted = matches[-1]
        else:
            boxed = extract_last_boxed(tail)
            if boxed is not None:
                extracted = boxed
    else:
        matches = _GSM8K_STRICT_RE.findall(tail)
        if matches:
            extracted = matches[-1]
        else:
            boxed = extract_last_boxed(tail)
            if boxed is not None:
                extracted = boxed
            else:
                nums = _LAST_NUMBER_RE.findall(tail)
                if nums:
                    extracted = nums[-1]

    if extracted is None:
        return 0.0
    return correct_score if answers_equivalent(extracted, ground_truth) else format_score


def score_math_boxed(
    solution_str: str,
    ground_truth: str,
    format_score: float = 0.0,
    correct_score: float = 1.0,
) -> float:
    """Score a MATH / MATH-500 completion using `\\boxed{...}` or Minerva `Final Answer:` extraction."""
    if not solution_str or not solution_str.strip():
        return 0.0
    tail = solution_str[-_SOLUTION_CLIP_CHARS:]
    extracted = extract_last_boxed(tail)
    if extracted is None:
        m = _MINERVA_ANSWER_RE.findall(tail)
        if m:
            extracted = m[-1].strip()
        else:
            m_hash = _GSM8K_STRICT_RE.findall(tail)
            if m_hash:
                extracted = m_hash[-1].strip()

    if extracted is None:
        return 0.0
    return correct_score if answers_equivalent(extracted, ground_truth) else format_score


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict[str, Any] | None = None,
) -> float:
    """Unified VeRL `custom_reward_function` entrypoint (`verl.utils.reward_score` signature)."""
    del extra_info
    src = (data_source or "").lower()
    if "gsm8k" in src:
        return float(score_gsm8k(solution_str=solution_str, ground_truth=str(ground_truth), method="strict"))
    return float(score_math_boxed(solution_str=solution_str, ground_truth=str(ground_truth)))


def format_verl_grpo_row(
    *,
    data_source: str,
    question: str,
    ground_truth: str,
    split: str,
    index: int,
    include_shared_system_prefix: bool = True,
    raw_answer: str = "",
) -> dict[str, Any]:
    """Construct a single row conforming to VeRL's RLHFDataset Parquet schema."""
    if "gsm8k" in data_source.lower():
        instruction_suffix = 'Let\'s think step by step and output the final answer after "####".'
    else:
        instruction_suffix = "Let's think step by step and output the final answer within \\boxed{}."

    user_content = f"{question.strip()}\n\n{instruction_suffix}"
    messages: list[dict[str, str]] = []
    if include_shared_system_prefix:
        messages.append({"role": "system", "content": SHARED_REASONING_SYSTEM_PROMPT})
    messages.append({"role": "user", "content": user_content})

    prefix_hash = hashlib.sha256(
        (SHARED_REASONING_SYSTEM_PROMPT if include_shared_system_prefix else "").encode("utf-8")
    ).hexdigest()[:16]

    return {
        "data_source": data_source,
        "prompt": messages,
        "ability": "math",
        "reward_model": {
            "style": "rule",
            "ground_truth": str(ground_truth),
        },
        "extra_info": {
            "split": split,
            "index": int(index),
            "question": question.strip(),
            "answer": raw_answer,
            "shared_prefix_hash": prefix_hash,
            "shared_prefix_chars": len(SHARED_REASONING_SYSTEM_PROMPT) if include_shared_system_prefix else 0,
        },
    }


def write_verl_parquet(rows: list[dict[str, Any]], output_path: str | Path) -> str:
    """Write rows to Parquet and return the SHA-256 hex digest of the file."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_parquet(out, index=False)
    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    return digest


def verify_manifest(manifest_path: str | Path) -> bool:
    """Verify SHA-256 checksums of prepared train/val Parquet files."""
    mpath = Path(manifest_path)
    if not mpath.exists():
        return False
    manifest = json.loads(mpath.read_text(encoding="utf-8"))
    for entry in manifest.get("files", []):
        fpath = Path(entry["path"])
        if not fpath.exists():
            return False
        actual = hashlib.sha256(fpath.read_bytes()).hexdigest()
        if actual != entry["sha256"]:
            return False
    return True

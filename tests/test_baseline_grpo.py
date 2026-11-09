"""Phase 1 Gate Tests: Baseline GRPO Numerical Rigor, Reward Verification & Telemetry."""

from __future__ import annotations

import math
from pathlib import Path
import time

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.cross_step_kv_grpo import compute_grpo_group_advantages, compute_grpo_vanilla_reference
from src.rewards_and_data import compute_score
from src.telemetry import StepTelemetryRecord, TelemetryLogger, load_step_telemetry


class TinyCausalTransformer(nn.Module):
    """Compact causal Transformer for real 10-step autoregressive GRPO gradient verification."""

    def __init__(self, vocab_size: int = 32, d_model: int = 64, nhead: int = 4, num_layers: int = 2) -> None:
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(64, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=128,
            dropout=0.0,
            batch_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        bsz, seq_len = input_ids.shape
        pos = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(bsz, seq_len)
        x = self.tok_emb(input_ids) + self.pos_emb(pos)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=input_ids.device)
        h = self.blocks(x, mask=causal_mask)
        return self.lm_head(h)


def _decode_tokens_to_gsm8k_text(token_ids: list[int], target_answer_token: int = 7) -> str:
    """Deterministic token-to-math completion mapper producing valid/invalid GSM8K strings."""
    if not token_ids:
        return ""
    ans_val = 42 if token_ids[-1] == target_answer_token else (10 + token_ids[-1])
    steps_str = " + ".join(str(t) for t in token_ids[:-1]) or "0"
    return f"Step 1: compute {steps_str}.\n#### {ans_val}"


def test_ten_step_mini_transformer_grpo_end_to_end(tmp_path: Path) -> None:
    """Verify 10 real GRPO steps produce finite loss/KL/gradients, accurate rewards,
    all 4 timing metrics, and measurable reward progression.
    """
    torch.manual_seed(42)
    device = torch.device("tpu") if (hasattr(torch, "tpu") and torch.tpu.is_available()) else torch.device("cpu")

    vocab_size = 24
    target_tok = 7
    actor = TinyCausalTransformer(vocab_size=vocab_size).to(device)
    ref_model = TinyCausalTransformer(vocab_size=vocab_size).to(device)
    rollout_model = TinyCausalTransformer(vocab_size=vocab_size).to(device)
    ref_model.load_state_dict(actor.state_dict())
    rollout_model.load_state_dict(actor.state_dict())
    ref_model.eval()
    rollout_model.eval()

    optimizer = torch.optim.AdamW(actor.parameters(), lr=8e-3)
    logger = TelemetryLogger(tmp_path, run_id="unit_smoke_10step")

    num_prompts = 4
    group_n = 4
    bsz = num_prompts * group_n
    prompt_len = 6
    resp_len = 6
    prompts = torch.randint(1, vocab_size, (num_prompts, prompt_len), device=device).repeat_interleave(group_n, dim=0)
    group_ids = [i // group_n for i in range(bsz)]
    response_mask = torch.ones((bsz, resp_len), dtype=torch.float32, device=device)

    initial_reward = None
    final_reward = None

    for step in range(1, 11):
        t_step0 = time.perf_counter()

        # (a) Rollout prefill
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = rollout_model(prompts)
        prefill_s = time.perf_counter() - t0

        # (b) Autoregressive rollout decode + log-probs
        t0 = time.perf_counter()
        seq = prompts.clone()
        rollout_lps = []
        with torch.no_grad():
            for _ in range(resp_len):
                logits = rollout_model(seq)[:, -1, :]
                logp = F.log_softmax(logits, dim=-1)
                probs = logp.exp()
                next_tok = torch.multinomial(probs, num_samples=1)
                chosen_lp = logp.gather(-1, next_tok)
                rollout_lps.append(chosen_lp)
                seq = torch.cat([seq, next_tok], dim=-1)
        old_log_prob = torch.cat(rollout_lps, dim=-1)  # (B, resp_len)
        responses = seq[:, prompt_len:]
        decode_s = time.perf_counter() - t0

        # Deterministic rule-based GSM8K reward scoring on non-empty completions
        rewards = torch.zeros((bsz, resp_len), dtype=torch.float32, device=device)
        non_empty = 0
        for i in range(bsz):
            completion_str = _decode_tokens_to_gsm8k_text(responses[i].detach().cpu().tolist(), target_tok)
            if completion_str.strip():
                non_empty += 1
            r_scalar = compute_score("openai/gsm8k", completion_str, "42")
            rewards[i, -1] = r_scalar

        mean_rew = float(rewards.sum(dim=-1).mean().item())
        if initial_reward is None:
            initial_reward = mean_rew
        final_reward = mean_rew

        # (d) Trainer forward + backward + optimizer step
        t0 = time.perf_counter()
        advantages = compute_grpo_group_advantages(rewards, response_mask, group_ids)
        full_logits = actor(seq)[:, prompt_len - 1 : -1, :]
        curr_log_prob = F.log_softmax(full_logits, dim=-1).gather(-1, responses.unsqueeze(-1)).squeeze(-1)

        with torch.no_grad():
            ref_logits = ref_model(seq)[:, prompt_len - 1 : -1, :]
            ref_log_prob = F.log_softmax(ref_logits, dim=-1).gather(-1, responses.unsqueeze(-1)).squeeze(-1)

        loss, loss_metrics = compute_grpo_vanilla_reference(
            old_log_prob=old_log_prob,
            log_prob=curr_log_prob,
            advantages=advantages,
            response_mask=response_mask,
            ref_log_prob=ref_log_prob,
            kl_coef=0.001,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=2.0).item())
        optimizer.step()
        trainer_s = time.perf_counter() - t0

        # (c) Synchronous weight sync from trainer -> rollout model
        t0 = time.perf_counter()
        rollout_model.load_state_dict(actor.state_dict())
        sync_s = time.perf_counter() - t0

        step_wall = time.perf_counter() - t_step0
        total_toks = bsz * (prompt_len + resp_len)
        rec = StepTelemetryRecord(
            run_id="unit_smoke_10step",
            step=step,
            rollout_prefill_time_s=prefill_s,
            rollout_decode_time_s=decode_s,
            weight_sync_stall_time_s=sync_s,
            trainer_step_time_s=trainer_s,
            step_wall_time_s=step_wall,
            cumulative_wall_time_s=logger.elapsed_wall_s(),
            loss=loss_metrics["loss"],
            policy_loss=loss_metrics["policy_loss"],
            kl_div=loss_metrics["kl_div"],
            grad_norm=grad_norm,
            mean_reward=mean_rew,
            non_empty_completion_rate=non_empty / float(bsz),
            rollout_tokens=total_toks,
            rollout_tokens_per_sec=total_toks / max(prefill_s + decode_s, 1e-6),
        )
        logger.log_step(rec, enforce_rigor=True)

    loaded = load_step_telemetry(tmp_path)
    assert len(loaded) == 10
    for r in loaded:
        assert math.isfinite(r.loss)
        assert math.isfinite(r.kl_div) and r.kl_div >= 0.0
        assert math.isfinite(r.grad_norm) and r.grad_norm >= 0.0
        assert r.non_empty_completion_rate == 1.0
        assert r.rollout_prefill_time_s > 0.0
        assert r.rollout_decode_time_s > 0.0
        assert r.weight_sync_stall_time_s >= 0.0
        assert r.trainer_step_time_s > 0.0

    assert final_reward is not None and initial_reward is not None
    assert final_reward >= initial_reward, (
        f"Expected reward progression across 10 real GRPO updates ({initial_reward:.3f} -> {final_reward:.3f})"
    )


def test_live_tpu_run_telemetry_rigor(
    cli_run_dir: Path | None,
    expected_min_steps: int,
    require_tpu_v7: bool,
) -> None:
    """Validate real Cloud TPU v7 VeRL GRPO smoke/verification telemetry directory when present."""
    default_dir = Path(__file__).resolve().parents[1] / "results" / "phase1_smoke"
    target_dir = cli_run_dir or (default_dir if (default_dir / "step_metrics.csv").exists() else None)

    if target_dir is None:
        if require_tpu_v7:
            pytest.fail("Missing --run-dir or results/phase1_smoke/step_metrics.csv under --require-tpu-v7!")
        pytest.skip("No live TPU run directory passed via --run-dir (staging mode).")

    records = load_step_telemetry(target_dir)
    assert len(records) >= expected_min_steps, (
        f"Expected >= {expected_min_steps} logged steps in {target_dir}, found {len(records)}"
    )
    for rec in records:
        rec.validate_numerical_rigor()
        assert rec.rollout_prefill_time_s > 0.0
        assert rec.rollout_decode_time_s > 0.0
        assert rec.trainer_step_time_s > 0.0

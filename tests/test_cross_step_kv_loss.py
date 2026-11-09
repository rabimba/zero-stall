"""Phase 2 Gate (Optimization B): Bounded-Staleness Cross-Step KV Prefix Reuse & TIS GRPO Loss."""

from __future__ import annotations

import math
import pytest
import torch

from src.cross_step_kv_grpo import (
    BoundedStalenessKVController,
    compute_cross_step_kv_grpo_loss,
    compute_grpo_vanilla_reference,
    estimate_empirical_kl_shift,
)


def test_exact_reduction_to_standard_grpo_at_zero_staleness() -> None:
    """(a) Verify exact value and gradient reduction to standard VeRL GRPO when staleness = 0."""
    torch.manual_seed(7)
    bsz, seq_len = 8, 32
    old_lp = -torch.rand((bsz, seq_len), dtype=torch.float32) * 2.5
    ref_lp = old_lp - 0.05 * torch.randn((bsz, seq_len), dtype=torch.float32)
    advantages = torch.randn((bsz, seq_len), dtype=torch.float32)
    mask = (torch.arange(seq_len).unsqueeze(0) < torch.randint(12, seq_len + 1, (bsz, 1))).float()

    curr_lp_ref = (old_lp + 0.08 * torch.randn((bsz, seq_len))).detach().requires_grad_(True)
    curr_lp_kv = curr_lp_ref.detach().clone().requires_grad_(True)
    synthetic_stale_lp = old_lp - 0.4 * torch.randn((bsz, seq_len))

    loss_ref, m_ref = compute_grpo_vanilla_reference(
        old_log_prob=old_lp,
        log_prob=curr_lp_ref,
        advantages=advantages,
        response_mask=mask,
        ref_log_prob=ref_lp,
        kl_coef=0.001,
    )
    loss_ref.backward()

    loss_kv0, m_kv0 = compute_cross_step_kv_grpo_loss(
        old_log_prob=old_lp,
        log_prob=curr_lp_kv,
        advantages=advantages,
        response_mask=mask,
        staleness_steps=0,
        stale_behavior_log_prob=synthetic_stale_lp,
        ref_log_prob=ref_lp,
        kl_coef=0.001,
    )
    loss_kv0.backward()

    assert torch.allclose(loss_ref, loss_kv0, atol=1e-7, rtol=1e-6)
    assert curr_lp_ref.grad is not None and curr_lp_kv.grad is not None
    assert torch.allclose(curr_lp_ref.grad, curr_lp_kv.grad, atol=1e-7, rtol=1e-6)
    assert m_kv0["tis_mean_weight"] == 1.0
    assert m_kv0["tis_ess_ratio"] == 1.0
    assert math.isclose(m_ref["kl_div"], m_kv0["kl_div"], rel_tol=1e-6, abs_tol=1e-7)


@pytest.mark.parametrize("kl_shift_scale", [0.002, 0.02, 0.2, 2.0, 15.0, 50.0])
def test_finite_bounded_gradients_under_synthetic_kl_shifts(kl_shift_scale: float) -> None:
    """(b) Verify finite, strictly bounded gradients under mild through extreme synthetic KL shifts."""
    torch.manual_seed(42)
    bsz, seq_len = 6, 24
    old_lp = -torch.rand((bsz, seq_len), dtype=torch.float32) * 2.0
    stale_lp = old_lp - kl_shift_scale * torch.randn((bsz, seq_len), dtype=torch.float32)
    ref_lp = old_lp - 0.1 * torch.randn((bsz, seq_len), dtype=torch.float32)
    advantages = torch.randn((bsz, seq_len), dtype=torch.float32) * 1.5
    mask = torch.ones((bsz, seq_len), dtype=torch.float32)
    prefix_mask = (torch.arange(seq_len).unsqueeze(0) < 12).float().expand(bsz, seq_len)

    curr_lp = (old_lp + kl_shift_scale * torch.randn((bsz, seq_len))).detach().requires_grad_(True)

    loss, metrics = compute_cross_step_kv_grpo_loss(
        old_log_prob=old_lp,
        log_prob=curr_lp,
        advantages=advantages,
        response_mask=mask,
        staleness_steps=2,
        stale_behavior_log_prob=stale_lp,
        prefix_reuse_mask=prefix_mask,
        ref_log_prob=ref_lp,
        kl_coef=0.001,
        tis_clip_min=0.2,
        tis_clip_max=3.0,
    )
    loss.backward()

    assert torch.isfinite(loss).item(), f"Non-finite loss at kl_shift_scale={kl_shift_scale}"
    assert curr_lp.grad is not None
    assert torch.isfinite(curr_lp.grad).all().item(), f"Non-finite gradient at kl_shift_scale={kl_shift_scale}"

    # Analytical per-token gradient bound:
    # Dual-clip PPO ratio has zero gradient outside clip range; max |d(pg)/d(lp)| <= tis_max * clip_c * max|A|
    max_adv = float(advantages.abs().max().item())
    analytical_token_bound = (3.0 * 3.0 * max_adv + 0.001 * 36.0) / 1.0  # before 1/(B*L) mean
    actual_max_grad = float(curr_lp.grad.abs().max().item())
    assert actual_max_grad <= analytical_token_bound + 1e-4
    assert 0.0 < metrics["tis_ess_ratio"] <= 1.0


def test_automatic_kv_cache_invalidation_when_kl_exceeds_delta() -> None:
    """(c) Verify prefix retention when KL <= delta and immediate invalidation when KL > delta."""
    reset_calls: list[int] = []
    released_batches: list[list[bytes]] = []

    class MockTPUSyncKVStore:
        def __init__(self) -> None:
            self.saved: dict[bytes, int] = {}

        def save(self, hashes: list[bytes], block_ids: list[int]) -> None:
            for h, bid in zip(hashes, block_ids):
                self.saved[h] = bid

        def release(self, hashes: list[bytes]) -> None:
            released_batches.append(list(hashes))
            for h in hashes:
                self.saved.pop(h, None)

    kv_store = MockTPUSyncKVStore()
    ctrl = BoundedStalenessKVController(
        delta_kl_threshold=0.02,
        max_staleness_steps=3,
        block_size_tokens=8,
        reset_prefix_cache_fn=lambda: reset_calls.append(1),
        tpu_sync_kv_store=kv_store,
    )

    prompt_tokens = list(range(100, 128))  # 28 tokens -> 3 full blocks of 8 tokens (24 cached tokens)
    registered = ctrl.register_prefix_tokens(
        prompt_tokens,
        policy_version=0,
        prefix_kind="system_exemplar",
        device_block_ids=[10, 11, 12],
    )
    assert len(registered) == 3
    assert len(kv_store.saved) == 3
    cached_len, _ = ctrl.lookup_cached_prefix_tokens(prompt_tokens + [999, 1000])
    assert cached_len == 24

    # Step 1: KL = 0.006 <= 0.02 -> retained (staleness = 1, cum_kl = 0.006)
    d1 = ctrl.step_policy_update(0.006, policy_version=1)
    assert d1.retained and not d1.invalidated
    assert d1.staleness_steps == 1
    assert ctrl.num_retained_blocks == 3
    assert len(reset_calls) == 0

    # Step 2: KL = 0.008 -> cum_kl = 0.014 <= 0.02 -> retained (staleness = 2)
    d2 = ctrl.step_policy_update(0.008, policy_version=2)
    assert d2.retained and not d2.invalidated
    assert d2.staleness_steps == 2
    assert len(reset_calls) == 0

    # Step 3: Single-step KL spike = 0.035 > delta (0.02) -> MUST auto-invalidate!
    d3 = ctrl.step_policy_update(0.035, policy_version=3)
    assert d3.invalidated and not d3.retained
    assert d3.staleness_steps == 0
    assert ctrl.current_staleness_steps == 0
    assert ctrl.cumulative_kl_drift == 0.0
    assert ctrl.num_retained_blocks == 0
    assert len(reset_calls) == 1
    assert len(released_batches) == 1 and len(released_batches[0]) == 3
    assert len(kv_store.saved) == 0

    # Re-populate and test max_staleness_steps cap even under tiny KL
    ctrl.register_prefix_tokens(prompt_tokens, policy_version=3, device_block_ids=[20, 21, 22])
    for v in (4, 5, 6):
        dv = ctrl.step_policy_update(0.001, policy_version=v)
        assert dv.retained
    # 4th step exceeds max_staleness_steps=3 -> invalidates!
    d7 = ctrl.step_policy_update(0.001, policy_version=7)
    assert d7.invalidated and d7.staleness_steps == 0
    assert len(reset_calls) == 2


def test_empirical_kl_estimator_monotonicity() -> None:
    """Verify $k_3$ empirical KL shift is 0 on identical distributions and grows monotonically with drift."""
    torch.manual_seed(11)
    mask = torch.ones((4, 16), dtype=torch.float32)
    base_lp = -torch.rand((4, 16), dtype=torch.float32)

    kl_zero = estimate_empirical_kl_shift(base_lp, base_lp, mask)
    kl_small = estimate_empirical_kl_shift(base_lp, base_lp - 0.05, mask)
    kl_large = estimate_empirical_kl_shift(base_lp, base_lp - 0.30, mask)

    assert math.isclose(kl_zero, 0.0, abs_tol=1e-7)
    assert 0.0 < kl_small < kl_large

"""Optimization B (Research Contribution): Bounded-Staleness Cross-Step KV Prefix Reuse for GRPO.

Core components:
1. `compute_grpo_group_advantages`: Group-relative outcome advantage estimator using explicit
   eager variance calculation (avoiding `aten::std.correction` kernel gaps on TorchTPU).
2. `estimate_empirical_kl_shift`: Non-negative low-variance $k_3$ policy shift estimator
   $\\widehat{D}_{\\mathrm{KL}}(\\pi_{\\theta_t} \\,\\|\\, \\pi_{\\theta_{t-1}})$.
3. `BoundedStalenessKVController`: KL-gated prefix KV cache retention controller bridging
   `vllm-torchtpu` (`PrefixAwareScheduler` / `reset_prefix_cache`) and `google/tpu-sync`
   (`tpu_sync.api.torch.kv_cache_store.KVCacheStore` & `KVCacheManager` HBM/DRAM tiers).
   Retains shared system rubrics, few-shot exemplars, curriculum/replay prompts, and
   low-divergence reasoning prefix blocks across $t-1 \\to t$ when $\\widehat{D}_{\\mathrm{KL}} \\le \\delta$
   and $\\tau < \\tau_{\\max}$, automatically invalidating on threshold violation.
4. `compute_cross_step_kv_grpo_loss`: Truncated Importance-Sampling (TIS) + dual-clip GRPO
   surrogate objective that reduces *identically* to standard VeRL GRPO when `staleness_steps == 0`
   while maintaining unbiased first-order improvement and strictly bounded gradients under stale KV prefixes.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
import hashlib
import math
from typing import Any

import torch


LOG_RATIO_SAFETY_CLAMP: float = 20.0
KL_MAX_CLAMP: float = 10.0


def masked_mean(values: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Compute token-level masked mean over `mask` (`1` = valid response token, `0` = pad)."""
    mask_f = mask.to(values.dtype)
    denom = mask_f.sum().clamp(min=eps)
    return (values * mask_f).sum() / denom


def compute_low_var_kl_matrix(
    log_prob: torch.Tensor,
    ref_log_prob: torch.Tensor,
) -> torch.Tensor:
    """Compute Schulman's non-negative low-variance $k_3$ KL estimator per token.

    Matches `verl.trainer.ppo.core_algos.kl_penalty(..., kl_penalty="low_var_kl")`:
      $k_3 = \\operatorname{clamp}(\\exp(u) - u - 1, -10, 10)$ where
      $u = \\operatorname{clamp}(\\log \\pi_{\\text{ref}} - \\log \\pi_\\theta, -20, 20)$.
    Note: Because `kld` is clamped at `max=10.0` (where $e^u - u - 1 = 10 \\implies u \\approx 3.577$),
    $|\\partial k_3 / \\partial \\log \\pi_\\theta| \\le e^{3.577} - 1 \\approx 34.8$ everywhere!
    """
    u = torch.clamp(ref_log_prob - log_prob, min=-LOG_RATIO_SAFETY_CLAMP, max=LOG_RATIO_SAFETY_CLAMP)
    ratio = torch.exp(u)
    kld = (ratio - u - 1.0).contiguous()
    return torch.clamp(kld, min=-KL_MAX_CLAMP, max=KL_MAX_CLAMP)


@torch.no_grad()
def estimate_empirical_kl_shift(
    prev_log_prob: torch.Tensor,
    curr_log_prob: torch.Tensor,
    response_mask: torch.Tensor,
) -> float:
    """Estimate empirical per-batch policy KL shift $\\widehat{D}_{\\mathrm{KL}}(\\pi_t \\,\\|\\, \\pi_{t-1})$."""
    kld = compute_low_var_kl_matrix(log_prob=curr_log_prob.float(), ref_log_prob=prev_log_prob.float())
    val = float(masked_mean(kld, response_mask.float()).item())
    return max(0.0, val) if math.isfinite(val) else float("inf")


def compute_grpo_group_advantages(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    group_ids: Sequence[int | str] | torch.Tensor,
    *,
    eps: float = 1e-6,
    norm_by_std: bool = True,
) -> torch.Tensor:
    """Compute group-relative outcome advantages matching `verl` GRPO without `aten::std.correction`.

    Args:
        token_level_rewards: Tensor of shape `(B, L)`.
        response_mask: Binary mask of shape `(B, L)`.
        group_ids: Prompt group identifier per trajectory of length `B`.
        eps: Numerical stability epsilon.
        norm_by_std: Divide centered group rewards by Bessel-corrected group std (`True` = GRPO).

    Returns:
        Token-level advantages tensor of shape `(B, L)`.
    """
    scores = (token_level_rewards * response_mask.to(token_level_rewards.dtype)).sum(dim=-1)  # (B,)
    bsz = scores.shape[0]
    if isinstance(group_ids, torch.Tensor):
        g_list: list[Any] = [int(x) for x in group_ids.detach().cpu().tolist()]
    else:
        g_list = list(group_ids)
    if len(g_list) != bsz:
        raise ValueError(f"group_ids length ({len(g_list)}) != batch size ({bsz})")

    buckets: dict[Any, list[int]] = {}
    for idx, gid in enumerate(g_list):
        buckets.setdefault(gid, []).append(idx)

    adv_1d = torch.zeros_like(scores)
    with torch.no_grad():
        for indices in buckets.values():
            if len(indices) <= 1:
                continue
            idx_t = torch.tensor(indices, device=scores.device, dtype=torch.long)
            grp_scores = scores.index_select(0, idx_t)
            mean = grp_scores.mean()
            centered = grp_scores - mean
            if norm_by_std:
                # Explicit Bessel-corrected variance avoids missing `aten::std.correction` on TorchTPU
                var = (centered * centered).sum() / float(len(indices) - 1)
                std = torch.sqrt(var.clamp(min=0.0))
                normed = centered / (std + eps)
            else:
                normed = centered
            adv_1d.index_copy_(0, idx_t, normed)

    return adv_1d.unsqueeze(-1) * response_mask.to(scores.dtype)


def compute_grpo_vanilla_reference(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    *,
    ref_log_prob: torch.Tensor | None = None,
    kl_coef: float = 0.001,
    clip_ratio_low: float = 0.2,
    clip_ratio_high: float = 0.2,
    clip_ratio_c: float = 3.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Reference standard VeRL dual-clip GRPO objective (`staleness == 0` baseline)."""
    neg_approx_kl = torch.clamp(
        log_prob - old_log_prob,
        min=-LOG_RATIO_SAFETY_CLAMP,
        max=LOG_RATIO_SAFETY_CLAMP,
    )
    ratio = torch.exp(neg_approx_kl)

    pg_losses1 = -advantages * ratio
    pg_losses2 = -advantages * torch.clamp(ratio, 1.0 - clip_ratio_low, 1.0 + clip_ratio_high)
    clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)

    pg_losses3 = -advantages * clip_ratio_c
    clip_pg_losses2 = torch.minimum(pg_losses3, clip_pg_losses1)
    pg_losses = torch.where(advantages < 0.0, clip_pg_losses2, clip_pg_losses1)

    policy_loss = masked_mean(pg_losses, response_mask)
    total_loss = policy_loss
    kl_val = 0.0

    if ref_log_prob is not None and kl_coef > 0.0:
        kld = compute_low_var_kl_matrix(log_prob=log_prob, ref_log_prob=ref_log_prob)
        kl_loss = masked_mean(kld, response_mask)
        total_loss = total_loss + kl_coef * kl_loss
        kl_val = float(kl_loss.detach().item())

    with torch.no_grad():
        clip_frac = float(
            masked_mean(
                ((ratio < 1.0 - clip_ratio_low) | (ratio > 1.0 + clip_ratio_high)).to(log_prob.dtype),
                response_mask,
            ).item()
        )
        approx_kl = float(masked_mean(-neg_approx_kl, response_mask).item())

    metrics = {
        "loss": float(total_loss.detach().item()),
        "policy_loss": float(policy_loss.detach().item()),
        "kl_div": kl_val,
        "approx_kl": approx_kl,
        "clip_frac": clip_frac,
        "tis_mean_weight": 1.0,
        "tis_ess_ratio": 1.0,
    }
    return total_loss, metrics


def compute_cross_step_kv_grpo_loss(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    *,
    staleness_steps: int = 0,
    stale_behavior_log_prob: torch.Tensor | None = None,
    prefix_reuse_mask: torch.Tensor | None = None,
    ref_log_prob: torch.Tensor | None = None,
    kl_coef: float = 0.001,
    clip_ratio_low: float = 0.2,
    clip_ratio_high: float = 0.2,
    clip_ratio_c: float = 3.0,
    tis_clip_min: float = 0.2,
    tis_clip_max: float = 3.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Bounded-Staleness Cross-Step KV Prefix Reuse GRPO Surrogate Loss with Truncated IS.

    Properties:
      (a) Exact Reduction: When `staleness_steps == 0` (or `stale_behavior_log_prob is None`),
          delegates along the exact algebraic path of standard dual-clip GRPO (`tis_weights == 1`),
          yielding bit-identical loss and parameter gradients.
      (b) Bounded Gradients: Both the dual-clip PPO ratio and the $k_3$ KL regularizer clamp
          extreme log-ratios, while detached TIS weights $w^{\\text{TIS}} \\in [c_{\\min}, c_{\\max}]$
          strictly bound gradient norm under arbitrarily large synthetic KL shifts.
    """
    if staleness_steps < 0:
        raise ValueError(f"staleness_steps must be >= 0, got {staleness_steps}")
    if tis_clip_min <= 0.0 or tis_clip_max < tis_clip_min:
        raise ValueError(f"Invalid TIS bounds [{tis_clip_min}, {tis_clip_max}]")

    # Fast exact reduction when staleness == 0
    if staleness_steps == 0 or stale_behavior_log_prob is None:
        return compute_grpo_vanilla_reference(
            old_log_prob=old_log_prob,
            log_prob=log_prob,
            advantages=advantages,
            response_mask=response_mask,
            ref_log_prob=ref_log_prob,
            kl_coef=kl_coef,
            clip_ratio_low=clip_ratio_low,
            clip_ratio_high=clip_ratio_high,
            clip_ratio_c=clip_ratio_c,
        )

    neg_approx_kl = torch.clamp(
        log_prob - old_log_prob,
        min=-LOG_RATIO_SAFETY_CLAMP,
        max=LOG_RATIO_SAFETY_CLAMP,
    )
    ratio = torch.exp(neg_approx_kl)

    pg_losses1 = -advantages * ratio
    pg_losses2 = -advantages * torch.clamp(ratio, 1.0 - clip_ratio_low, 1.0 + clip_ratio_high)
    clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)

    pg_losses3 = -advantages * clip_ratio_c
    clip_pg_losses2 = torch.minimum(pg_losses3, clip_pg_losses1)
    pg_losses = torch.where(advantages < 0.0, clip_pg_losses2, clip_pg_losses1)

    # Truncated Importance-Sampling (TIS) correction for stale prefix KV conditioning:
    # w_i = clip(exp(log pi_rollout(y_i | K_t) - log mu_kv(y_i | K_{t-tau})), c_min, c_max)
    with torch.no_grad():
        kv_log_delta = torch.clamp(
            old_log_prob.detach() - stale_behavior_log_prob.detach(),
            min=-LOG_RATIO_SAFETY_CLAMP,
            max=LOG_RATIO_SAFETY_CLAMP,
        )
        if prefix_reuse_mask is not None:
            kv_log_delta = kv_log_delta * prefix_reuse_mask.to(kv_log_delta.dtype)
        raw_is = torch.exp(kv_log_delta)
        tis_weights = torch.clamp(raw_is, min=tis_clip_min, max=tis_clip_max).to(pg_losses.dtype)

        # Effective Sample Size (ESS) diagnostic over valid response tokens:
        # ESS / N = (mean w)^2 / mean(w^2)
        w_mean = float(masked_mean(tis_weights, response_mask).item())
        w_sq_mean = float(masked_mean(tis_weights * tis_weights, response_mask).item())
        ess_ratio = (w_mean * w_mean) / max(w_sq_mean, 1e-12)
        ess_ratio = min(max(ess_ratio, 0.0), 1.0)

    weighted_pg_losses = pg_losses * tis_weights.detach()
    policy_loss = masked_mean(weighted_pg_losses, response_mask)
    total_loss = policy_loss
    kl_val = 0.0

    if ref_log_prob is not None and kl_coef > 0.0:
        kld = compute_low_var_kl_matrix(log_prob=log_prob, ref_log_prob=ref_log_prob)
        kl_loss = masked_mean(kld, response_mask)
        total_loss = total_loss + kl_coef * kl_loss
        kl_val = float(kl_loss.detach().item())

    with torch.no_grad():
        clip_frac = float(
            masked_mean(
                ((ratio < 1.0 - clip_ratio_low) | (ratio > 1.0 + clip_ratio_high)).to(log_prob.dtype),
                response_mask,
            ).item()
        )
        approx_kl = float(masked_mean(-neg_approx_kl, response_mask).item())

    metrics = {
        "loss": float(total_loss.detach().item()),
        "policy_loss": float(policy_loss.detach().item()),
        "kl_div": kl_val,
        "approx_kl": approx_kl,
        "clip_frac": clip_frac,
        "tis_mean_weight": w_mean,
        "tis_ess_ratio": ess_ratio,
    }
    return total_loss, metrics


@dataclass
class PrefixBlockEntry:
    """Metadata for a retained prefix KV cache block in vLLM-TorchTPU / TPU Sync memory tiers."""

    block_hash: bytes
    policy_version: int
    token_count: int
    prefix_kind: str  # "system_exemplar" | "problem_prompt" | "reasoning_prefix"
    tier: str = "HBM_AND_HOST"  # mirrors tpu_sync KVCacheStore BlockStatus.HOST_AND_HBM


@dataclass
class KVRetentionDecision:
    """Outcome of a cross-step KV prefix retention gate evaluation at policy update `t`."""

    policy_version: int
    step_kl_shift: float
    cumulative_kl_drift: float
    retained: bool
    invalidated: bool
    staleness_steps: int
    retained_blocks: int
    reason: str


class BoundedStalenessKVController:
    """KL-gated cross-step KV prefix retention controller for `vllm-torchtpu` + `tpu-sync`.

    Instead of unconditionally calling `engine.reset_prefix_cache()` and
    `release_kv_cache_replicas()` at every GRPO weight sync, this controller retains
    content-addressed prefix KV blocks across step `t-1 -> t` whenever:
      1. Single-step empirical KL shift `step_kl_shift <= delta_kl_threshold`,
      2. Cumulative cross-step drift `cumulative_kl_drift + step_kl_shift <= delta_kl_threshold`,
      3. Prefix age `current_staleness_steps + 1 <= max_staleness_steps`.

    Upon any violation, all cached prefix blocks in both vLLM HBM (`reset_prefix_cache_fn`)
    and `tpu_sync` `KVCacheStore` (`kv_store.release`) are invalidated atomically and
    staleness resets to `0`.
    """

    def __init__(
        self,
        *,
        delta_kl_threshold: float = 0.02,
        max_staleness_steps: int = 4,
        block_size_tokens: int = 16,
        token_kl_prefix_threshold: float | None = None,
        reset_prefix_cache_fn: Callable[[], None] | None = None,
        tpu_sync_kv_store: Any | None = None,
    ) -> None:
        if delta_kl_threshold < 0.0:
            raise ValueError(f"delta_kl_threshold must be >= 0.0, got {delta_kl_threshold}")
        if max_staleness_steps < 0:
            raise ValueError(f"max_staleness_steps must be >= 0, got {max_staleness_steps}")
        if block_size_tokens <= 0:
            raise ValueError(f"block_size_tokens must be > 0, got {block_size_tokens}")

        self.delta_kl_threshold = float(delta_kl_threshold)
        self.max_staleness_steps = int(max_staleness_steps)
        self.block_size_tokens = int(block_size_tokens)
        self.token_kl_prefix_threshold = (
            float(token_kl_prefix_threshold)
            if token_kl_prefix_threshold is not None
            else float(delta_kl_threshold) * 1.5
        )
        self._reset_prefix_cache_fn = reset_prefix_cache_fn
        self._tpu_sync_kv_store = tpu_sync_kv_store

        self.current_staleness_steps: int = 0
        self.cumulative_kl_drift: float = 0.0
        self.invalidation_count: int = 0
        self.retention_count: int = 0
        self._blocks: dict[bytes, PrefixBlockEntry] = {}
        self._decisions: list[KVRetentionDecision] = []

    @property
    def num_retained_blocks(self) -> int:
        return len(self._blocks)

    @property
    def decisions(self) -> list[KVRetentionDecision]:
        return list(self._decisions)

    @staticmethod
    def compute_block_hash(token_ids: Sequence[int], parent_hash: bytes = b"") -> bytes:
        h = hashlib.sha256(parent_hash)
        h.update(bytes(f"{len(token_ids)}:", "ascii"))
        for tok in token_ids:
            h.update(int(tok).to_bytes(4, byteorder="little", signed=True))
        return h.digest()[:16]

    def register_prefix_tokens(
        self,
        token_ids: Sequence[int],
        *,
        policy_version: int,
        prefix_kind: str = "problem_prompt",
        device_block_ids: Sequence[int] | None = None,
    ) -> list[bytes]:
        """Register full `block_size_tokens`-aligned prefix blocks into HBM + TPU Sync store."""
        num_full_blocks = len(token_ids) // self.block_size_tokens
        parent = b""
        new_hashes: list[bytes] = []

        for b_idx in range(num_full_blocks):
            chunk = token_ids[b_idx * self.block_size_tokens : (b_idx + 1) * self.block_size_tokens]
            b_hash = self.compute_block_hash(chunk, parent_hash=parent)
            parent = b_hash
            if b_hash not in self._blocks:
                self._blocks[b_hash] = PrefixBlockEntry(
                    block_hash=b_hash,
                    policy_version=policy_version,
                    token_count=len(chunk),
                    prefix_kind=prefix_kind,
                )
                new_hashes.append(b_hash)

        if new_hashes and self._tpu_sync_kv_store is not None and device_block_ids is not None:
            src_ids = list(device_block_ids)[: len(new_hashes)]
            if hasattr(self._tpu_sync_kv_store, "save") and len(src_ids) == len(new_hashes):
                self._tpu_sync_kv_store.save(new_hashes, src_ids)

        return new_hashes

    def lookup_cached_prefix_tokens(self, token_ids: Sequence[int]) -> tuple[int, list[bytes]]:
        """Return `(cached_token_count, matched_block_hashes)` for a request's token sequence."""
        num_full_blocks = len(token_ids) // self.block_size_tokens
        parent = b""
        matched: list[bytes] = []

        for b_idx in range(num_full_blocks):
            chunk = token_ids[b_idx * self.block_size_tokens : (b_idx + 1) * self.block_size_tokens]
            b_hash = self.compute_block_hash(chunk, parent_hash=parent)
            parent = b_hash
            if b_hash in self._blocks:
                matched.append(b_hash)
            else:
                break

        return len(matched) * self.block_size_tokens, matched

    @torch.no_grad()
    def compute_low_divergence_prefix_mask(
        self,
        prev_log_prob: torch.Tensor,
        curr_log_prob: torch.Tensor,
        response_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Identify contiguous low-divergence reasoning prefix tokens (`(B, L)` binary mask).

        Retains prefix blocks up to the first token block whose per-block empirical $k_3$ KL
        exceeds `token_kl_prefix_threshold`, refreshing divergent reasoning suffixes cleanly.
        """
        bsz, seq_len = curr_log_prob.shape
        tok_kl = compute_low_var_kl_matrix(curr_log_prob.float(), prev_log_prob.float()) * response_mask.float()
        prefix_mask = torch.zeros_like(response_mask, dtype=torch.float32)
        blk = self.block_size_tokens

        for b in range(bsz):
            for start in range(0, seq_len, blk):
                end = min(start + blk, seq_len)
                valid_cnt = float(response_mask[b, start:end].sum().item())
                if valid_cnt <= 0:
                    break
                blk_kl = float(tok_kl[b, start:end].sum().item()) / valid_cnt
                if blk_kl <= self.token_kl_prefix_threshold:
                    prefix_mask[b, start:end] = response_mask[b, start:end].float()
                else:
                    break
        return prefix_mask

    def invalidate_all(self, reason: str = "manual") -> int:
        """Flush all retained prefix KV blocks across vLLM HBM and `tpu_sync` `KVCacheStore`."""
        flushed = len(self._blocks)
        hashes = list(self._blocks.keys())
        self._blocks.clear()

        if self._reset_prefix_cache_fn is not None:
            self._reset_prefix_cache_fn()
        if hashes and self._tpu_sync_kv_store is not None and hasattr(self._tpu_sync_kv_store, "release"):
            self._tpu_sync_kv_store.release(hashes)

        self.current_staleness_steps = 0
        self.cumulative_kl_drift = 0.0
        self.invalidation_count += 1
        del reason
        return flushed

    def step_policy_update(
        self,
        step_kl_shift: float,
        *,
        policy_version: int,
    ) -> KVRetentionDecision:
        """Evaluate KL staleness bound at policy update `t-1 -> t` and retain or invalidate KV cache."""
        if not math.isfinite(step_kl_shift):
            self.invalidate_all(reason="non_finite_kl")
            decision = KVRetentionDecision(
                policy_version=policy_version,
                step_kl_shift=step_kl_shift,
                cumulative_kl_drift=0.0,
                retained=False,
                invalidated=True,
                staleness_steps=0,
                retained_blocks=0,
                reason="invalidated_non_finite_kl",
            )
            self._decisions.append(decision)
            return decision

        cand_drift = self.cumulative_kl_drift + step_kl_shift
        cand_staleness = self.current_staleness_steps + 1

        if self.max_staleness_steps == 0 or self.delta_kl_threshold == 0.0:
            self.invalidate_all(reason="zero_staleness_policy")
            decision = KVRetentionDecision(
                policy_version=policy_version,
                step_kl_shift=step_kl_shift,
                cumulative_kl_drift=0.0,
                retained=False,
                invalidated=True,
                staleness_steps=0,
                retained_blocks=0,
                reason="invalidated_zero_staleness_config",
            )
            self._decisions.append(decision)
            return decision

        if step_kl_shift > self.delta_kl_threshold:
            self.invalidate_all(reason="step_kl_exceeded_delta")
            decision = KVRetentionDecision(
                policy_version=policy_version,
                step_kl_shift=step_kl_shift,
                cumulative_kl_drift=0.0,
                retained=False,
                invalidated=True,
                staleness_steps=0,
                retained_blocks=0,
                reason=f"invalidated_step_kl_{step_kl_shift:.5f}_gt_delta_{self.delta_kl_threshold:.5f}",
            )
            self._decisions.append(decision)
            return decision

        if cand_drift > self.delta_kl_threshold:
            self.invalidate_all(reason="cumulative_kl_exceeded_delta")
            decision = KVRetentionDecision(
                policy_version=policy_version,
                step_kl_shift=step_kl_shift,
                cumulative_kl_drift=0.0,
                retained=False,
                invalidated=True,
                staleness_steps=0,
                retained_blocks=0,
                reason=f"invalidated_cum_kl_{cand_drift:.5f}_gt_delta_{self.delta_kl_threshold:.5f}",
            )
            self._decisions.append(decision)
            return decision

        if cand_staleness > self.max_staleness_steps:
            self.invalidate_all(reason="max_staleness_reached")
            decision = KVRetentionDecision(
                policy_version=policy_version,
                step_kl_shift=step_kl_shift,
                cumulative_kl_drift=0.0,
                retained=False,
                invalidated=True,
                staleness_steps=0,
                retained_blocks=0,
                reason=f"invalidated_max_staleness_{self.max_staleness_steps}",
            )
            self._decisions.append(decision)
            return decision

        # Safe to retain prefix KV blocks across policy update t-1 -> t!
        self.current_staleness_steps = cand_staleness
        self.cumulative_kl_drift = cand_drift
        self.retention_count += 1
        decision = KVRetentionDecision(
            policy_version=policy_version,
            step_kl_shift=step_kl_shift,
            cumulative_kl_drift=self.cumulative_kl_drift,
            retained=True,
            invalidated=False,
            staleness_steps=self.current_staleness_steps,
            retained_blocks=len(self._blocks),
            reason="retained_within_kl_and_staleness_bounds",
        )
        self._decisions.append(decision)
        return decision

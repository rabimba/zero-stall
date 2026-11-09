"""VeRL External Module Extension (`VERL_USE_EXTERNAL_MODULES=verl_hardware_plugin,src.verl_tpu_plugin_ext`).

Integrates Optimization A (Async Block-Scaled FP8 `tpu-sync` Weight Streaming) and
Optimization B (Bounded-Staleness Cross-Step KV Prefix Reuse + TIS GRPO Loss) directly into
upstream `verl` (`verl.trainer.main_ppo`), `torchtitan`, and `vllm-torchtpu` without forking
core upstream files.

Safe to import on non-TPU or non-VeRL machines (all upstream hooks activate conditionally).
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Iterable

import torch

from src.async_tpu_sync_weights import (
    BlockScaledFP8Quantizer,
    QuantizedTensorPacket,
    inject_packet_into_vllm_module,
)
from src.cross_step_kv_grpo import (
    BoundedStalenessKVController,
    LOG_RATIO_SAFETY_CLAMP,
    compute_cross_step_kv_grpo_loss,
    estimate_empirical_kl_shift,
)

logger = logging.getLogger(__name__)

# Process-wide singleton KV controller configured via environment variables on driver/workers
_GLOBAL_KV_CONTROLLER: BoundedStalenessKVController | None = None
_LAST_OBSERVED_BATCH_KL: float = 0.0


def get_global_kv_controller() -> BoundedStalenessKVController:
    """Return or lazily construct the process-level `BoundedStalenessKVController`."""
    global _GLOBAL_KV_CONTROLLER
    if _GLOBAL_KV_CONTROLLER is None:
        delta = float(os.environ.get("ZERO_STALL_KV_KL_DELTA", "0.02"))
        max_stale = int(os.environ.get("ZERO_STALL_KV_MAX_STALENESS", "4"))
        blk_tokens = int(os.environ.get("ZERO_STALL_KV_BLOCK_SIZE", "16"))
        _GLOBAL_KV_CONTROLLER = BoundedStalenessKVController(
            delta_kl_threshold=delta,
            max_staleness_steps=max_stale,
            block_size_tokens=blk_tokens,
        )
    return _GLOBAL_KV_CONTROLLER


def verl_cross_step_kv_grpo_policy_loss(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Any | None = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """VeRL `@register_policy_loss("cross_step_kv_grpo")` compatible entrypoint.

    Matches exact signature of `verl.trainer.ppo.core_algos.compute_policy_loss_vanilla`
    and `compute_policy_loss_bypass_mode`, preserving `agg_loss` `dp_size` scaling for
    TorchTitan `SimpleFSDP`.
    """
    global _LAST_OBSERVED_BATCH_KL

    clip_low = float(getattr(config, "clip_ratio_low", 0.2) or 0.2) if config is not None else 0.2
    clip_high = float(getattr(config, "clip_ratio_high", 0.2) or 0.2) if config is not None else 0.2
    clip_c = float(getattr(config, "clip_ratio_c", 3.0) or 3.0) if config is not None else 3.0
    tis_max = float(os.environ.get("ZERO_STALL_TIS_MAX", "3.0"))
    tis_min = float(os.environ.get("ZERO_STALL_TIS_MIN", "0.2"))

    kv_ctrl = get_global_kv_controller()
    staleness = kv_ctrl.current_staleness_steps if os.environ.get("ZERO_STALL_ENABLE_KV_REUSE") == "1" else 0

    with torch.no_grad():
        _LAST_OBSERVED_BATCH_KL = estimate_empirical_kl_shift(
            prev_log_prob=old_log_prob,
            curr_log_prob=log_prob.detach(),
            response_mask=response_mask,
        )

    # Try using VeRL's native `agg_loss` if installed so multi-rank SimpleFSDP `dp_size` scaling matches 1:1
    try:
        from verl.trainer.ppo.core_algos import agg_loss  # type: ignore[import-not-found]

        has_verl_agg = True
    except Exception:
        agg_loss = None
        has_verl_agg = False

    if not has_verl_agg or config is None:
        loss, metrics = compute_cross_step_kv_grpo_loss(
            old_log_prob=old_log_prob,
            log_prob=log_prob,
            advantages=advantages,
            response_mask=response_mask,
            staleness_steps=staleness,
            stale_behavior_log_prob=old_log_prob if staleness == 0 else old_log_prob.detach(),
            kl_coef=0.0,  # Outer `verl.workers.utils.losses.ppo_loss` adds ref KL loss separately
            clip_ratio_low=clip_low,
            clip_ratio_high=clip_high,
            clip_ratio_c=clip_c,
            tis_clip_min=tis_min,
            tis_clip_max=tis_max,
        )
        metrics["actor/empirical_step_kl"] = _LAST_OBSERVED_BATCH_KL
        metrics["actor/kv_staleness_steps"] = float(staleness)
        return loss, metrics

    # Native VeRL path with `agg_loss(..., **config.global_batch_info)`
    neg_approx_kl = torch.clamp(log_prob - old_log_prob, min=-LOG_RATIO_SAFETY_CLAMP, max=LOG_RATIO_SAFETY_CLAMP)
    ratio = torch.exp(neg_approx_kl)

    pg_losses1 = -advantages * ratio
    pg_losses2 = -advantages * torch.clamp(ratio, 1.0 - clip_low, 1.0 + clip_high)
    clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)
    pg_losses3 = -advantages * clip_c
    clip_pg_losses2 = torch.minimum(pg_losses3, clip_pg_losses1)
    pg_losses = torch.where(advantages < 0.0, clip_pg_losses2, clip_pg_losses1)

    tis_ess_ratio = 1.0
    if rollout_is_weights is not None:
        w = torch.clamp(rollout_is_weights.detach(), min=tis_min, max=tis_max)
        pg_losses = pg_losses * w
        with torch.no_grad():
            mask_f = response_mask.float()
            denom = mask_f.sum().clamp(min=1.0)
            wm = float(((w * mask_f).sum() / denom).item())
            w2 = float(((w * w * mask_f).sum() / denom).item())
            tis_ess_ratio = (wm * wm) / max(w2, 1e-12)
    elif staleness > 0:
        with torch.no_grad():
            prefix_mask = kv_ctrl.compute_low_divergence_prefix_mask(
                prev_log_prob=old_log_prob,
                curr_log_prob=log_prob.detach(),
                response_mask=response_mask,
            )
            kv_delta = torch.clamp(log_prob.detach() - old_log_prob.detach(), -LOG_RATIO_SAFETY_CLAMP, LOG_RATIO_SAFETY_CLAMP)
            w = torch.clamp(torch.exp(-kv_delta * prefix_mask), min=tis_min, max=tis_max).to(pg_losses.dtype)
            mask_f = response_mask.float()
            denom = mask_f.sum().clamp(min=1.0)
            wm = float(((w * mask_f).sum() / denom).item())
            w2 = float(((w * w * mask_f).sum() / denom).item())
            tis_ess_ratio = (wm * wm) / max(w2, 1e-12)
        pg_losses = pg_losses * w.detach()

    global_batch_info = getattr(config, "global_batch_info", {}) or {}
    pg_loss = agg_loss(
        loss_mat=pg_losses,
        loss_mask=response_mask,
        loss_agg_mode=loss_agg_mode,
        **global_batch_info,
    )
    metrics = {
        "actor/empirical_step_kl": _LAST_OBSERVED_BATCH_KL,
        "actor/kv_staleness_steps": float(staleness),
        "actor/tis_ess_ratio": float(tis_ess_ratio),
    }
    return pg_loss, metrics


def quantize_parameter_stream_fp8(
    per_tensor_params: Iterable[tuple[str, torch.Tensor]],
    quantizer: BlockScaledFP8Quantizer | None = None,
) -> Iterable[tuple[str, torch.Tensor]]:
    """Transform outgoing `(hf_name, bf16_tensor)` generator from `TorchTitanEngine.get_per_tensor_param`
    into an interleaved `(name, fp8_tensor)` + `(name.__fp8_scale__, float32_scale)` stream
    compatible with `tpu-sync`'s native `float8_e4m3fn` + `float32` C++ `UnpackTorchTensor` pipeline!
    """
    q = quantizer or BlockScaledFP8Quantizer()
    for name, param in per_tensor_params:
        pkt = q.quantize_tensor(name, param)
        if pkt.is_quantized and pkt.scale is not None:
            yield (pkt.name, pkt.data)
            yield (f"{pkt.name}.__fp8_scale__", pkt.scale)
        else:
            yield (pkt.name, pkt.data)


def install_vllm_tpu_weight_sync_patch() -> bool:
    """Patch `vllm_torchtpu.worker.tpu_worker.TPUWorker` to handle FP8 + `_tpu_weight_flipped` reloads."""
    try:
        from vllm_torchtpu.worker.tpu_worker import TPUWorker  # type: ignore[import-not-found]
    except Exception:
        return False

    if getattr(TPUWorker, "_zero_stall_fp8_patched", False):
        return True

    quantizer = BlockScaledFP8Quantizer()

    def load_fp8_stream_weights(worker_self: Any, weights_iter: Iterable[tuple[str, torch.Tensor]]) -> int:
        model = worker_self.model_runner.model
        pending_data: dict[str, torch.Tensor] = {}
        pending_scales: dict[str, torch.Tensor] = {}
        applied = 0

        for name, tensor in weights_iter:
            if name.endswith(".__fp8_scale__"):
                base = name[: -len(".__fp8_scale__")]
                pending_scales[base] = tensor
                if base in pending_data:
                    d_t = pending_data.pop(base)
                    sc_t = pending_scales.pop(base)
                    pkt = QuantizedTensorPacket(
                        name=base,
                        is_quantized=True,
                        data=d_t,
                        scale=sc_t,
                        orig_shape=tuple(d_t.shape),
                        orig_dtype=torch.bfloat16,
                        block_size=quantizer.block_size,
                    )
                    applied += int(inject_packet_into_vllm_module(model, pkt, quantizer))
            else:
                if tensor.dtype == torch.float8_e4m3fn:
                    if name in pending_scales:
                        sc_t = pending_scales.pop(name)
                        pkt = QuantizedTensorPacket(
                            name=name,
                            is_quantized=True,
                            data=tensor,
                            scale=sc_t,
                            orig_shape=tuple(tensor.shape),
                            orig_dtype=torch.bfloat16,
                            block_size=quantizer.block_size,
                        )
                        applied += int(inject_packet_into_vllm_module(model, pkt, quantizer))
                    else:
                        pending_data[name] = tensor
                else:
                    pkt = QuantizedTensorPacket(
                        name=name,
                        is_quantized=False,
                        data=tensor,
                        scale=None,
                        orig_shape=tuple(tensor.shape),
                        orig_dtype=tensor.dtype,
                        block_size=quantizer.block_size,
                    )
                    applied += int(inject_packet_into_vllm_module(model, pkt, quantizer))

        if hasattr(torch, "accelerator") and torch.accelerator.is_available():
            torch.accelerator.synchronize()
        return applied

    TPUWorker.load_fp8_stream_weights = load_fp8_stream_weights
    TPUWorker._zero_stall_fp8_patched = True
    return True


def register_verl_extensions() -> dict[str, bool]:
    """Register `cross_step_kv_grpo` policy loss, `tpu-sync-fp8-async` checkpoint backend,
    and KL-gated KV cache retention hook inside `verl` if imported.
    """
    status = {
        "policy_loss_registered": False,
        "checkpoint_engine_registered": False,
        "kv_retention_hook_installed": False,
        "vllm_worker_patched": install_vllm_tpu_weight_sync_patch(),
    }

    # 1. Register Policy Loss
    try:
        from verl.trainer.ppo.core_algos import POLICY_LOSS_REGISTRY, register_policy_loss  # type: ignore[import-not-found]

        if "cross_step_kv_grpo" not in POLICY_LOSS_REGISTRY:
            register_policy_loss("cross_step_kv_grpo")(verl_cross_step_kv_grpo_policy_loss)
        status["policy_loss_registered"] = True
    except Exception as exc:
        logger.debug("Skipping VeRL policy loss registration (verl not imported): %s", exc)

    # 2. Register `tpu-sync-fp8-async` CheckpointEngine subclassing `tpu-sync`
    try:
        from verl.checkpoint_engine.base import CheckpointEngineManager, CheckpointEngineRegistry  # type: ignore[import-not-found]

        base_tpu_sync_cls = CheckpointEngineRegistry.get("tpu-sync") if hasattr(CheckpointEngineRegistry, "get") else None
        if base_tpu_sync_cls is not None:

            @CheckpointEngineRegistry.register("tpu-sync-fp8-async")
            class TPUSyncFP8AsyncCheckpointEngine(base_tpu_sync_cls):  # type: ignore[misc,valid-type]
                """Streams `BlockScaledFP8Quantizer` (`float8_e4m3fn` + `float32` scales) over `tpu-sync`."""

                support_publish_weights: bool = True

                async def send_weights(self, per_tensor_param: Iterable[tuple[str, torch.Tensor]], **kwargs: Any) -> Any:
                    t0 = time.perf_counter()
                    fp8_stream = quantize_parameter_stream_fp8(per_tensor_param)
                    res = await super().send_weights(fp8_stream, **kwargs)
                    logger.info(
                        "[Zero-Stall] Streamed FP8 block-scaled policy weights via tpu-sync in %.3fs",
                        time.perf_counter() - t0,
                    )
                    return res

            status["checkpoint_engine_registered"] = True

        # 3. Install conditional KL-gated KV prefix retention hook on CheckpointEngineManager
        if not getattr(CheckpointEngineManager, "_zero_stall_kv_hooked", False):
            orig_release = getattr(CheckpointEngineManager, "release_kv_cache_replicas", None)
            if callable(orig_release):

                async def gated_release_kv_cache_replicas(mgr_self: Any, *args: Any, **kwargs: Any) -> Any:
                    if os.environ.get("ZERO_STALL_ENABLE_KV_REUSE") != "1":
                        return await orig_release(mgr_self, *args, **kwargs)

                    ctrl = get_global_kv_controller()
                    step_version = int(getattr(mgr_self, "_zero_stall_step", 0)) + 1
                    mgr_self._zero_stall_step = step_version
                    decision = ctrl.step_policy_update(_LAST_OBSERVED_BATCH_KL, policy_version=step_version)
                    if decision.retained:
                        logger.info(
                            "[Zero-Stall KV] Retained prefix KV cache at policy v%d (KL=%.5f, cum=%.5f, staleness=%d)",
                            step_version,
                            decision.step_kl_shift,
                            decision.cumulative_kl_drift,
                            decision.staleness_steps,
                        )
                        return None
                    logger.info(
                        "[Zero-Stall KV] Invalidating prefix KV cache at policy v%d (%s)",
                        step_version,
                        decision.reason,
                    )
                    return await orig_release(mgr_self, *args, **kwargs)

                CheckpointEngineManager.release_kv_cache_replicas = gated_release_kv_cache_replicas
                CheckpointEngineManager._zero_stall_kv_hooked = True
                status["kv_retention_hook_installed"] = True
    except Exception as exc:
        logger.debug("Skipping VeRL CheckpointEngine hook (verl not imported): %s", exc)

    return status


# Auto-register upon module import by `VERL_USE_EXTERNAL_MODULES`
REGISTRATION_STATUS = register_verl_extensions()

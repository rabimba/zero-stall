"""Phase 2 Gate (Optimization A): Block-Scaled FP8 TPU Sync Weight Streaming & 20-Cycle Stress Test."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.async_tpu_sync_weights import (
    FP8_DTYPE,
    AsyncTPUSyncWeightStreamer,
    BlockScaledFP8Quantizer,
    compute_reconstruction_metrics,
    inject_packet_into_vllm_module,
)


class MockQwenDecoderBlock(nn.Module):
    """Representative Qwen3 decoder block with BF16 projections and RMSNorm weights."""

    def __init__(self, hidden: int = 512, inter: int = 1536) -> None:
        super().__init__()
        self.input_layernorm_weight = nn.Parameter(torch.ones(hidden, dtype=torch.bfloat16))
        self.qkv_proj = nn.Linear(hidden, hidden * 3, bias=False, dtype=torch.bfloat16)
        self.o_proj = nn.Linear(hidden, hidden, bias=False, dtype=torch.bfloat16)
        self.gate_up_proj = nn.Linear(hidden, inter * 2, bias=False, dtype=torch.bfloat16)
        self.down_proj = nn.Linear(inter, hidden, bias=False, dtype=torch.bfloat16)


class FlippedTPULinearModule(nn.Module):
    """Simulates `vllm-torchtpu`'s post-load `[K_in, N_out]` `_tpu_weight_flipped = True` layer."""

    def __init__(self, k_in: int, n_out: int, native_fp8: bool = False) -> None:
        super().__init__()
        self._tpu_weight_flipped = True
        if native_fp8:
            self.weight = nn.Parameter(
                torch.zeros((k_in, n_out), dtype=FP8_DTYPE),
                requires_grad=False,
            )
            self.register_buffer(
                "weight_scale",
                torch.ones((1, k_in // 128, 1, n_out), dtype=torch.float32),
            )
        else:
            self.weight = nn.Parameter(torch.zeros((k_in, n_out), dtype=torch.bfloat16))


def test_block_scaled_fp8_reconstruction_error_bounds() -> None:
    """Assert cosine similarity > 0.999 against both exact FP8 cast and BF16 source tensors."""
    torch.manual_seed(1234)
    quantizer = BlockScaledFP8Quantizer(block_size=(128, 128))

    shapes_to_test = [
        ("layers.0.self_attn.qkv_proj.weight", (1536, 1024)),
        ("layers.0.mlp.gate_up_proj.weight", (3072, 1024)),
        ("layers.0.mlp.down_proj.weight", (1024, 3072)),
        ("layers.0.unaligned_tail_proj.weight", (307, 519)),  # non-multiple of 128 edge case
    ]

    for name, shape in shapes_to_test:
        src_bf16 = (torch.randn(shape, dtype=torch.float32) * 0.04).to(torch.bfloat16)
        pkt = quantizer.quantize_tensor(name, src_bf16)
        assert pkt.is_quantized
        assert pkt.data.dtype == FP8_DTYPE
        assert pkt.scale is not None and pkt.scale.dtype == torch.float32

        recon_bf16 = quantizer.dequantize_packet(pkt, target_dtype=torch.bfloat16)
        exact_fp8 = quantizer.exact_per_block_fp8_reference(src_bf16, block_size=(128, 128))
        metrics = compute_reconstruction_metrics(
            original=src_bf16,
            reconstructed=recon_bf16,
            exact_fp8_ref=exact_fp8,
            transferred_bytes=pkt.payload_bytes,
        )

        assert metrics.cosine_sim_vs_exact_fp8 > 0.9999, (
            f"{name}: cos_sim vs exact FP8 = {metrics.cosine_sim_vs_exact_fp8:.6f} <= 0.9999"
        )
        assert metrics.cosine_sim_vs_bf16 > 0.999, (
            f"{name}: cos_sim vs BF16 = {metrics.cosine_sim_vs_bf16:.6f} <= 0.999"
        )
        if shape[0] >= 1024 and shape[1] >= 1024:
            assert metrics.compression_ratio > 1.98, (
                f"Expected ~2x payload reduction on {shape}, got {metrics.compression_ratio:.3f}x"
            )

    # Verify 1D RMSNorm stays lossless in BF16
    norm_w = torch.randn(1024, dtype=torch.bfloat16)
    norm_pkt = quantizer.quantize_tensor("layers.0.input_layernorm.weight", norm_w)
    assert not norm_pkt.is_quantized
    assert torch.equal(norm_w, quantizer.dequantize_packet(norm_pkt))


def test_vllm_torchtpu_flipped_layout_injection() -> None:
    """Verify injection into `vllm-torchtpu`'s `[K_in, N_out]` `_tpu_weight_flipped` layers."""
    torch.manual_seed(99)
    quantizer = BlockScaledFP8Quantizer(block_size=(128, 128))
    n_out, k_in = 256, 512
    trainer_weight = (torch.randn(n_out, k_in, dtype=torch.float32) * 0.05).to(torch.bfloat16)
    pkt = quantizer.quantize_tensor("linear.weight", trainer_weight)

    # 1. BF16 flipped module
    container_bf16 = nn.Module()
    container_bf16.linear = FlippedTPULinearModule(k_in=k_in, n_out=n_out, native_fp8=False)
    assert inject_packet_into_vllm_module(container_bf16, pkt, quantizer)
    expected_t = quantizer.dequantize_packet(pkt, transpose_2d_for_tpu=True)
    assert container_bf16.linear.weight.shape == (k_in, n_out)
    assert torch.equal(container_bf16.linear.weight, expected_t)

    # 2. Native FP8 + gmm_v2 4D scale flipped module
    container_fp8 = nn.Module()
    container_fp8.linear = FlippedTPULinearModule(k_in=k_in, n_out=n_out, native_fp8=True)
    assert inject_packet_into_vllm_module(container_fp8, pkt, quantizer)
    assert container_fp8.linear.weight.dtype == FP8_DTYPE
    assert container_fp8.linear.weight.shape == (k_in, n_out)
    assert torch.equal(container_fp8.linear.weight, pkt.data.transpose(0, 1).contiguous())


def test_twenty_rapid_async_sync_cycles_zero_deadlocks(require_tpu_v7: bool) -> None:
    """Run 20 consecutive rapid async BF16->FP8 sync cycles overlapping simulated rollout prep.
    Asserts zero deadlocks, monotonic versions, and >0.999 cosine similarity at cycle 20.
    """
    torch.manual_seed(2026)
    device = torch.device("tpu") if (hasattr(torch, "tpu") and torch.tpu.is_available()) else torch.device("cpu")

    trainer_model = MockQwenDecoderBlock(hidden=512, inter=1536).to(device)
    rollout_model = MockQwenDecoderBlock(hidden=512, inter=1536).to(device)

    with AsyncTPUSyncWeightStreamer(require_tpu=require_tpu_v7, timeout_s=30.0) as streamer:
        completed_versions: list[int] = []
        pending_future = None

        for cycle in range(1, 21):
            # Mutate trainer parameters as if an optimizer step just finished
            with torch.no_grad():
                for p in trainer_model.parameters():
                    p.add_(torch.randn_like(p) * 0.01)

            # Drain previous in-flight transfer before mutating target again, overlapping with prep
            if pending_future is not None:
                stats = streamer.wait_sync(pending_future)
                completed_versions.append(stats.policy_version)

            pending_future = streamer.submit_async_sync(
                trainer_model,
                rollout_model,
                policy_version=cycle,
            )

            # Simulate concurrent rollout prompt batch preparation on host/rollout worker
            dummy_rollout_prep = torch.randn((32, 512), dtype=torch.bfloat16, device=device)
            _ = dummy_rollout_prep.sum()

        assert pending_future is not None
        final_stats = streamer.wait_sync(pending_future)
        completed_versions.append(final_stats.policy_version)

    assert completed_versions == list(range(1, 21)), f"Unexpected sync sequence: {completed_versions}"

    # Verify cycle-20 rollout weights match trainer weights with cosine similarity > 0.999
    t_params = dict(trainer_model.named_parameters())
    r_params = dict(rollout_model.named_parameters())
    for k, src_p in t_params.items():
        dst_p = r_params[k]
        cos = float(
            F.cosine_similarity(
                src_p.detach().reshape(1, -1).cpu().double(),
                dst_p.detach().reshape(1, -1).cpu().double(),
            ).item()
        )
        assert cos > 0.999, f"Cycle 20 parameter {k} cosine similarity {cos:.6f} <= 0.999"

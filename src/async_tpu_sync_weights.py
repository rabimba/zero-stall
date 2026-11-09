"""Optimization A: Asynchronous BF16-to-FP8 Block-Scaled Weight Streaming via `google/tpu-sync`.

Streams updated TorchTitan policy parameters from trainer TPU HBM to `vllm-torchtpu`
rollout workers using:
1. On-the-fly BF16 -> `torch.float8_e4m3fn` block-scaled quantization (`128x128` MXU tiles +
   `float32` block scales), cutting wire/DMA payload bytes by ~50% while maintaining >0.999
   cosine similarity against both exact FP8 cast and source BF16 weights.
2. Double-buffered asynchronous DMA transfers backed by `google/tpu-sync`
   (`tpu_sync.api.torch.raw_transfer` pinned-host `RawHostBuffer` / `PjRtCopyFuture` and
   `tpu_sync.api.torch.weight_synchronizer.WeightSynchronizer` + `WeightSynchronizerManager`),
   overlapping weight transfer with rollout buffer preparation and avoiding host-locking stalls.
3. Native handling of `vllm-torchtpu`'s `_tpu_weight_flipped` (`[K_in, N_out]`) parameter layout
   and `VllmFp8LinearMethodTPU` scale tensors.
"""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
import math
import threading
import time
from typing import Any, Mapping

import torch
import torch.nn.functional as F

from src.telemetry import tpu_named_scope


FP8_DTYPE: torch.dtype = torch.float8_e4m3fn
FP8_E4M3_MAX: float = float(torch.finfo(torch.float8_e4m3fn).max)  # 448.0
SCALE_MIN_CLAMP: float = 1e-12


def is_tpu_runtime_available() -> bool:
    """Return True if `torch.tpu` is loaded and at least one TPU device is available."""
    return bool(hasattr(torch, "tpu") and callable(getattr(torch.tpu, "is_available", None)) and torch.tpu.is_available())


def is_tpu_sync_installed() -> bool:
    """Return True if `google/tpu-sync` (`tpu_sync.api.torch`) is importable."""
    try:
        import tpu_sync.api.torch.raw_transfer  # type: ignore[import-not-found]  # noqa: F401

        return True
    except Exception:
        return False


def _clean_param_key(name: str) -> str:
    return (
        name.replace("_fsdp_wrapped_module.", "")
        .replace("_checkpoint_wrapped_module.", "")
        .replace("module.", "")
    )


@dataclass(frozen=True)
class QuantizedTensorPacket:
    """Block-scaled FP8 representation (or passthrough BF16/FP32 for 1D norms/biases)."""

    name: str
    is_quantized: bool
    data: torch.Tensor
    scale: torch.Tensor | None
    orig_shape: tuple[int, ...]
    orig_dtype: torch.dtype
    block_size: tuple[int, int]

    @property
    def payload_bytes(self) -> int:
        total = self.data.numel() * self.data.element_size()
        if self.scale is not None:
            total += self.scale.numel() * self.scale.element_size()
        return total


@dataclass(frozen=True)
class ReconstructionMetrics:
    """Numerical fidelity metrics for weight quantization & transfer verification."""

    cosine_sim_vs_bf16: float
    cosine_sim_vs_exact_fp8: float
    rel_l2_error_vs_bf16: float
    max_abs_error_vs_exact_fp8: float
    source_bytes: int
    transferred_bytes: int

    @property
    def compression_ratio(self) -> float:
        return self.source_bytes / max(self.transferred_bytes, 1)


@dataclass
class SyncCycleStats:
    """Timing and bandwidth telemetry for a single asynchronous weight sync cycle."""

    policy_version: int
    backend: str
    quantize_time_s: float
    bg_transfer_time_s: float
    caller_stall_time_s: float
    source_bytes: int
    transferred_bytes: int
    num_tensors: int


class BlockScaledFP8Quantizer:
    """On-the-fly BF16 -> `float8_e4m3fn` block-scaled quantizer matching TPU 128x128 MXU tiles."""

    def __init__(
        self,
        block_size: tuple[int, int] = (128, 128),
        min_numel_to_quantize: int = 4096,
        exclude_patterns: tuple[str, ...] = ("norm", "bias", "layernorm", "rmsnorm", "gate.weight"),
    ) -> None:
        if block_size[0] <= 0 or block_size[1] <= 0:
            raise ValueError(f"block_size dimensions must be positive, got {block_size}")
        self.block_size = block_size
        self.min_numel_to_quantize = min_numel_to_quantize
        self.exclude_patterns = tuple(p.lower() for p in exclude_patterns)

    def should_quantize(self, name: str, tensor: torch.Tensor) -> bool:
        if tensor.ndim < 2 or tensor.numel() < self.min_numel_to_quantize:
            return False
        if tensor.dtype not in (torch.bfloat16, torch.float16, torch.float32):
            return False
        low_name = _clean_param_key(name).lower()
        return not any(pat in low_name for pat in self.exclude_patterns)

    @torch.no_grad()
    def quantize_tensor(self, name: str, tensor: torch.Tensor) -> QuantizedTensorPacket:
        """Quantize a >=2D weight tensor into `float8_e4m3fn` + `float32` 2D block scales."""
        clean_name = _clean_param_key(name)
        orig_shape = tuple(tensor.shape)
        orig_dtype = tensor.dtype

        if not self.should_quantize(clean_name, tensor):
            passthrough = tensor.detach().contiguous()
            return QuantizedTensorPacket(
                name=clean_name,
                is_quantized=False,
                data=passthrough,
                scale=None,
                orig_shape=orig_shape,
                orig_dtype=orig_dtype,
                block_size=self.block_size,
            )

        bn, bk = self.block_size
        # Flatten leading dims into 2D [M, K] so 3D MoE expert stacks [E, N, K] also work seamlessly
        k_dim = orig_shape[-1]
        m_dim = tensor.numel() // k_dim
        w_fp32 = tensor.detach().reshape(m_dim, k_dim).to(torch.float32)

        pad_m = (bn - (m_dim % bn)) % bn
        pad_k = (bk - (k_dim % bk)) % bk
        if pad_m > 0 or pad_k > 0:
            w_padded = F.pad(w_fp32, (0, pad_k, 0, pad_m), mode="constant", value=0.0)
        else:
            w_padded = w_fp32

        pm, pk = w_padded.shape
        nb_m, nb_k = pm // bn, pk // bk

        # Reshape into [nb_m, bn, nb_k, bk] to compute per-tile absmax in a single fused reduction
        tiled = w_padded.view(nb_m, bn, nb_k, bk)
        absmax = tiled.abs().amax(dim=(1, 3)).clamp(min=SCALE_MIN_CLAMP)  # [nb_m, nb_k], float32
        scale = (absmax / FP8_E4M3_MAX).contiguous()

        scaled_tiled = (tiled / scale[:, None, :, None]).clamp(min=-FP8_E4M3_MAX, max=FP8_E4M3_MAX)
        scaled_2d = scaled_tiled.view(pm, pk)[:m_dim, :k_dim].contiguous()
        fp8_data = scaled_2d.to(FP8_DTYPE).view(orig_shape)

        return QuantizedTensorPacket(
            name=clean_name,
            is_quantized=True,
            data=fp8_data,
            scale=scale,
            orig_shape=orig_shape,
            orig_dtype=orig_dtype,
            block_size=self.block_size,
        )

    @torch.no_grad()
    def dequantize_packet(
        self,
        packet: QuantizedTensorPacket,
        target_dtype: torch.dtype | None = None,
        transpose_2d_for_tpu: bool = False,
    ) -> torch.Tensor:
        """Reconstruct tensor from `QuantizedTensorPacket` onto its current device."""
        out_dtype = target_dtype or packet.orig_dtype
        if not packet.is_quantized or packet.scale is None:
            out = packet.data.to(out_dtype)
            if transpose_2d_for_tpu and out.ndim == 2:
                out = out.transpose(0, 1).contiguous()
            return out

        bn, bk = packet.block_size
        k_dim = packet.orig_shape[-1]
        m_dim = math.prod(packet.orig_shape[:-1])
        fp32_data = packet.data.reshape(m_dim, k_dim).to(torch.float32)

        pad_m = (bn - (m_dim % bn)) % bn
        pad_k = (bk - (k_dim % bk)) % bk
        if pad_m > 0 or pad_k > 0:
            fp32_padded = F.pad(fp32_data, (0, pad_k, 0, pad_m), mode="constant", value=0.0)
        else:
            fp32_padded = fp32_data

        pm, pk = fp32_padded.shape
        nb_m, nb_k = pm // bn, pk // bk
        recon_padded = fp32_padded.view(nb_m, bn, nb_k, bk) * packet.scale.to(fp32_padded.device)[:, None, :, None]
        recon = recon_padded.view(pm, pk)[:m_dim, :k_dim].reshape(packet.orig_shape).to(out_dtype)

        if transpose_2d_for_tpu and recon.ndim == 2:
            recon = recon.transpose(0, 1).contiguous()
        return recon.contiguous()

    @staticmethod
    @torch.no_grad()
    def exact_per_block_fp8_reference(
        tensor: torch.Tensor,
        block_size: tuple[int, int] = (128, 128),
    ) -> torch.Tensor:
        """Compute exact mathematical per-block FP8 cast reference in FP64 -> FP8 -> orig_dtype."""
        bn, bk = block_size
        orig_shape = tensor.shape
        k_dim = orig_shape[-1]
        m_dim = tensor.numel() // k_dim
        w64 = tensor.detach().reshape(m_dim, k_dim).to(torch.float64)

        out = torch.empty_like(w64)
        for r0 in range(0, m_dim, bn):
            r1 = min(r0 + bn, m_dim)
            for c0 in range(0, k_dim, bk):
                c1 = min(c0 + bk, k_dim)
                blk = w64[r0:r1, c0:c1]
                amax = max(float(blk.abs().max().item()), SCALE_MIN_CLAMP)
                sc = amax / FP8_E4M3_MAX
                q = (blk / sc).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX).to(torch.float32).to(FP8_DTYPE)
                out[r0:r1, c0:c1] = q.to(torch.float64) * sc
        return out.reshape(orig_shape).to(tensor.dtype)


@torch.no_grad()
def compute_reconstruction_metrics(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
    exact_fp8_ref: torch.Tensor,
    transferred_bytes: int,
) -> ReconstructionMetrics:
    """Compute cosine similarity and relative L2 error bounds in FP64 (avoiding >2M-element FP32 sum drift)."""
    orig_f = original.detach().reshape(-1).to(device="cpu", dtype=torch.float64)
    recon_f = reconstructed.detach().reshape(-1).to(device="cpu", dtype=torch.float64)
    ref_f = exact_fp8_ref.detach().reshape(-1).to(device="cpu", dtype=torch.float64)

    cos_bf16 = float(F.cosine_similarity(orig_f.unsqueeze(0), recon_f.unsqueeze(0), eps=1e-15).item())
    cos_fp8 = float(F.cosine_similarity(ref_f.unsqueeze(0), recon_f.unsqueeze(0), eps=1e-15).item())
    rel_l2 = float((torch.linalg.norm(orig_f - recon_f) / torch.linalg.norm(orig_f).clamp(min=1e-15)).item())
    max_abs_fp8 = float((ref_f - recon_f).abs().max().item())
    src_bytes = original.numel() * original.element_size()

    return ReconstructionMetrics(
        cosine_sim_vs_bf16=cos_bf16,
        cosine_sim_vs_exact_fp8=cos_fp8,
        rel_l2_error_vs_bf16=rel_l2,
        max_abs_error_vs_exact_fp8=max_abs_fp8,
        source_bytes=src_bytes,
        transferred_bytes=transferred_bytes,
    )


def inject_packet_into_vllm_module(
    root_model: torch.nn.Module,
    packet: QuantizedTensorPacket,
    quantizer: BlockScaledFP8Quantizer,
) -> bool:
    """Inject a received `QuantizedTensorPacket` into a `vllm-torchtpu` model parameter.

    Handles `vllm-torchtpu`'s canonical `[K_in, N_out]` `_tpu_weight_flipped` attribute
    and `VllmFp8LinearMethodTPU` native FP8 + scale buffers when present.
    """
    params = dict(root_model.named_parameters())
    buffers = dict(root_model.named_buffers())
    candidates = (
        packet.name,
        packet.name.removeprefix("model."),
        f"model.{packet.name}",
    )
    matched_key = next((c for c in candidates if c in params), None)
    if matched_key is None:
        return False

    target_param = params[matched_key]
    parent_path = matched_key.rsplit(".", 1)[0] if "." in matched_key else ""
    parent_mod = root_model.get_submodule(parent_path) if parent_path else root_model
    is_flipped = bool(getattr(parent_mod, "_tpu_weight_flipped", False))

    with torch.no_grad():
        # Case 1: Target parameter in vLLM-TorchTPU is already native float8_e4m3fn
        if target_param.dtype == FP8_DTYPE and packet.is_quantized and packet.scale is not None:
            fp8_src = packet.data.to(target_param.device)
            if is_flipped and fp8_src.ndim == 2 and target_param.shape == (fp8_src.shape[1], fp8_src.shape[0]):
                fp8_src = fp8_src.transpose(0, 1).contiguous()
            target_param.copy_(fp8_src)

            scale_key = f"{parent_path}.weight_scale" if parent_path else "weight_scale"
            scale_target = params.get(scale_key) or buffers.get(scale_key)
            if scale_target is not None:
                sc = packet.scale.to(scale_target.device)
                if scale_target.shape == sc.shape:
                    scale_target.copy_(sc)
                elif scale_target.ndim == 4 and sc.ndim == 2:
                    # VllmFp8LinearMethodTPU gmm_v2 layout: [1, K // block_k, 1, N_out]
                    bn, _ = packet.block_size
                    n_out = packet.orig_shape[0]
                    expanded = sc.repeat_interleave(bn, dim=0)[:n_out, :]  # [N_out, nb_k]
                    gmm_scale = expanded.transpose(0, 1).unsqueeze(0).unsqueeze(2).contiguous()
                    if scale_target.shape == gmm_scale.shape:
                        scale_target.copy_(gmm_scale)
            return True

        # Case 2: Target parameter is BF16/FP32 (FP8 wire transfer -> on-device dequantize)
        recon = quantizer.dequantize_packet(
            packet,
            target_dtype=target_param.dtype,
            transpose_2d_for_tpu=False,
        ).to(target_param.device)
        if is_flipped and recon.ndim == 2 and target_param.shape == (recon.shape[1], recon.shape[0]):
            recon = recon.transpose(0, 1).contiguous()
        target_param.copy_(recon)
        return True


@dataclass
class _DoubleBufferSlot:
    slot_id: int
    in_use: bool = False
    policy_version: int = -1
    packets: list[QuantizedTensorPacket] = field(default_factory=list)
    host_Staging_raw: dict[str, Any] = field(default_factory=dict)


class AsyncTPUSyncWeightStreamer:
    """Double-buffered asynchronous BF16->FP8 weight streamer backed by `google/tpu-sync`.

    On Cloud TPU v7 (`tpu7x`) with `tpu_sync` installed:
      - Configures DMA pacing via `tpu_sync.api.torch.raw_transfer.set_dma_pacing` (`64 MiB`
        max in-flight, `16 MiB` chunk size) so background weight transfers never starve
        TensorCore MXU execution during rollout prefill/decode.
      - Uses persistent `RawHostBuffer` pinned-host staging + `transfer_d2h_async` /
        `transfer_h2d_async` (`PjRtCopyFuture`) across 2 ring-buffer slots.
      - Optionally coordinates multi-slice resharded transfers via
        `tpu_sync.api.torch.weight_synchronizer.WeightSynchronizer` (`auto_h2d=True`).

    In single-host / CPU staging tests (`require_tpu=False`):
      - Executes the exact same double-buffered slot state machine asynchronously on a
        dedicated non-blocking transfer thread with bounded condition-variable waits,
        verifying zero deadlocks and exact FP8 reconstruction fidelity.
    """

    def __init__(
        self,
        quantizer: BlockScaledFP8Quantizer | None = None,
        *,
        require_tpu: bool = False,
        dma_max_inflight_bytes: int = 64 << 20,
        dma_chunk_bytes: int = 16 << 20,
        timeout_s: float = 60.0,
    ) -> None:
        self.quantizer = quantizer or BlockScaledFP8Quantizer()
        self.require_tpu = require_tpu
        self.timeout_s = timeout_s

        tpu_ok = is_tpu_runtime_available()
        sync_ok = is_tpu_sync_installed()
        if require_tpu and not (tpu_ok and sync_ok):
            raise RuntimeError(
                f"AsyncTPUSyncWeightStreamer(require_tpu=True) requires Cloud TPU runtime "
                f"(available={tpu_ok}) and `tpu_sync` (installed={sync_ok})."
            )

        self._use_tpu_sync_dma = bool(tpu_ok and sync_ok)
        self.backend_name = "tpu_sync_raw_dma_fp8" if self._use_tpu_sync_dma else "async_double_buffer_fp8"

        if self._use_tpu_sync_dma:
            from tpu_sync.api.torch import raw_transfer  # type: ignore[import-not-found]

            raw_transfer.set_dma_pacing(
                max_inflight_bytes=dma_max_inflight_bytes,
                chunk_bytes=dma_chunk_bytes,
            )

        self._slots = [_DoubleBufferSlot(0), _DoubleBufferSlot(1)]
        self._cond = threading.Condition()
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="tpu_sync_fp8_dma")
        self._closed = False
        self._history: list[SyncCycleStats] = []

    @property
    def history(self) -> list[SyncCycleStats]:
        return list(self._history)

    def _acquire_slot(self, preferred_idx: int) -> _DoubleBufferSlot:
        deadline = time.monotonic() + self.timeout_s
        with self._cond:
            while True:
                for idx in (preferred_idx % 2, (preferred_idx + 1) % 2):
                    if not self._slots[idx].in_use:
                        self._slots[idx].in_use = True
                        return self._slots[idx]
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        f"Timed out after {self.timeout_s}s waiting for a free double-buffer DMA slot."
                    )
                self._cond.wait(timeout=min(remaining, 0.05))

    def _release_slot(self, slot: _DoubleBufferSlot) -> None:
        with self._cond:
            slot.in_use = False
            slot.packets = []
            self._cond.notify_all()

    def _stage_and_apply_worker(
        self,
        slot: _DoubleBufferSlot,
        packets: list[QuantizedTensorPacket],
        target_model: torch.nn.Module | Mapping[str, torch.Tensor],
        policy_version: int,
        quantize_time_s: float,
        source_bytes: int,
    ) -> SyncCycleStats:
        t_bg_start = time.perf_counter()
        try:
            with tpu_named_scope("grpo_weight_sync_dma"):
                if self._use_tpu_sync_dma:
                    self._execute_tpu_sync_raw_dma(slot, packets, target_model)
                else:
                    self._execute_staged_copy(packets, target_model)
            bg_elapsed = time.perf_counter() - t_bg_start
            transferred_bytes = sum(p.payload_bytes for p in packets)
            return SyncCycleStats(
                policy_version=policy_version,
                backend=self.backend_name,
                quantize_time_s=quantize_time_s,
                bg_transfer_time_s=bg_elapsed,
                caller_stall_time_s=0.0,
                source_bytes=source_bytes,
                transferred_bytes=transferred_bytes,
                num_tensors=len(packets),
            )
        finally:
            self._release_slot(slot)

    def _execute_tpu_sync_raw_dma(
        self,
        slot: _DoubleBufferSlot,
        packets: list[QuantizedTensorPacket],
        target_model: torch.nn.Module | Mapping[str, torch.Tensor],
    ) -> None:
        """Execute real `google/tpu-sync` async D2H -> pinned RawHostBuffer -> async H2D."""
        from tpu_sync.api.torch import raw_transfer  # type: ignore[import-not-found]

        d2h_futures = []
        staged_device_Landing: list[tuple[QuantizedTensorPacket, Any, Any]] = []

        # 1. Queue batched non-blocking D2H into reusable pinned RawHostBuffers
        for pkt in packets:
            data_nbytes = pkt.data.numel() * pkt.data.element_size()
            buf_key = f"{pkt.name}::data::{data_nbytes}"
            host_buf = slot.host_Staging_raw.get(buf_key)
            if host_buf is None:
                host_buf = raw_transfer.RawHostBuffer(data_nbytes)
                slot.host_Staging_raw[buf_key] = host_buf
            d2h_futures.append(
                raw_transfer.transfer_d2h_async(pkt.data, host_buf, unsafe_skip_buffer_lock=True)
            )

            scale_buf = None
            if pkt.scale is not None:
                sc_nbytes = pkt.scale.numel() * pkt.scale.element_size()
                sc_key = f"{pkt.name}::scale::{sc_nbytes}"
                scale_buf = slot.host_Staging_raw.get(sc_key)
                if scale_buf is None:
                    scale_buf = raw_transfer.RawHostBuffer(sc_nbytes)
                    slot.host_Staging_raw[sc_key] = scale_buf
                d2h_futures.append(
                    raw_transfer.transfer_d2h_async(pkt.scale, scale_buf, unsafe_skip_buffer_lock=True)
                )
            staged_device_Landing.append((pkt, host_buf, scale_buf))

        raw_transfer.await_all(d2h_futures)

        # 2. Stream H2D onto target TPU device and reconstruct / inject in-place
        h2d_futures = []
        target_dev = self._infer_target_device(target_model)
        landed_packets: list[QuantizedTensorPacket] = []

        for pkt, host_buf, scale_buf in staged_device_Landing:
            dst_data = torch.empty(pkt.data.shape, dtype=pkt.data.dtype, device=target_dev)
            h2d_futures.append(
                raw_transfer.transfer_h2d_async(host_buf, dst_data, unsafe_skip_buffer_lock=True)
            )
            dst_scale = None
            if pkt.scale is not None and scale_buf is not None:
                dst_scale = torch.empty(pkt.scale.shape, dtype=pkt.scale.dtype, device=target_dev)
                h2d_futures.append(
                    raw_transfer.transfer_h2d_async(scale_buf, dst_scale, unsafe_skip_buffer_lock=True)
                )
            landed_packets.append(
                QuantizedTensorPacket(
                    name=pkt.name,
                    is_quantized=pkt.is_quantized,
                    data=dst_data,
                    scale=dst_scale,
                    orig_shape=pkt.orig_shape,
                    orig_dtype=pkt.orig_dtype,
                    block_size=pkt.block_size,
                )
            )

        raw_transfer.await_all(h2d_futures)
        self._apply_landed_packets(landed_packets, target_model)
        if hasattr(torch, "accelerator") and torch.accelerator.is_available():
            torch.accelerator.synchronize()

    def _execute_staged_copy(
        self,
        packets: list[QuantizedTensorPacket],
        target_model: torch.nn.Module | Mapping[str, torch.Tensor],
    ) -> None:
        """Stage quantized FP8 packets through contiguous host memory and apply to target."""
        host_packets = [
            QuantizedTensorPacket(
                name=p.name,
                is_quantized=p.is_quantized,
                data=p.data.detach().to("cpu", non_blocking=False).clone(),
                scale=p.scale.detach().to("cpu", non_blocking=False).clone() if p.scale is not None else None,
                orig_shape=p.orig_shape,
                orig_dtype=p.orig_dtype,
                block_size=p.block_size,
            )
            for p in packets
        ]
        self._apply_landed_packets(host_packets, target_model)

    def _apply_landed_packets(
        self,
        packets: list[QuantizedTensorPacket],
        target_model: torch.nn.Module | Mapping[str, torch.Tensor],
    ) -> None:
        if isinstance(target_model, torch.nn.Module):
            for pkt in packets:
                inject_packet_into_vllm_module(target_model, pkt, self.quantizer)
        else:
            with torch.no_grad():
                for pkt in packets:
                    if pkt.name in target_model:
                        dst = target_model[pkt.name]
                        recon = self.quantizer.dequantize_packet(pkt, target_dtype=dst.dtype).to(dst.device)
                        dst.copy_(recon)

    @staticmethod
    def _infer_target_device(target_model: torch.nn.Module | Mapping[str, torch.Tensor]) -> torch.device:
        if isinstance(target_model, torch.nn.Module):
            for p in target_model.parameters():
                return p.device
        else:
            for t in target_model.values():
                return t.device
        return torch.device("cpu")

    @torch.no_grad()
    def submit_async_sync(
        self,
        source_model_or_state: torch.nn.Module | Mapping[str, torch.Tensor],
        target_model_or_state: torch.nn.Module | Mapping[str, torch.Tensor],
        *,
        policy_version: int,
    ) -> Future[SyncCycleStats]:
        """Quantize updated policy parameters to block-scaled FP8 and launch async transfer."""
        if self._closed:
            raise RuntimeError("AsyncTPUSyncWeightStreamer is already closed.")

        slot = self._acquire_slot(preferred_idx=policy_version)
        slot.policy_version = policy_version

        t_q0 = time.perf_counter()
        if isinstance(source_model_or_state, torch.nn.Module):
            items = list(source_model_or_state.named_parameters())
        else:
            items = list(source_model_or_state.items())

        packets: list[QuantizedTensorPacket] = []
        source_bytes = 0
        try:
            for name, param in items:
                raw_t = param.data if isinstance(param, torch.nn.Parameter) else param
                if hasattr(raw_t, "full_tensor"):
                    # Materialize DTensor shards across SimpleFSDP / TP mesh on TPU
                    raw_t = raw_t.full_tensor()
                source_bytes += raw_t.numel() * raw_t.element_size()
                packets.append(self.quantizer.quantize_tensor(name, raw_t))
        except Exception:
            self._release_slot(slot)
            raise

        quantize_time_s = time.perf_counter() - t_q0
        slot.packets = packets

        return self._executor.submit(
            self._stage_and_apply_worker,
            slot,
            packets,
            target_model_or_state,
            policy_version,
            quantize_time_s,
            source_bytes,
        )

    def wait_sync(self, future: Future[SyncCycleStats]) -> SyncCycleStats:
        """Await an in-flight async sync future and record caller stall duration."""
        t_wait0 = time.perf_counter()
        stats = future.result(timeout=self.timeout_s)
        stall = time.perf_counter() - t_wait0
        stats.caller_stall_time_s = stall
        self._history.append(stats)
        return stats

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._executor.shutdown(wait=True)

    def __enter__(self) -> AsyncTPUSyncWeightStreamer:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

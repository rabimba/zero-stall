# Phase 0 Architecture & Upstream Stack Setup (`docs/architecture_setup.md`)

This document records the exact upstream C++/Python APIs, configuration keys, environment variables, and Cloud TPU v7 (`Ironwood` / `tpu7x`) slice partitioning rules verified against the live open-source repositories:

- **`verl-project/verl`** (`examples/tpu/grpo/`, `verl/workers/engine/torchtitan/`, `verl/checkpoint_engine/`, `verl/trainer/ppo/core_algos.py`)
- **`google-pytorch/torchtitan`** (`torchtitan/experiments/tpu/recipes/ironwood/`, `SimpleFSDP`, `TPUTrainer` / `TpuGraphTrainer`, `SplashAttention`, `TpuPackedQKVLinear`)
- **`google-pytorch/torch_tpu`** (`torch.tpu`, `torch.tpu.profiler.TpuProfilerConfig`, Tier-2/Tier-3 compilation caches)
- **`vllm-project/vllm-torchtpu`** (`TpuPlatform`, `TPUModelRunner`, `OnlineFp8Policy`, `PrefixAwareScheduler`, `TPURaidenOffloadingConnector`, `tools/merge_xplane.py`)
- **`google/tpu-sync`** (`tpu_sync.api.torch.raw_transfer`, `WeightSynchronizer`, `WeightSynchronizerManager`, `KVCacheManager`, `KVCacheStore`)

---

## 1. Cloud TPU v7 (`Ironwood` / `tpu7x`) Hardware & Dynamic Partitioning Plan

### 1.1 Ironwood Hardware Architecture
- **Dual-TensorCore Topology**: Each physical Cloud TPU v7 (`tpu7x`) chip exposes **2 TensorCores** (`2` logical PJRT devices in 1-process-per-core execution). For example, a 4-chip host exposes `8` logical PJRT devices (`torch.tpu.device_count() == 8`), and a 2-host `tpu7x-2x2x2` (8-chip) slice exposes `16` logical PJRT devices.
- **Memory Hierarchy**: **192 GiB HBM3E per physical chip** (~96 GiB HBM3E per logical TensorCore) at ~7.37 TB/s HBM bandwidth, backed by host DRAM and high-speed ICI links.
- **Dense Compute**: ~4,614 TFLOPS FP8 (`float8_e4m3fn`) and ~2,307 TFLOPS BF16 per physical dual-core chip, plus dedicated **SparseCores** capable of offloading FSDP `all_gather` and `reduce_scatter` collectives asynchronously.

### 1.2 Dynamic Slice Partitioning Rules (`src/tpu_topology.py:compute_partition_plan`)

 Because `libtpu` locks `/dev/vfio/*` exclusively to one OS process per logical TPU device (`PlatformTPU.supports_colocated_worker_groups() == False`), `actor_rollout_ref.hybrid_engine=False` is mandatory on TPU. `src/tpu_topology.py` dynamically partitions the detected `tpu7x` slice as follows:

| Detected Physical `tpu7x` Chips | Logical PJRT Cores | Total HBM3E | Partitioning Mode | TorchTitan (`Actor` + `Ref` Colocated in `WorkerDict`) | Standalone `vLLM-TorchTPU` Rollout Pool | Recommended Models (Smoke / Scale) |
| :---: | :---: | :---: | :--- | :--- | :--- | :--- |
| **`>= 8` chips** (e.g., `2x2x2` multi-host) | `>= 16` | `>= 1,536 GiB` | `disaggregated_50_50_slice` | Half slice (`dp_shard = N_cores/2`, `tp=1`, `SimpleFSDP` + SparseCore offload) | Half slice (`TP=4`, `DP = (N_cores/2)/4`, `gpu_mem_util=0.65`) | `Qwen/Qwen3-4B-Base` / `Qwen/Qwen3-8B-Base` |
| **`4` chips** (`2x2x1` single-host) | `8` | `768 GiB` | `disaggregated_50_50_slice` | `4` logical cores (`2` chips, `384 GiB`, `dp_shard=4`, `SimpleFSDP` + SparseCore offload) | `4` logical cores (`2` chips, `384 GiB`, `TP=4`, `DP=1`, `gpu_mem_util=0.65`) | `Qwen/Qwen3-4B-Base` / `Qwen/Qwen3-8B-Base` |
| **`2` chips** (`1x2x1` single-host) | `4` | `384 GiB` | `split_2chip_disaggregated` | Chip 0 (`2` cores, `192 GiB`, `dp_shard=2`, port base `8471`) | Chip 1 (`2` cores, `192 GiB`, `TP=2`, `DP=1`, port base `8070`) | `Qwen/Qwen3-4B-Base` / `Qwen/Qwen3-8B-Base` |
| **`1` chip** (`1x1x1` single-host) | `2` | `192 GiB` | `single_chip_dual_core_split` | TensorCore 0 (`~96 GiB`, `dp_shard=1`, `XLA_PYTHON_CLIENT_MEM_FRACTION=0.45`, `TPU_CHIPS_PER_PROCESS_BOUNDS=1,1,1`) | TensorCore 1 (`~96 GiB`, `TP=1`, `gpu_mem_util=0.45`, `XLA_PYTHON_CLIENT_PREALLOCATE=false`) | `Qwen/Qwen3-4B-Base` |

---

## 2. Discovered Upstream APIs & Integration Contracts

### 2.1 `verl` V1 Disaggregated Separate-Async + `TorchTitanEngine`
- **Entrypoint**: `python3 -m verl.trainer.main_ppo` with `VERL_PLATFORM=tpu` and `VERL_USE_EXTERNAL_MODULES=verl_hardware_plugin,src.verl_tpu_plugin_ext`.
- **Async Overlap**: `trainer.use_v1=True`, `trainer.v1.trainer_mode=separate_async`, `trainer.v1.separate_async.parameter_sync_step=1`, `transfer_queue.enable=True`.
- **TorchTitan Engine**: `model_engine=torchtitan`, `actor_rollout_ref.actor.strategy=torchtitan`, `actor_rollout_ref.actor.torchtitan.attn_type=varlen`, `pad_to_length=True`, `pad_to_length_bucket=512`, plus `engine_kwargs.tpu.{use_simple_fsdp=True, use_splash_attention=True, fuse_qkv=True, tpu_eager_mode=DEFER_AND_FUSE, optim_impl=foreach}`.
- **Why `SimpleFSDP` Is Mandatory on TPU**: Standard PyTorch FSDP2 (`fully_shard`) places `all_gather` inside a `forward_pre_hook`, forcing TorchDynamo to graph-break on `Module.__call__` (~414 HLO modules/step). `SimpleFSDP` parametrizes `.weight` via `ReplicateComputation` so the graph break lands cleanly *between* the collective and the compiled `@spmd_safe` transformer block (~30 HLO modules/step).
- **Checkpoint Engine & Weight Sync**: `verl/workers/sharding_manager/` is superseded by `verl/checkpoint_engine/base.py` (`CheckpointEngineManager` + `@CheckpointEngineRegistry.register(...)`). On TPU, `CheckpointEngineWorker` runs as a `0-TPU` CPU sidecar inside the rollout placement group.
- **Stock Bottleneck Identified**:
  1. `TorchTitanEngine.get_per_tensor_param` all-gathers full BF16 weights (`2` bytes/param) and streams them via `tpu-sync`, while `CheckpointEngineManager.update_weights` aborts admissions and calls `release_kv_cache_replicas()` synchronously.
  2. Stock `run_qwen3_4b_torchtitan.sh` sets `enable_prefix_caching=False` (`no_enable_prefix_caching=True`), recomputing every shared system prompt, few-shot exemplar, and problem prefix from scratch on every RL step.

### 2.2 `google/tpu-sync` (`tpu_sync`) C++/Python Primitives
- **Canonical Import Root**: `tpu_sync` (`tpu-sync-torch` wheel, Python >= 3.12, loads `libpywrap_torch_tpu_common.so` with `RTLD_LOCAL | RTLD_NOW`).
- **Supported Native TorchTPU Dtypes (`torch_tpu_utils.cc:UnpackTorchTensor`)**: `torch.bfloat16`, `torch.float32`, **`torch.float8_e4m3fn`**, `torch.float8_e5m2`, `torch.int32`, `torch.int8`, `torch.uint8`.
- **Async Raw DMA (`tpu_sync.api.torch.raw_transfer`)**:
  - `RawHostBuffer(size_bytes)` (binds to PJRT `"pinned_host"` memory space or 4 KiB-aligned `PjRtClient::DmaMap` memory).
  - `transfer_d2h_async(src_tpu, dst_host_buf, unsafe_skip_buffer_lock=True) -> PjRtCopyFuture`
  - `transfer_h2d_async(src_host_buf, dst_tpu, unsafe_skip_buffer_lock=True) -> PjRtCopyFuture`
  - `set_dma_pacing(max_inflight_bytes=64<<20, chunk_bytes=16<<20)` & `await_all(futures)`.
- **Multi-Slice Resharded Weight Sync (`tpu_sync.api.torch.weight_synchronizer`)**:
  - `WeightSynchronizer(device_tensors, auto_h2d=True)` (`d2h()`, `unbind_weights()`, `bind_weights()`, `wait_for_transfer_completion(uuid)`) coordinated by `WeightSynchronizerManager` (`ReshardPlanner` + `BroadcastEngine`).
- **Multi-Tier KV Cache Orchestration**:
  - `tpu_sync.api.torch.kv_cache_manager.KVCacheManager` (`d2h`, `h2d`, `register_pools`) and `tpu_sync.api.torch.kv_cache_store.KVCacheStore` (`lookup`, `load`, `insert`, `save`, `release`, `BlockStatus.{HBM, HOST, HOST_AND_HBM, REMOTE, SHARED_STORAGE}`).

### 2.3 `vllm-project/vllm-torchtpu` Runtime & Layout Contracts
- **Plugins & Pool Layout**: `VLLM_TARGET_DEVICE=tpu`, `VLLM_PLUGINS=torchtpu,torchtpu_layers`, `VLLM_TPU_BLOCK_MAJOR_KV=1` (contiguous `[num_blocks, num_layers, ...]` single-row DMA layout), `TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL=1`.
- **FP8 & Flipped Weight Layout**: `OnlineFp8Policy` uses `torch.float8_e4m3fn` (`max=448.0`) with `128x128` block scales and transposes weights to **`[K_in, N_out]`** (`_tpu_weight_flipped = True`). `src/async_tpu_sync_weights.py:inject_packet_into_vllm_module` handles both `[K_in, N_out]` flipped BF16 parameters and native `VllmFp8LinearMethodTPU` `float8_e4m3fn` + 4D `[1, K//128, 1, N]` `gmm_v2` scale buffers.
- **Prefix Caching & Offloading**: `vllm_torchtpu.core.prefix_aware_scheduler.PrefixAwareScheduler` ranks waiting requests by `(tier, cold_tokens, arrival_time)` to minimize uncached prefill tokens, paired with `TPURaidenOffloadingConnector` (`vllm_torchtpu.offload.raiden_connector`).

### 2.4 Persistent Compilation Caches & XProf Profiling (`google-pytorch/torch_tpu`)
- **Compilation Caches**:
  - `TORCH_TPU_TIER2_COMPILATION_CACHE="zero_stall_grpo"` (`/dev/shm/torch_tpu_cache/...`)
  - `TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT="/var/tmp/zero_stall_tpu_cache/torch_tpu_tier3"` (persistent C++ PJRT executable cache across restarts)
  - `VLLM_CACHE_ROOT="/var/tmp/zero_stall_tpu_cache/vllm_cache"` (persistent `TpuCompilationHandle` AOT bundle cache)
- **XProf `.xplane.pb` Capture**:
  - `torch.tpu.profiler.TpuProfilerConfig(host_tracer_level=2, device_tracer_level=1, run_dir=...)` + `torch.profiler.profile(activities=[ProfilerActivity.CPU, ProfilerActivity.TPU], ...)`.
  - Multi-rank traces land under `<save_path>/plugins/profile/<timestamp>/rank<N>_<host>.xplane.pb` and can be merged via `python -m vllm_torchtpu.tools.merge_xplane` or analyzed directly by [`scripts/analyze_grpo_traces.py`](../scripts/analyze_grpo_traces.py).

---

## 3. Machine-Local Phase 0 Hardware Probe

<!-- BEGIN_AUTO_PHASE0_PROBE -->
### Latest Automated Hardware Probe (`Pending Cloud TPU v7 Execution`)

> [!IMPORTANT]
> Repository staged and unit-tested (`20 passed`) on local development host (`rabimba-aipc`, x86_64 Linux, 0 TPU chips attached). When picking up this repository on the Cloud TPU v7 (`tpu7x`) VM, run:
> ```bash
> python3 scripts/probe_tpu_v7.py --require-tpu-v7 --update-doc
> ```
> to automatically populate this block and `results/phase0_topology.json` with live Ironwood chip counts, ICI mesh coordinates, and VFIO verification.
<!-- END_AUTO_PHASE0_PROBE -->

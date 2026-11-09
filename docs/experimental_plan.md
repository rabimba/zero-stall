# End-to-End Experimental Plan (`docs/experimental_plan.md`)

This document specifies the complete, controlled experimental design for **"Zero-Stall Asynchronous GRPO on Cloud TPU v7 via TPU Sync DMA and Bounded-Staleness Cross-Step KV Prefix Reuse"**, engineered so an autonomous coding agent or researcher on a Cloud TPU v7 (`Ironwood` / `tpu7x`) VM can execute and validate every phase deterministically.

---

## 1. Experimental Hypotheses & Systems Bottlenecks Targeted

In disaggregated RL post-training (`TorchTitan` actor/ref trainer + `vLLM-TorchTPU` rollout generator on Cloud TPU v7), each GRPO iteration suffers from two compounding synchronization & prefill bottlenecks:

1. **Bottleneck A — Host-Locking BF16 Weight Synchronization Stalls**:
   - At every policy update (`parameter_sync_step = 1`), `TorchTitanEngine.get_per_tensor_param` materializes full `bfloat16` parameter tensors (`2` bytes/element; `~8 GiB` for 4B, `~16 GiB` for 8B) and transfers them across host/slice boundaries while `CheckpointEngineManager.update_weights` pauses vLLM admissions.
   - **Optimization A (`src/async_tpu_sync_weights.py`)**: Quantizes 2D/3D projection matrices on-the-fly into `torch.float8_e4m3fn` (`1` byte/element) with `128x128` MXU-tile `float32` scales (`~1.00024` bytes/element total — **49.99% wire & PCIe/ICI DMA reduction**), streamed asynchronously via double-buffered `google/tpu-sync` pinned-host `RawHostBuffer` / `WeightSynchronizer` (`auto_h2d=True`) overlapped with rollout batch preparation.

2. **Bottleneck B — Unconditional Cross-Step KV Prefix Cache Flushes**:
   - Upstream VeRL (`run_qwen3_4b_torchtitan.sh`) disables prefix caching (`no_enable_prefix_caching=True`) and `CheckpointEngineManager.update_weights` calls `release_kv_cache_replicas()` (`engine.reset_prefix_cache()`) after every single optimizer step — forcing full prefill recomputation of shared system rubrics, few-shot exemplars, curriculum/replay problem prompts, and low-divergence reasoning prefixes.
   - **Optimization B (`src/cross_step_kv_grpo.py`)**: Retains content-addressed prefix KV blocks in `vLLM-TorchTPU` HBM and `google/tpu-sync` `KVCacheStore` across policy step $t-1 \to t$ whenever the empirical $k_3$ policy KL shift satisfies $\widehat{D}_{\mathrm{KL}}(\pi_{\theta_t} \,\|\, \pi_{\theta_{t-1}}) \le \delta$ and cumulative drift $\sum_{s=t-\tau}^{t-1} \widehat{D}_{\mathrm{KL}}^{(s+1)} \le \delta$ ($\tau < \tau_{\max}$), paired with Truncated Importance-Sampling (TIS) correction in the dual-clip GRPO surrogate objective.

---

## 2. Controlled Hyperparameter Matrix (`configs/grpo_tpu_v7.yaml`)

All runs share **exact identical hyperparameters, data ordering, and random seed (`seed = 42`)** so every measured wall-clock and accuracy delta is strictly attributable to Optimization A and Optimization B:

| Parameter | Phase 1 Smoke (`phase1_smoke`) | Run 1 (`run1_sync_bf16_baseline`) | Run 2 (`run2_async_fp8_tpu_sync`) | Run 3 (`run3_zero_stall_kv_grpo`) |
| :--- | :---: | :---: | :---: | :---: |
| **Base Model** | `Qwen/Qwen3-4B-Base` | `Qwen/Qwen3-4B-Base` | `Qwen/Qwen3-4B-Base` | `Qwen/Qwen3-4B-Base` |
| **Dataset (Train / Val)** | GSM8K (`7,473` / `1,319`) | GSM8K (`7,473` / `1,319`) + MATH-500 | GSM8K (`7,473` / `1,319`) + MATH-500 | GSM8K (`7,473` / `1,319`) + MATH-500 |
| **Total Steps** | `10` | `60` | `60` | `60` |
| **Validation Eval Freq** | Every `5` steps (`val_before_train=True`) | Every `15` steps (`0, 15, 30, 45, 60`) | Every `15` steps (`0, 15, 30, 45, 60`) | Every `15` steps (`0, 15, 30, 45, 60`) |
| **Prompts / Step × Group $G$** | `32 × 8 = 256` completions | `32 × 8 = 256` completions | `32 × 8 = 256` completions | `32 × 8 = 256` completions |
| **Max Prompt / Response Len** | `512` / `1024` (`bucket=512`) | `512` / `1024` (`bucket=512`) | `512` / `1024` (`bucket=512`) | `512` / `1024` (`bucket=512`) |
| **Optimizer & LR** | Foreach AdamW, `lr=2e-6`, warmup `5` | Foreach AdamW, `lr=2e-6`, warmup `5` | Foreach AdamW, `lr=2e-6`, warmup `5` | Foreach AdamW, `lr=2e-6`, warmup `5` |
| **KL Penalty ($\beta_{\mathrm{KL}}$)** | `low_var_kl` ($k_3$), `0.001` | `low_var_kl` ($k_3$), `0.001` | `low_var_kl` ($k_3$), `0.001` | `low_var_kl` ($k_3$), `0.001` |
| **Weight Sync Backend** | Stock `tpu-sync` (Sync BF16) | Stock `tpu-sync` (Sync BF16) | **`tpu-sync-fp8-async`** (`128×128` FP8) | **`tpu-sync-fp8-async`** (`128×128` FP8) |
| **Cross-Step KV Prefix Reuse** | Disabled (`reset_prefix_cache` every step) | Disabled (`reset_prefix_cache` every step) | Disabled (`reset_prefix_cache` every step) | **Enabled** ($\delta=0.02$, $\tau_{\max}=4$, `PrefixAwareScheduler`) |
| **Policy Loss Mode** | `bypass_mode` (dual-clip PPO) | `bypass_mode` (dual-clip PPO) | `bypass_mode` (dual-clip PPO) | **`cross_step_kv_grpo`** (TIS $[0.2, 3.0]$ + dual-clip) |
| **XProf `.xplane.pb` Window** | Off | Steady-state steps `[5, 6, 7, 8]` | Steady-state steps `[5, 6, 7, 8]` | Steady-state steps `[5, 6, 7, 8]` |

---

## 3. Auxiliary Sensitivity & Scaling Experiments

1. **KL Staleness Threshold ($\delta$) Sensitivity Sweep**:
   - Executed automatically at the end of `scripts/run_phase3_ablation.sh` over:
     $$\delta \in \{0.0,\; 0.005,\; 0.01,\; 0.02,\; 0.05,\; 0.10\} \text{ nats}$$
     with $\tau_{\max} = 4$ and `steps = 30`.
   - Note $\delta = 0.0$ forces invalidation after every step (`0%` cross-step KV retention), empirically anchoring the Run 2 equivalence point, while large $\delta = 0.10$ tests whether TIS $[0.2, 3.0]$ preserves policy stability even under aggressive cache retention.
2. **Optional Model Scale-Up (`Qwen/Qwen3-8B-Base`)**:
   - On `>= 4` physical `tpu7x` chips (`>= 768 GiB` HBM3E), launch with `--model Qwen/Qwen3-8B-Base` to quantify how weight sync bandwidth savings scale linearly from `4B` (`~8 GiB` BF16 -> `~4 GiB` FP8) to `8B` (`~16 GiB` BF16 -> `~8 GiB` FP8).

---

## 4. Telemetry & XProf Hardware Measurement Protocol

### 4.1 Per-Step Telemetry Schema (`step_metrics.csv` & `metrics.jsonl`)
Every step logs:
- **Timing breakdown**: `rollout_prefill_time_s`, `rollout_decode_time_s`, `weight_sync_stall_time_s`, `trainer_step_time_s`, `step_wall_time_s`, `cumulative_wall_time_s`.
- **Optimization health**: `loss`, `policy_loss`, `kl_div`, `grad_norm`, `mean_reward`, `non_empty_completion_rate`, `rollout_tokens`, `rollout_tokens_per_sec`, `kv_prefix_hit_rate`, `kv_staleness_steps`, `kv_cache_invalidated`, `tis_ess_ratio`.

### 4.2 XProf `.xplane.pb` Steady-State Window (Steps 5–8)
- Steps `0–4` warm up XLA/PJRT Tier-1/2/3 compilation caches and vLLM token bucket graphs so compilation latency never contaminates steady-state hardware traces.
- Steps `5–8` activate `torch.profiler` (`ProfilerActivity.CPU` + `ProfilerActivity.TPU`) across actor and rollout workers, emitting `.xplane.pb` protobufs under `results/phase3_ablation/<run>/xprof_traces/`.
- `scripts/analyze_grpo_traces.py` decodes every `/device:TPU:*` TensorCore plane and `/host:CPU*` plane to compute:
  - **Pipeline Bubble %**: $100 \times \left(1 - \frac{|\bigcup_e [t_{\text{start}}(e), t_{\text{end}}(e)]|}{t_{\max} - t_{\min}}\right)$
  - **Weight Sync DMA Latency (ms)**
  - **Prefill & Decode Throughput (tokens/s)**
  - **TorchTitan MFU (%) & MXU Utilization (%)** relative to Ironwood `tpu7x` peak BF16 throughput (`2,307` TFLOPS/chip).

---

## 5. Gate-by-Gate Acceptance Checklist

- [x] **Local Pre-Deployment Staging Gate** (`./scripts/run_and_verify.sh --stage-verify`):
  - 20/20 unit tests passing (`test_baseline_grpo.py`, `test_weight_sync.py`, `test_cross_step_kv_loss.py`, `test_xprof_trace_analyzer.py`, `test_rewards_and_topology.py`).
  - 16/16 generated Hydra launch scripts syntax-verified across 1, 2, 4, and 8 physical `tpu7x` chips.
- [ ] **Cloud TPU v7 Gate 0** (`python3 scripts/probe_tpu_v7.py --require-tpu-v7 --update-doc`): Live `tpu7x` topology recorded in `results/phase0_topology.json` & `docs/architecture_setup.md`.
- [ ] **Cloud TPU v7 Gate 1** (`bash scripts/run_phase1_smoke.sh`): 10-step smoke run passes `tests/test_baseline_grpo.py --require-tpu-v7`.
- [ ] **Cloud TPU v7 Gate 2** (`pytest tests/test_weight_sync.py tests/test_cross_step_kv_loss.py --require-tpu-v7`): Native `tpu_sync` DMA 20-cycle test passes with zero deadlocks.
- [ ] **Cloud TPU v7 Gate 3** (`bash scripts/run_phase3_ablation.sh`): Runs 1, 2, 3 + $\delta$-sweep + `.xplane.pb` analysis + plots + paper table population completed.
- [ ] **Cloud TPU v7 Gate 4** (`./scripts/run_and_verify.sh`): Exit code `0` on real TPU v7 hardware.

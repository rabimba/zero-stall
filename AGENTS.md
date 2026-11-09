# Autonomous TPU v7 Agent Handoff Playbook (`AGENTS.md`)

> **TARGET ENVIRONMENT**: Google Cloud TPU v7 (`Ironwood` / `tpu7x`) VM  
> **CURRENT REPOSITORY STATE**: Complete codebase, upstream bindings, mathematical optimizations, `.xplane.pb` wire-format XProf trace analyzer, plotting suite, and 20-test staging verification (`./scripts/run_and_verify.sh --stage-verify` = **PASS, Exit Code 0**) are implemented and verified. **Zero hardware numbers have been mocked or fabricated.**  
> **YOUR MISSION ON THE TPU v7 VM**: Execute Phases 0 → 1 → 2 → 3 → 4 sequentially on real Cloud TPU v7 hardware, populate the telemetry artifacts (`results/`) and paper tables, run `./scripts/run_and_verify.sh` to confirm Exit Code 0, and commit/push the measured results.

---

## 1. Non-Negotiable Execution Contract (Must Enforce at Every Step)

1. **REAL HARDWARE ONLY**: Every training step, rollout generation, weight transfer, and XProf `.xplane.pb` trace MUST run natively on Cloud TPU v7 (`tpu7x`). **Never** mock gradients, rewards, DMA transfers, or timing numbers. `scripts/run_grpo_experiment.py` and `AsyncTPUSyncWeightStreamer(require_tpu=True)` explicitly refuse to run without real TPU devices.
2. **INSPECT LIVE REPOS FIRST**: Before launching Phase 1, run `python3 scripts/probe_tpu_v7.py` and inspect the exact installed commits/versions of `verl`, `verl_hardware_plugin`, `torchtitan`, `vllm_torchtpu`, `torch_tpu`, `ray`, and `tpu_sync` on the TPU VM. If any upstream signature on your VM differs from the Oct/Nov 2026 pins documented in [`docs/architecture_setup.md`](docs/architecture_setup.md), align [`src/verl_tpu_plugin_ext.py`](src/verl_tpu_plugin_ext.py) and [`scripts/run_grpo_experiment.py`](scripts/run_grpo_experiment.py) to match the live VM packages before running.
3. **PROCESS & RAY HYGIENE**: Always run `bash scripts/tpu_hygiene.sh` (`ray stop --force` + `/tmp/libtpu_lockfile*` cleanup + `/dev/vfio/*` FD verification + persistent compilation caches at `${ZERO_STALL_CACHE_ROOT:-/var/tmp/zero_stall_tpu_cache}`) before **every** experiment run.
4. **NUMERICAL & CONVERGENCE RIGOR**: Do not advance past any phase gate until its `pytest` assertions and convergence checks pass completely.

---

## 2. Sequential Execution Runbook for the TPU v7 Agent

Execute each gate in strict order from the repository root:

### Gate 0 — Topology Discovery, Environment Bootstrap & Dataset Prep
```bash
# 1. Reset any lingering Ray processes or VFIO locks and set persistent cache dirs
bash scripts/tpu_hygiene.sh

# 2. Inspect installed TPU stack, install any missing repos into third_party/,
#    download & verify deterministic GSM8K + MATH-500 Parquet splits, and update docs/architecture_setup.md
bash scripts/bootstrap_tpu_vm.sh

# 3. Enforce strict Phase 0 TPU v7 hardware & VFIO cleanliness gate (must exit 0)
python3 scripts/probe_tpu_v7.py --require-tpu-v7 --update-doc
```
**Gate 0 Pass Criteria**:
- `results/phase0_topology.json` exists with `physical_tpu_chips >= 1`, `vfio_report.is_clean == true`, and all 6 core modules (`torch_tpu`, `verl`, `torchtitan`, `vllm_torchtpu`, `ray`, `tpu_sync`) `installed == true`.
- `data/manifest.json` and `data/gsm8k/{train,test}.parquet` + `data/math500/test.parquet` exist and pass SHA-256 verification.
- [`docs/architecture_setup.md`](docs/architecture_setup.md) contains the updated `<!-- BEGIN_AUTO_PHASE0_PROBE -->` hardware snapshot.

---

### Gate 1 — 10-Step Baseline VeRL + TorchTitan + vLLM-TorchTPU Smoke Run
```bash
bash scripts/run_phase1_smoke.sh
```
**What this executes**:
1. Runs `scripts/tpu_hygiene.sh`.
2. Launches 10 steps of stock synchronous BF16 VeRL TPU GRPO (`profile=phase1_smoke` in [`configs/grpo_tpu_v7.yaml`](configs/grpo_tpu_v7.yaml)) pairing `TorchTitan` (`SimpleFSDP` + `SplashAttention` + `fuse_qkv` + `torch.compile(backend="tpu")`) with standalone `vLLM-TorchTPU` rollout workers (`VLLM_PLUGINS=torchtpu,torchtpu_layers`).
3. Runs `pytest tests/test_baseline_grpo.py --run-dir results/phase1_smoke --require-tpu-v7`.

**Gate 1 Pass Criteria**:
- Finite `loss`, `policy_loss`, `kl_div`, and `grad_norm` across all 10 steps in `results/phase1_smoke/step_metrics.csv`.
- `non_empty_completion_rate > 0` scored deterministically by `src/rewards_and_data.py:compute_score`.
- All 4 per-step timing fields populated (`rollout_prefill_time_s`, `rollout_decode_time_s`, `weight_sync_stall_time_s`, `trainer_step_time_s`).

---

### Gate 2 — Hardware Verification of Optimization A (Async FP8 TPU Sync) & Optimization B (Cross-Step KV Reuse)
```bash
bash scripts/tpu_hygiene.sh
PYTHONPATH=. pytest tests/test_weight_sync.py tests/test_cross_step_kv_loss.py --require-tpu-v7
```
**Gate 2 Pass Criteria**:
- `test_weight_sync.py` runs natively on `torch.device("tpu")` + `tpu_sync.api.torch.raw_transfer` (`RawHostBuffer` + `transfer_d2h_async` / `transfer_h2d_async`), confirming:
  - Block-scaled `float8_e4m3fn` reconstruction cosine similarity `> 0.9999` vs exact FP8 cast and `> 0.999` vs BF16.
  - `~2.0x` DMA payload compression.
  - **Zero deadlocks** across 20 rapid consecutive async sync cycles overlapped with rollout tensor prep.
- `test_cross_step_kv_loss.py` confirms:
  - **(a)** Bit-exact value & gradient match (`atol=1e-7`) to standard VeRL GRPO when `staleness_steps = 0`.
  - **(b)** Strictly finite, bounded gradients across synthetic KL shifts up to `50.0` nats.
  - **(c)** Automatic invalidation of vLLM prefix cache + `tpu_sync` `KVCacheStore` the instant single-step or cumulative KL exceeds $\delta$.

---

### Gate 3 — Controlled 3×60-Step Convergence Ablation, $\delta$-Sweep & XProf `.xplane.pb` Profiling
```bash
bash scripts/run_phase3_ablation.sh
```
**What this executes** (all from identical `seed=42`, with full `tpu_hygiene.sh` reset before every run):
1. **Run 1 (`run1_sync_bf16_baseline`)**: Stock Synchronous BF16 VeRL TPU GRPO Baseline (60 steps, validation every 15 steps, XProf `.xplane.pb` capture on steps 5–8).
2. **Run 2 (`run2_async_fp8_tpu_sync`)**: VeRL + Async BF16-to-FP8 Block-Scaled Weight Streaming via `tpu-sync-fp8-async` (60 steps, validation every 15 steps, XProf on steps 5–8).
3. **Run 3 (`run3_zero_stall_kv_grpo`)**: VeRL + Async FP8 TPU Sync + Bounded-Staleness Cross-Step KV Prefix Reuse (`delta=0.02`, `max_staleness=4`, `PrefixAwareScheduler`, `cross_step_kv_grpo` TIS loss; 60 steps, validation every 15 steps, XProf on steps 5–8).
4. **KL-Threshold Sensitivity Sweep**: Sweeps $\delta \in \{0.0, 0.005, 0.01, 0.02, 0.05, 0.10\}$ into `results/kl_sweep/delta_*`.
5. **XProf Trace Analysis & Plotting**:
   - `scripts/analyze_grpo_traces.py` parses `.xplane.pb` traces across steady-state steps 5–8 to compute `pipeline_bubble_pct`, `weight_sync_latency_ms`, `prefill_tokens_per_sec`, `decode_tokens_per_sec`, `torchtitan_mfu_pct`, and `mxu_utilization_pct` -> `results/phase3_ablation/xprof_summary.json`.
   - `scripts/generate_plots.py` generates `results/plots/{reward_and_val_vs_wallclock.png, step_breakdown_stacked_bar.png, kl_threshold_sensitivity_sweep.png}`.
   - `scripts/populate_paper_tables.py` populates the real measured tables inside [`docs/research_paper_draft_notes.md`](docs/research_paper_draft_notes.md) and [`README.md`](README.md).

**Gate 3 Pass Criteria**:
- Runs 2 and 3 match or exceed Run 1 held-out GSM8K pass@1 validation accuracy in substantially lower total wall-clock time, with reduced XProf `pipeline_bubble_pct` and `weight_sync_latency_ms`.

---

### Gate 4 — Master End-to-End Verification & Final Commit
```bash
./scripts/run_and_verify.sh
```
**Gate 4 Pass Criteria**:
- `./scripts/run_and_verify.sh` (without `--stage-verify`) completes with **Exit Code 0**, confirming clean VFIO locks, passing hardware `pytest`, passing cached TPU verification run, and non-empty validated CSVs, JSONs, XProf summaries, and PNG plots.
- Commit and push `results/phase0_topology.json`, `results/phase1_smoke/*.csv`, `results/phase3_ablation/**/*.csv`, `results/phase3_ablation/**/xprof_summary.json`, `results/plots/*.png`, `docs/architecture_setup.md`, `docs/research_paper_draft_notes.md`, and `README.md` (note `.gitignore` excludes multi-GB raw `.xplane.pb` and `.parquet` binaries while keeping all CSV/JSON/PNG summaries).

---

## 3. Critical Ironwood (`tpu7x`) & TorchTPU Gotchas (Already Handled in `src/`, Know Before Debugging)

1. **Dual-TensorCore per Physical Ironwood Chip**:
   - Each physical Cloud TPU v7 (`tpu7x`) chip has **192 GiB HBM3E** and exposes **2 TensorCores** (`torch.tpu.device_count() == 2 * physical_chips` in 1-process-per-core mode, ~96 GiB HBM3E per logical device). `src/tpu_topology.py` automatically accounts for both physical chip count and logical PJRT device count.
2. **Never Initialize `torch_tpu` in the Ray Orchestrator Driver Process**:
   - `import torch` autoloads `torch_tpu` (`TORCH_DEVICE_BACKEND_AUTOLOAD`). Calling `torch.tpu.is_available()` or creating a `"tpu"` tensor in the parent launcher locks `/dev/vfio/*` and causes spawned Ray workers to fail PJRT init. `src/tpu_topology.py` always runs TPU device queries inside a short-lived child subprocess (`_probe_tpu_subprocess()`).
3. **SparseCore Collective Offload Must Be Trainer-Only**:
   - `VERL_TPU_TRAINER_XLA_FLAGS` (`--xla_tpu_enable_sparse_core_collective_aggregator=true`, etc.) boosts `TorchTitan` `SimpleFSDP` MFU on Ironwood, but must **not** be set globally in `XLA_FLAGS` because `vLLM-TorchTPU` rollout kernels expect standard non-SparseCore collective scheduling. `PlatformTPU` routes `VERL_TPU_TRAINER_*` exclusively to trainer/ref workers.
4. **`vllm-torchtpu` `[K_in, N_out]` Flipped Weight Layout (`_tpu_weight_flipped`)**:
   - After initial model load, `VllmUnquantizedLinearMethod` and `VllmFp8LinearMethodTPU` transpose linear weights from `[N_out, K_in]` to `[K_in, N_out]` and set `layer._tpu_weight_flipped = True`. `inject_packet_into_vllm_module` in [`src/async_tpu_sync_weights.py`](src/async_tpu_sync_weights.py) inspects `_tpu_weight_flipped` and transposes incoming `[N_out, K_in]` trainer weights automatically.
5. **`TorchTPU` Missing `aten::std.correction` Kernel**:
   - Calling `tensor.std(unbiased=True)` on `device="tpu"` can hit an unimplemented `aten::std.correction` op. `compute_grpo_group_advantages` in [`src/cross_step_kv_grpo.py`](src/cross_step_kv_grpo.py) computes Bessel-corrected variance explicitly via elementary ops `((x - mean) ** 2).sum() / (n - 1)`.
6. **Persistent Multi-Tier Compilation Caches**:
   - `torch_tpu` uses `TORCH_TPU_TIER2_COMPILATION_CACHE` (`/dev/shm/torch_tpu_cache/...`) and `TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT` (persistent disk PJRT executable cache), while `vllm-torchtpu` uses `VLLM_CACHE_ROOT` (`TpuCompilationHandle` cache). `scripts/tpu_hygiene.sh` and `src/tpu_topology.py` configure all three on local disk so runs 2 & 3 skip cold XLA compilation.

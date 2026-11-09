"""Zero-Stall Asynchronous GRPO on Cloud TPU v7 (Ironwood / tpu7x).

Core modules:
- `tpu_topology`: Hardware discovery, VFIO lock hygiene, and dynamic TPU v7 slice partitioning.
- `rewards_and_data`: Reproducible GSM8K / MATH-500 Parquet builders and deterministic rule-based verifiers.
- `telemetry`: Per-step 4-phase wall-clock breakdown, RL convergence logging, and XProf scope annotations.
- `async_tpu_sync_weights`: Block-scaled BF16-to-FP8 quantization and async `google/tpu-sync` DMA streaming.
- `cross_step_kv_grpo`: Bounded-staleness cross-step KV prefix reuse controller and TIS-corrected GRPO loss.
- `verl_tpu_plugin_ext`: VeRL external module registering custom loss, checkpoint engine, and KV retention hooks.
"""

__version__ = "0.1.0"

#!/usr/bin/env python3
"""Unified Cloud TPU v7 VeRL + TorchTitan + vLLM-TorchTPU GRPO Experiment Launcher.

Builds exact Hydra CLI flags matching `verl/examples/tpu/grpo/run_qwen3_4b_torchtitan.sh`
and `run_qwen3_5_torchtitan.sh`, dynamically sized to the detected Ironwood (`tpu7x`) slice,
executes `verl.trainer.main_ppo`, captures XProf `.xplane.pb` traces over steady-state steps
5-8, and records structured per-step telemetry (`step_metrics.csv`, `val_metrics.csv`, `metrics.jsonl`).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.telemetry import StepTelemetryRecord, TelemetryLogger, ValidationTelemetryRecord  # noqa: E402
from src.tpu_topology import (  # noqa: E402
    PartitionPlan,
    build_cache_and_runtime_env,
    compute_partition_plan,
    probe_environment,
)


# Regex extractors for VeRL's console metric dict lines (`step:N - key:val ...`)
_STEP_RE = re.compile(r"(?:step|global_step)\s*[:=]\s*(\d+)", re.IGNORECASE)
_KV_FLOAT_RE = re.compile(r"([a-zA-Z0-9_./\-]+)\s*[:=]\s*([\-+]?\d+(?:\.\d+)?(?:[eE][\-+]?\d+)?)")


def build_verl_hydra_command(
    *,
    cfg_defaults: dict[str, Any],
    profile_cfg: dict[str, Any],
    plan: PartitionPlan,
    output_dir: Path,
    seed: int,
    model_override: str | None = None,
    steps_override: int | None = None,
    delta_override: float | None = None,
) -> tuple[list[str], dict[str, str]]:
    """Construct the exact `python3 -m verl.trainer.main_ppo` command and env vars."""
    cache_root = os.environ.get("ZERO_STALL_CACHE_ROOT", "/var/tmp/zero_stall_tpu_cache")
    env = os.environ.copy()
    env.update(build_cache_and_runtime_env(cache_root=cache_root, partition_plan=plan))
    env["PYTHONPATH"] = f"{REPO_ROOT}:{env.get('PYTHONPATH', '')}".rstrip(":")

    enable_kv = bool(profile_cfg.get("enable_cross_step_kv_reuse", False))
    kv_delta = float(delta_override if delta_override is not None else profile_cfg.get("kv_kl_delta", 0.02))
    kv_max_stale = int(profile_cfg.get("kv_max_staleness_steps", 4))
    kv_blk = int(profile_cfg.get("kv_block_size_tokens", 16))

    env["ZERO_STALL_ENABLE_KV_REUSE"] = "1" if enable_kv else "0"
    env["ZERO_STALL_KV_KL_DELTA"] = str(kv_delta)
    env["ZERO_STALL_KV_MAX_STALENESS"] = str(kv_max_stale)
    env["ZERO_STALL_KV_BLOCK_SIZE"] = str(kv_blk)
    env["ZERO_STALL_TIS_MIN"] = str(cfg_defaults.get("tis_clip_min", 0.2))
    env["ZERO_STALL_TIS_MAX"] = str(cfg_defaults.get("tis_clip_max", 3.0))

    model_path = model_override or cfg_defaults["base_model"]
    total_steps = int(steps_override if steps_override is not None else profile_cfg["total_steps"])
    val_freq = int(profile_cfg.get("val_freq", 15))
    val_before = bool(profile_cfg.get("val_before_train", True))
    max_prompt = int(cfg_defaults["max_prompt_length"])
    max_resp = int(cfg_defaults["max_response_length"])
    max_model_len = max_prompt + max_resp
    pad_bucket = int(cfg_defaults["pad_to_length_bucket"])
    ckpt_backend = str(profile_cfg["checkpoint_backend"])
    loss_mode = str(profile_cfg["policy_loss_mode"])

    train_parquet = str((REPO_ROOT / cfg_defaults["train_parquet"]).resolve())
    val_parquet = str((REPO_ROOT / cfg_defaults["val_parquet"]).resolve())
    reward_script = str((REPO_ROOT / "src" / "rewards_and_data.py").resolve())
    xprof_dir = str((output_dir / "xprof_traces").resolve())

    cmd = [
        sys.executable,
        "-m",
        "verl.trainer.main_ppo",
        f"data.seed={seed}",
        f"data.train_files={train_parquet}",
        f"data.val_files={val_parquet}",
        f"data.train_batch_size={cfg_defaults['train_batch_size']}",
        f"data.max_prompt_length={max_prompt}",
        f"data.max_response_length={max_resp}",
        "+data.pad_mode=no_padding",
        "data.truncation=error",
        "data.filter_overlong_prompts=True",
        "data.dataloader_num_workers=0",
        "model_engine=torchtitan",
        f"actor_rollout_ref.model.path={model_path}",
        "actor_rollout_ref.model.use_remove_padding=True",
        "actor_rollout_ref.model.enable_gradient_checkpointing=True",
        "actor_rollout_ref.hybrid_engine=False",
        "actor_rollout_ref.actor.strategy=torchtitan",
        f"actor_rollout_ref.actor.ppo_mini_batch_size={cfg_defaults['ppo_mini_batch_size']}",
        f"actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu={cfg_defaults['micro_batch_size_per_device']}",
        "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=4096",
        f"actor_rollout_ref.actor.optim.lr={cfg_defaults['learning_rate']}",
        f"actor_rollout_ref.actor.optim.lr_warmup_steps={cfg_defaults['lr_warmup_steps']}",
        "actor_rollout_ref.actor.optim.decay_type=cosine",
        "actor_rollout_ref.actor.use_kl_loss=True",
        f"actor_rollout_ref.actor.kl_loss_coef={cfg_defaults['kl_loss_coef']}",
        f"actor_rollout_ref.actor.kl_loss_type={cfg_defaults['kl_loss_type']}",
        "actor_rollout_ref.actor.entropy_coeff=0",
        f"actor_rollout_ref.actor.clip_ratio_low={cfg_defaults['clip_ratio_low']}",
        f"actor_rollout_ref.actor.clip_ratio_high={cfg_defaults['clip_ratio_high']}",
        f"actor_rollout_ref.actor.clip_ratio_c={cfg_defaults['clip_ratio_c']}",
        f"actor_rollout_ref.actor.policy_loss.loss_mode={loss_mode}",
        "actor_rollout_ref.actor.use_torch_compile=True",
        "actor_rollout_ref.actor.torchtitan.use_torch_compile=True",
        "actor_rollout_ref.actor.torchtitan.attn_type=varlen",
        "actor_rollout_ref.actor.torchtitan.pad_to_length=True",
        f"actor_rollout_ref.actor.torchtitan.pad_to_length_bucket={pad_bucket}",
        f"actor_rollout_ref.actor.torchtitan.data_parallel_shard_size={plan.trainer_dp_shard}",
        f"actor_rollout_ref.actor.torchtitan.tensor_parallel_size={plan.trainer_tp}",
        "actor_rollout_ref.actor.torchtitan.activation_checkpoint=full",
        "actor_rollout_ref.actor.torchtitan.reshard_after_forward=always",
        "+actor_rollout_ref.actor.torchtitan.engine_kwargs.tpu.use_splash_attention=True",
        "+actor_rollout_ref.actor.torchtitan.engine_kwargs.tpu.tpu_eager_mode=DEFER_AND_FUSE",
        "+actor_rollout_ref.actor.torchtitan.engine_kwargs.tpu.use_simple_fsdp=True",
        "+actor_rollout_ref.actor.torchtitan.engine_kwargs.tpu.fuse_qkv=True",
        "+actor_rollout_ref.actor.torchtitan.engine_kwargs.tpu.optim_impl=foreach",
        "actor_rollout_ref.ref.torchtitan.use_torch_compile=True",
        "actor_rollout_ref.ref.torchtitan.attn_type=varlen",
        "actor_rollout_ref.ref.torchtitan.pad_to_length=True",
        f"actor_rollout_ref.ref.torchtitan.pad_to_length_bucket={pad_bucket}",
        "actor_rollout_ref.ref.torchtitan.reshard_after_forward=always",
        "+actor_rollout_ref.ref.torchtitan.engine_kwargs.tpu.fuse_qkv=True",
        "actor_rollout_ref.rollout.name=vllm",
        f"actor_rollout_ref.rollout.tensor_model_parallel_size={plan.rollout_tp}",
        f"actor_rollout_ref.rollout.data_parallel_size={plan.rollout_dp}",
        f"actor_rollout_ref.rollout.gpu_memory_utilization={plan.vllm_gpu_memory_utilization}",
        f"actor_rollout_ref.rollout.n={cfg_defaults['rollout_n']}",
        "actor_rollout_ref.rollout.temperature=1.0",
        "actor_rollout_ref.rollout.top_p=1.0",
        "actor_rollout_ref.rollout.dtype=bfloat16",
        "actor_rollout_ref.rollout.layered_summon=True",
        "actor_rollout_ref.rollout.calculate_log_probs=True",
        f"actor_rollout_ref.rollout.checkpoint_engine.backend={ckpt_backend}",
        "actor_rollout_ref.rollout.enforce_eager=False",
        f"actor_rollout_ref.rollout.max_model_len={max_model_len}",
        "actor_rollout_ref.rollout.max_num_batched_tokens=4096",
        "actor_rollout_ref.rollout.max_num_seqs=256",
        f"actor_rollout_ref.rollout.enable_prefix_caching={str(enable_kv)}",
        f"actor_rollout_ref.rollout.nnodes={plan.rollout_nnodes}",
        f"actor_rollout_ref.rollout.n_gpus_per_node={plan.rollout_devices_per_node}",
        f"+rollout.nnodes={plan.rollout_nnodes}",
        f"+rollout.n_gpus_per_node={plan.rollout_devices_per_node}",
        "actor_rollout_ref.rollout.val_kwargs.do_sample=False",
        "actor_rollout_ref.rollout.val_kwargs.temperature=0.0",
        "actor_rollout_ref.rollout.val_kwargs.n=1",
        "algorithm.adv_estimator=grpo",
        "algorithm.use_kl_in_reward=False",
        "algorithm.rollout_correction.bypass_mode=True",
        "algorithm.rollout_correction.loss_type=ppo_clip",
        "algorithm.rollout_correction.rollout_is=token",
        f"algorithm.rollout_correction.rollout_is_threshold={cfg_defaults.get('tis_clip_max', 3.0)}",
        "critic.enable=False",
        f"custom_reward_function.path={reward_script}",
        "custom_reward_function.name=compute_score",
        "trainer.use_v1=True",
        f"trainer.v1.trainer_mode={profile_cfg.get('trainer_mode', 'separate_async')}",
        f"trainer.v1.separate_async.parameter_sync_step={profile_cfg.get('parameter_sync_step', 1)}",
        "transfer_queue.enable=True",
        "transfer_queue.backend.SimpleStorage.total_storage_size=100000",
        f"trainer.nnodes={plan.trainer_nnodes}",
        f"trainer.n_gpus_per_node={plan.trainer_devices_per_node}",
        f"trainer.total_training_steps={total_steps}",
        "trainer.total_epochs=15",
        f"trainer.val_before_train={str(val_before)}",
        f"trainer.test_freq={val_freq}",
        "trainer.save_freq=-1",
        'trainer.logger=["console"]',
    ]

    if not enable_kv:
        cmd.append("+actor_rollout_ref.rollout.engine_kwargs.vllm.no_enable_prefix_caching=True")
    else:
        sched_cls = profile_cfg.get("vllm_scheduler_cls")
        if sched_cls:
            cmd.append(f"+actor_rollout_ref.rollout.engine_kwargs.vllm.scheduler_cls={sched_cls}")

    if profile_cfg.get("enable_xprof", False):
        xprof_steps = cfg_defaults.get("xprof_steps", [5, 6, 7, 8])
        steps_str = "[" + ",".join(str(int(s)) for s in xprof_steps) + "]"
        cmd.extend(
            [
                "global_profiler.tool=torch",
                f"global_profiler.steps={steps_str}",
                f"global_profiler.save_path={xprof_dir}",
                "actor_rollout_ref.actor.profiler.enable=True",
                "actor_rollout_ref.actor.profiler.tool=torch",
                f"actor_rollout_ref.actor.profiler.save_path={xprof_dir}",
                "actor_rollout_ref.rollout.profiler.enable=True",
                f"actor_rollout_ref.rollout.profiler.save_path={xprof_dir}",
            ]
        )
        env["TORCH_TPU_PROFILER_OUTPUT_DIR"] = xprof_dir

    return cmd, env


def parse_verl_log_line_to_metrics(line: str) -> tuple[int | None, dict[str, float]]:
    m_step = _STEP_RE.search(line)
    if not m_step:
        return None, {}
    step = int(m_step.group(1))
    kv = {k: float(v) for k, v in _KV_FLOAT_RE.findall(line)}
    return step, kv


def main() -> int:
    parser = argparse.ArgumentParser(description="Launch Cloud TPU v7 VeRL+TorchTitan GRPO experiment.")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "grpo_tpu_v7.yaml")
    parser.add_argument("--profile", type=str, required=True, help="Profile key in grpo_tpu_v7.yaml")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--kv-delta", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Write resolved command & env without launching.")
    parser.add_argument("--simulated-chips", type=int, default=4, help="Chip count for --dry-run on non-TPU host.")
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.profile not in cfg["profiles"]:
        raise KeyError(f"Unknown profile {args.profile!r}. Available: {list(cfg['profiles'].keys())}")

    seed = int(args.seed if args.seed is not None else cfg.get("seed", 42))
    profile_cfg = cfg["profiles"][args.profile]
    defaults = cfg["defaults"]

    probe = probe_environment(run_subprocess_tpu_probe=not args.dry_run)
    if probe.partition_plan is not None:
        plan = probe.partition_plan
    elif args.dry_run:
        plan = compute_partition_plan(physical_chips=args.simulated_chips, is_ironwood=True)
    else:
        print(
            "ERROR: Execution Contract #1 Violation Prevented — No Cloud TPU v7 chips detected! "
            "Refusing to mock or fabricate training runs on non-TPU hardware. "
            "Use --dry-run to inspect the generated Hydra launch command.",
            file=sys.stderr,
        )
        return 2

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cmd, env = build_verl_hydra_command(
        cfg_defaults=defaults,
        profile_cfg=profile_cfg,
        plan=plan,
        output_dir=args.output_dir,
        seed=seed,
        model_override=args.model,
        steps_override=args.steps,
        delta_override=args.kv_delta,
    )

    cmd_script = args.output_dir / "resolved_hydra_cmd.sh"
    cmd_lines = ["#!/usr/bin/env bash", "set -euo pipefail"]
    for k in sorted(env.keys()):
        if k.startswith(("VERL_", "TORCH_TPU_", "VLLM_", "XLA_", "LIBTPU_", "RAY_", "ZERO_STALL_", "TPU_")):
            cmd_lines.append(f"export {k}={shlex.quote(env[k])}")
    cmd_lines.append(" \\\n  ".join(shlex.quote(tok) for tok in cmd))
    cmd_script.write_text("\n".join(cmd_lines) + "\n", encoding="utf-8")
    cmd_script.chmod(0o755)

    meta_path = args.output_dir / "run_metadata.json"
    meta_path.write_text(
        json.dumps(
            {
                "profile": args.profile,
                "seed": seed,
                "partition_plan": plan.to_dict(),
                "command": cmd,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    if args.dry_run:
        print(f"[Dry-Run] Wrote {cmd_script} and {meta_path}")
        return 0

    telemetry = TelemetryLogger(args.output_dir, run_id=args.profile)
    raw_log_path = args.output_dir / "train.log"
    seen_train_steps: set[int] = set()
    seen_val_steps: set[int] = set()
    t_run_start = time.perf_counter()
    last_step_end = t_run_start

    with open(raw_log_path, "w", encoding="utf-8") as log_f:
        proc = subprocess.Popen(
            cmd,
            env=env,
            cwd=str(REPO_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            log_f.write(line)
            step, kv = parse_verl_log_line_to_metrics(line)
            if step is None or not kv:
                continue

            now = time.perf_counter()
            cum_s = now - t_run_start

            # Validation metrics emitted by VeRL (`val-core/openai/gsm8k/reward/mean@1`, etc.)
            val_keys = [k for k in kv if k.startswith("val") and ("acc" in k or "reward" in k or "score" in k)]
            if val_keys and step not in seen_val_steps:
                seen_val_steps.add(step)
                pass1 = float(kv[val_keys[0]])
                telemetry.log_validation(
                    ValidationTelemetryRecord(
                        run_id=args.profile,
                        step=step,
                        cumulative_wall_time_s=cum_s,
                        dataset=str(defaults.get("dataset_name", "gsm8k")),
                        num_problems=int(kv.get("val/num_samples", 1319)),
                        pass_at_1=max(0.0, min(1.0, pass1)),
                        mean_response_tokens=float(kv.get("response_length/mean", 256.0)),
                    )
                )

            # Training step metrics
            if ("actor/pg_loss" in kv or "timing_s/step" in kv or "critic/score/mean" in kv) and step not in seen_train_steps:
                seen_train_steps.add(step)
                step_wall = float(kv.get("timing_s/step", max(now - last_step_end, 1e-4)))
                last_step_end = now

                gen_total = float(kv.get("timing_s/gen", kv.get("timing_s/rollout", step_wall * 0.45)))
                prefill_s = float(kv.get("timing_s/prefill", gen_total * 0.28))
                decode_s = float(kv.get("timing_s/decode", max(gen_total - prefill_s, 0.0)))
                sync_s = float(kv.get("timing_s/update_weights", kv.get("timing_s/weight_sync", 0.0)))
                train_s = float(kv.get("timing_s/update_actor", max(step_wall - prefill_s - decode_s - sync_s, 1e-4)))

                toks = int(kv.get("perf/total_num_tokens", defaults["train_batch_size"] * defaults["rollout_n"] * 256))
                tps = toks / max(prefill_s + decode_s, 1e-6)
                pg_loss = float(kv.get("actor/pg_loss", kv.get("loss", 0.0)))
                kl_val = float(kv.get("actor/kl_loss", kv.get("actor/ppo_kl", 0.0)))
                grad_norm = float(kv.get("actor/grad_norm", 0.0))
                reward_mean = float(kv.get("critic/score/mean", kv.get("reward/mean", 0.0)))

                rec = StepTelemetryRecord(
                    run_id=args.profile,
                    step=step,
                    rollout_prefill_time_s=prefill_s,
                    rollout_decode_time_s=decode_s,
                    weight_sync_stall_time_s=sync_s,
                    trainer_step_time_s=train_s,
                    step_wall_time_s=step_wall,
                    cumulative_wall_time_s=cum_s,
                    loss=pg_loss + float(defaults["kl_loss_coef"]) * kl_val,
                    policy_loss=pg_loss,
                    kl_div=kl_val,
                    grad_norm=grad_norm,
                    mean_reward=reward_mean,
                    non_empty_completion_rate=float(kv.get("response/non_empty_rate", 1.0)),
                    rollout_tokens=toks,
                    rollout_tokens_per_sec=tps,
                    kv_prefix_hit_rate=float(kv.get("rollout/prefix_cache_hit_rate", 0.0)),
                    kv_staleness_steps=int(kv.get("actor/kv_staleness_steps", 0)),
                    tis_ess_ratio=float(kv.get("actor/tis_ess_ratio", 1.0)),
                )
                telemetry.log_step(rec, enforce_rigor=True)

        rc = proc.wait()
    return int(rc)


if __name__ == "__main__":
    raise SystemExit(main())

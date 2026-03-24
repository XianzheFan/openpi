"""Co-evolution loop: RLT policy + DreamDojo world model.

Iteratively improves both the RLT policy and DreamDojo world model:

  Round 0 (bootstrap):
    - Train RL token encoder-decoder on real demonstrations (Phase 1)

  Round k >= 1:
    Step A: Generate synthetic trajectories using current RLT policy + current DreamDojo
    Step B: (External) Fine-tune DreamDojo on new + old trajectory data
    Step C: Online RL for RLT in (updated) DreamDojo imagination
    Step D: (Optional) Re-train RL token on combined real + synthetic data

The loop converges when the RLT policy generates trajectories that are
realistic enough for DreamDojo to learn from, and DreamDojo is accurate
enough for RLT to learn useful behaviors in imagination.

Usage:
  python scripts/coevolve_rlt_dreamdojo.py \
      --num_iterations 5 \
      --task_suite_name libero_10 \
      --base_checkpoint gs://openpi-assets/checkpoints/pi05_libero \
      --dd_checkpoint_dir checkpoints/dreamdojo_libero \
      --output_base checkpoints/coevo_libero

Prerequisites:
  1. Trained RL token checkpoint (Phase 1) or pass --run_phase1 to train it first
  2. DreamDojo server binary / training script (external dependency)
  3. Gemini API access for VLM reward (or use --imagination_reward heuristic)
"""

import dataclasses
import json
import logging
import pathlib
import subprocess
import sys
import time

import tyro

logger = logging.getLogger("coevolve")


@dataclasses.dataclass
class Args:
    """Co-evolution loop configuration."""

    # Number of co-evolution iterations.
    num_iterations: int = 5

    # Base VLA config and checkpoint.
    base_config: str = "pi05_libero"
    base_checkpoint: str = "gs://openpi-assets/checkpoints/pi05_libero"

    # LIBERO task suite.
    task_suite_name: str = "libero_10"

    # Output base directory. Each iteration creates a subdirectory.
    output_base: str = "checkpoints/coevo_{task}"

    # --- Phase 1 (bootstrap) ---
    run_phase1: bool = True
    phase1_demo_dir: str = "data/libero/perturbed_trajs_{task}/perturbed_success"
    phase1_steps: int = 5000
    phase1_finetune_vla: bool = True

    # --- Step A: Synthetic data generation ---
    # Port for the RLT policy server during data generation.
    vla_port: int = 8000
    # DreamDojo server port(s) for VLAW rollouts.
    dd_base_port: int = 8020
    # Number of parallel rollouts per initial state.
    num_rollouts_per_init: int = 8
    # Number of initial states per task.
    num_inits_per_task: int = 50
    # Max time steps per rollout.
    max_time_steps: int = 520

    # --- Step B: DreamDojo fine-tuning ---
    # Path to DreamDojo training script (external).
    dd_train_script: str = "examples/train_dreamdojo.py"
    # DreamDojo checkpoint directory (updated each iteration).
    dd_checkpoint_dir: str = "checkpoints/dreamdojo_{task}"
    # DreamDojo training config (passed to dd_train_script).
    dd_experiment: str = "dreamdojo_2b_480_640_libero"

    # --- Step C: Online RL in imagination ---
    rl_num_episodes: int = 500
    rl_warmup_episodes: int = 20
    imagination_reward: str = "vlm"
    vlm_success_threshold: float = 0.80

    # --- Step D: Optional RL token re-training ---
    retrain_rl_token: bool = False
    retrain_steps: int = 3000

    seed: int = 42

    def __post_init__(self):
        task = self.task_suite_name
        for f in dataclasses.fields(self):
            val = getattr(self, f.name)
            if isinstance(val, str) and "{task}" in val:
                object.__setattr__(self, f.name, val.replace("{task}", task))


def _run(cmd: str, description: str, check: bool = True) -> int:
    """Run a shell command, logging its output."""
    logger.info(f"[RUN] {description}")
    logger.info(f"  $ {cmd}")
    result = subprocess.run(cmd, shell=True, text=True)
    if check and result.returncode != 0:
        logger.error(f"Command failed with return code {result.returncode}")
        raise RuntimeError(f"Failed: {description}")
    return result.returncode


def run_phase1(args: Args) -> str:
    """Phase 1: Train RL token encoder-decoder on demonstrations."""
    output_dir = f"{args.output_base}/phase1_rl_token"
    cmd = (
        f"python scripts/train_rlt.py"
        f" --phase rl_token"
        f" --base_config {args.base_config}"
        f" --base_checkpoint {args.base_checkpoint}"
        f" --demo_data_dir {args.phase1_demo_dir}"
        f" --output_dir {output_dir}"
        f" --num_steps {args.phase1_steps}"
        f" --task_suite_name {args.task_suite_name}"
        f" --seed {args.seed}"
    )
    if args.phase1_finetune_vla:
        cmd += " --finetune_vla"
    _run(cmd, "Phase 1: RL Token Training")
    return f"{output_dir}/best"


def run_step_a(args: Args, iteration: int, rlt_checkpoint: str) -> str:
    """Step A: Generate synthetic trajectories with current RLT + DreamDojo.

    Note: This assumes serve_rlt_policy.py is already running on vla_port,
    and DreamDojo server(s) are running on dd_base_port.
    """
    output_dir = f"{args.output_base}/iter{iteration}/synthetic_data"
    cmd = (
        f"python third_party/libero/generate_vlaw_synthetic_data.py"
        f" --task_suite_name {args.task_suite_name}"
        f" --num_rollouts_per_init {args.num_rollouts_per_init}"
        f" --num_inits_per_task {args.num_inits_per_task}"
        f" --max_time_steps {args.max_time_steps}"
        f" --vla_port {args.vla_port}"
        f" --dd_base_port {args.dd_base_port}"
        f" --output_dir {output_dir}"
        f" --seed {args.seed + iteration * 1000}"
    )
    _run(cmd, f"Step A (iter {iteration}): Generate synthetic data")
    return output_dir


def run_step_b(args: Args, iteration: int, synthetic_data_dir: str) -> str:
    """Step B: Fine-tune DreamDojo on new synthetic data.

    Returns the path to the updated DreamDojo checkpoint.
    """
    new_dd_ckpt = f"{args.dd_checkpoint_dir}/iter{iteration}"
    prev_dd_ckpt = (
        f"{args.dd_checkpoint_dir}/iter{iteration - 1}"
        if iteration > 1
        else args.dd_checkpoint_dir
    )
    cmd = (
        f"python {args.dd_train_script}"
        f" --data_dir {synthetic_data_dir}"
        f" --checkpoint {prev_dd_ckpt}"
        f" --experiment {args.dd_experiment}"
        f" --output_dir {new_dd_ckpt}"
    )
    rc = _run(cmd, f"Step B (iter {iteration}): Fine-tune DreamDojo", check=False)
    if rc != 0:
        logger.warning(
            f"DreamDojo fine-tuning failed (rc={rc}). "
            f"Continuing with previous checkpoint: {prev_dd_ckpt}"
        )
        return prev_dd_ckpt
    return new_dd_ckpt


def run_step_c(args: Args, iteration: int, rlt_checkpoint: str) -> str:
    """Step C: Online RL in DreamDojo imagination.

    Note: Assumes the RLT policy server is running on vla_port,
    and DreamDojo server is running on dd_base_port.
    """
    output_dir = f"{args.output_base}/iter{iteration}/online_rl"
    cmd = (
        f"python scripts/train_rlt.py"
        f" --phase online_rl"
        f" --use_imagination"
        f" --base_config {args.base_config}"
        f" --rlt_checkpoint {rlt_checkpoint}"
        f" --task_suite_name {args.task_suite_name}"
        f" --dd_base_port {args.dd_base_port}"
        f" --num_episodes {args.rl_num_episodes}"
        f" --warmup_episodes {args.rl_warmup_episodes}"
        f" --imagination_reward {args.imagination_reward}"
        f" --vlm_success_threshold {args.vlm_success_threshold}"
        f" --coevo_iteration {iteration}"
        f" --output_dir {output_dir}"
        f" --host 0.0.0.0 --port {args.vla_port}"
        f" --seed {args.seed + iteration * 2000}"
    )
    _run(cmd, f"Step C (iter {iteration}): Online RL in imagination")
    return f"{output_dir}/final"


def run_step_d(args: Args, iteration: int, synthetic_data_dir: str) -> str:
    """Step D (optional): Re-train RL token on combined real + synthetic data."""
    output_dir = f"{args.output_base}/iter{iteration}/rl_token_retrain"
    cmd = (
        f"python scripts/train_rlt.py"
        f" --phase rl_token"
        f" --base_config {args.base_config}"
        f" --base_checkpoint {args.base_checkpoint}"
        f" --demo_data_dir {synthetic_data_dir}"
        f" --output_dir {output_dir}"
        f" --num_steps {args.retrain_steps}"
        f" --task_suite_name {args.task_suite_name}"
        f" --finetune_vla"
        f" --seed {args.seed + iteration * 3000}"
    )
    _run(cmd, f"Step D (iter {iteration}): Re-train RL token")
    return f"{output_dir}/best"


def main(args: Args):
    output_base = pathlib.Path(args.output_base)
    output_base.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(output_base / "coevo_config.json", "w") as f:
        json.dump(dataclasses.asdict(args), f, indent=2)

    logger.info("=" * 70)
    logger.info("RLT + DreamDojo Co-Evolution")
    logger.info("=" * 70)
    logger.info(f"Iterations:      {args.num_iterations}")
    logger.info(f"Task suite:      {args.task_suite_name}")
    logger.info(f"Output:          {args.output_base}")
    logger.info(f"Reward source:   {args.imagination_reward}")
    logger.info("")

    # --- Round 0: Bootstrap ---
    if args.run_phase1:
        logger.info(">>> Round 0: Bootstrap (Phase 1 RL Token Training)")
        rlt_checkpoint = run_phase1(args)
    else:
        rlt_checkpoint = f"{args.output_base}/phase1_rl_token/best"
        logger.info(f">>> Skipping Phase 1. Using existing: {rlt_checkpoint}")

    # --- Co-evolution loop ---
    dd_checkpoint = args.dd_checkpoint_dir

    for iteration in range(1, args.num_iterations + 1):
        logger.info("")
        logger.info("=" * 70)
        logger.info(f">>> Co-Evolution Iteration {iteration}/{args.num_iterations}")
        logger.info("=" * 70)
        iter_dir = output_base / f"iter{iteration}"
        iter_dir.mkdir(parents=True, exist_ok=True)

        logger.info("")
        logger.info(
            f"NOTE: Before running Step A, ensure the following servers are running:\n"
            f"  1. RLT policy server:\n"
            f"     python scripts/serve_rlt_policy.py \\\n"
            f"       --base_checkpoint {args.base_checkpoint} \\\n"
            f"       --rlt_checkpoint {rlt_checkpoint} \\\n"
            f"       --port {args.vla_port}\n"
            f"  2. DreamDojo server(s):\n"
            f"     python examples/dreamdojo_server.py \\\n"
            f"       --checkpoint {dd_checkpoint} \\\n"
            f"       --port {args.dd_base_port}\n"
        )
        logger.info("Waiting 5 seconds for servers to be ready...")
        time.sleep(5)

        # Step A: Generate synthetic data
        logger.info(f"--- Step A: Generate synthetic trajectories ---")
        synthetic_data_dir = run_step_a(args, iteration, rlt_checkpoint)

        # Step B: Fine-tune DreamDojo
        logger.info(f"--- Step B: Fine-tune DreamDojo ---")
        dd_checkpoint = run_step_b(args, iteration, synthetic_data_dir)

        # Step C: Online RL in imagination
        logger.info(f"--- Step C: Online RL in DreamDojo imagination ---")
        rlt_checkpoint = run_step_c(args, iteration, rlt_checkpoint)

        # Step D: Optional RL token re-training
        if args.retrain_rl_token:
            logger.info(f"--- Step D: Re-train RL token ---")
            rlt_checkpoint = run_step_d(args, iteration, synthetic_data_dir)

        # Save iteration state
        state = {
            "iteration": iteration,
            "rlt_checkpoint": rlt_checkpoint,
            "dd_checkpoint": dd_checkpoint,
            "synthetic_data_dir": synthetic_data_dir,
        }
        with open(iter_dir / "iteration_state.json", "w") as f:
            json.dump(state, f, indent=2)

        logger.info(f"Iteration {iteration} complete.")
        logger.info(f"  RLT checkpoint: {rlt_checkpoint}")
        logger.info(f"  DD checkpoint:  {dd_checkpoint}")

    logger.info("")
    logger.info("=" * 70)
    logger.info("Co-evolution complete!")
    logger.info(f"  Final RLT:       {rlt_checkpoint}")
    logger.info(f"  Final DreamDojo: {dd_checkpoint}")
    logger.info("=" * 70)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    main(tyro.cli(Args))

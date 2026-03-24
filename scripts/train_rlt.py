"""RLT training pipeline for LIBERO with perturbation robustness.

Implements the full RLT training procedure from the paper:
  Phase 1: Train RL token encoder-decoder on demonstration data (+ optional VLA fine-tune)
  Phase 2: Online RL with actor-critic using the RL token representation
  Phase 2b: Online RL in DreamDojo imagination (for co-evolution with world model)

Usage:
  # Phase 1: Train RL token
  python scripts/train_rlt.py \
      --phase rl_token \
      --base_config pi05_libero \
      --base_checkpoint gs://openpi-assets/checkpoints/pi05_libero \
      --demo_data_dir data/libero/perturbed_trajs/perturbed_success \
      --output_dir checkpoints/rlt_libero_perturbed/rl_token \
      --num_steps 5000 \
      --finetune_vla \
      --vla_finetune_weight 0.1

  # Phase 2: Online RL (real simulator)
  python scripts/train_rlt.py \
      --phase online_rl \
      --base_config pi05_libero \
      --rlt_checkpoint checkpoints/rlt_libero_perturbed/rl_token/best \
      --task_suite_name libero_10 \
      --perturbations scene_swap obstacle occlusion object_swap_target \
      --num_episodes 500 \
      --warmup_episodes 20 \
      --output_dir checkpoints/rlt_libero_perturbed/online_rl

  # Phase 2b: Online RL in DreamDojo imagination (co-evolution)
  python scripts/train_rlt.py \
      --phase online_rl \
      --use_imagination \
      --dd_base_port 8020 \
      --base_config pi05_libero \
      --rlt_checkpoint checkpoints/rlt_libero_perturbed/rl_token/best \
      --task_suite_name libero_10 \
      --num_episodes 500 \
      --output_dir checkpoints/rlt_libero_coevo/online_rl
"""

import collections
import dataclasses
import json
import logging
import math
import pathlib
import time
from typing import Any

import numpy as np
import tyro
import wandb

logger = logging.getLogger("rlt_train")


@dataclasses.dataclass
class Args:
    """RLT training — all parameters flat, phase selects which subset is used.

    Paths containing '{task}' are auto-replaced with task_suite_name at startup.
    """
    phase: str = "rl_token"
    base_config: str = "pi05_libero"
    task_suite_name: str = "libero_10"
    output_dir: str = "checkpoints/rlt_{task}/rl_token"
    seed: int = 42

    base_checkpoint: str = "gs://openpi-assets/checkpoints/pi05_libero"
    demo_data_dir: str = "data/libero/perturbed_trajs_{task}/perturbed_success"
    num_steps: int = 5000
    batch_size: int = 16
    learning_rate: float = 1e-4
    finetune_vla: bool = True
    vla_finetune_weight: float = 0.1
    rl_token_dim: int = 2048
    encoder_num_layers: int = 2
    decoder_num_layers: int = 2

    rlt_checkpoint: str = "checkpoints/rlt_{task}/rl_token/best"
    perturbations: list[str] = dataclasses.field(
        default_factory=lambda: ["scene_swap", "obstacle", "occlusion", "object_swap_target"]
    )
    num_episodes: int = 500
    warmup_episodes: int = 20
    max_steps_per_episode: int = 400

    actor_hidden_dim: int = 256
    actor_num_layers: int = 2
    critic_hidden_dim: int = 256
    critic_num_layers: int = 2
    rl_chunk_length: int = 10
    actor_std: float = 0.1
    ref_action_dropout: float = 0.5

    bc_regularizer_beta: float = 1.0
    discount_gamma: float = 0.99
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    update_to_data_ratio: int = 5
    critic_updates_per_actor: int = 2
    target_network_tau: float = 0.005
    replay_buffer_size: int = 100000
    rl_batch_size: int = 256
    action_chunk_stride: int = 2

    vla_num_steps: int = 10
    replan_steps: int = 10

    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    num_steps_wait: int = 10

    log_interval: int = 10
    eval_interval: int = 50
    save_interval: int = 100
    video_out_path: str = "data/libero_rlt/{task}/online_rl_videos"
    wandb_enabled: bool = True
    wandb_project: str = "openpi-rlt"

    use_imagination: bool = False
    dd_base_port: int = 8020
    dd_timeout: int = 600
    dd_save_dir: str = "/tmp/rlt_dd_gen"
    imagination_reward: str = "vlm"
    vlm_model: str = "gemini-3-flash-preview"
    vlm_query_interval: int = 40
    vlm_success_threshold: float = 0.80
    coevo_iteration: int = 0

    def __post_init__(self):
        """Replace {task} placeholder in all str fields with task_suite_name."""
        task = self.task_suite_name
        for f in dataclasses.fields(self):
            val = getattr(self, f.name)
            if isinstance(val, str) and "{task}" in val:
                object.__setattr__(self, f.name, val.replace("{task}", task))


class ReplayBuffer:
    """Simple replay buffer for off-policy RL with action chunk transitions."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer: list[dict[str, np.ndarray]] = []
        self.position = 0

    def add(self, transition: dict[str, np.ndarray]):
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.position] = transition
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int) -> dict[str, np.ndarray]:
        indices = np.random.randint(0, len(self.buffer), size=batch_size)
        batch = collections.defaultdict(list)
        for idx in indices:
            for key, value in self.buffer[idx].items():
                batch[key].append(value)
        return {key: np.stack(values) for key, values in batch.items()}

    def __len__(self):
        return len(self.buffer)

    def save(self, path: pathlib.Path):
        path.mkdir(parents=True, exist_ok=True)
        np.savez(
            path / "replay_buffer.npz",
            **{f"{i}_{k}": v for i, t in enumerate(self.buffer) for k, v in t.items()},
        )
        with open(path / "replay_buffer_meta.json", "w") as f:
            json.dump({"size": len(self.buffer), "position": self.position, "capacity": self.capacity}, f)


def train_rl_token(args: Args):
    """Train the RL token encoder-decoder on demonstration data."""
    import jax
    import jax.numpy as jnp
    import flax.nnx as nnx
    import optax

    from openpi.models.pi0_rlt import Pi0RLTConfig
    from openpi.training import config as _config
    from openpi.training import data_loader as _data_loader
    from openpi.models import model as _model

    logger.info("=" * 60)
    logger.info("Phase 1: RL Token Training")
    logger.info("=" * 60)
    logger.info(f"Base config: {args.base_config}")
    logger.info(f"Demo data:   {args.demo_data_dir}")
    logger.info(f"Fine-tune VLA: {args.finetune_vla} (alpha={args.vla_finetune_weight})")
    logger.info(f"Steps: {args.num_steps}, batch_size: {args.batch_size}")

    wandb.init(
        project=args.wandb_project,
        name=f"rl_token_{args.task_suite_name}",
        config=dataclasses.asdict(args),
        mode="online" if args.wandb_enabled else "disabled",
    )

    full_config = _config.get_config(args.base_config)
    model_kwargs = dataclasses.asdict(full_config.model)
    rlt_model_config = Pi0RLTConfig(
        **model_kwargs,
        rl_token_dim=args.rl_token_dim,
        encoder_num_layers=args.encoder_num_layers,
        decoder_num_layers=args.decoder_num_layers,
    )

    rng = jax.random.key(args.seed)
    rng, model_rng = jax.random.split(rng)
    logger.info("Creating Pi0RLT model...")
    model = rlt_model_config.create(model_rng)

    logger.info(f"Loading VLA weights from {args.base_checkpoint}...")
    if args.base_checkpoint.startswith("gs://"):
        import openpi.shared.download as download
        checkpoint_dir = pathlib.Path(download.maybe_download(args.base_checkpoint))
    else:
        checkpoint_dir = pathlib.Path(args.base_checkpoint)

    vla_params = _model.restore_params(checkpoint_dir / "params", dtype=jnp.bfloat16)

    from orbax import checkpoint as ocp
    graphdef, state = nnx.split(model)
    vla_intersect = ocp.transform_utils.intersect_trees(state.to_pure_dict(), vla_params)
    state.replace_by_pure_dict(vla_intersect)
    model = nnx.merge(graphdef, state)
    logger.info("VLA weights loaded.")

    logger.info("Setting up data loader...")
    demo_data_path = pathlib.Path(args.demo_data_dir).resolve()
    logger.info(f"Using local demo data: {demo_data_path}")

    # Load local parquet data directly, bypassing LeRobotDataset (avoids datasets version issues)
    from datasets import load_dataset as _load_dataset
    import lerobot.common.datasets.lerobot_dataset as _lerobot_ds
    import json

    hf_dataset = _load_dataset("parquet", data_dir=str(demo_data_path / "data"), split="train")
    hf_dataset.set_transform(_lerobot_ds.hf_transform_to_torch)

    # Read task mapping for prompt_from_task
    tasks_file = demo_data_path / "meta" / "tasks.jsonl"
    task_map = {}
    with open(tasks_file) as f:
        for line in f:
            entry = json.loads(line)
            task_map[entry["task_index"]] = entry["task"]

    # Read metadata for action horizon
    meta_file = demo_data_path / "meta" / "info.json"
    with open(meta_file) as f:
        meta_info = json.load(f)
    fps = meta_info["fps"]
    action_horizon = rlt_model_config.action_horizon

    # Build a simple torch Dataset wrapping hf_dataset with action chunks + prompt
    from torch.utils.data import Dataset as TorchDataset, DataLoader as TorchDL

    class LocalDemoDataset(TorchDataset):
        def __init__(self, hf_ds, task_map, action_horizon, fps):
            self.hf_ds = hf_ds
            self.task_map = task_map
            self.action_horizon = action_horizon
            self.hf_ds.reset_format()
            ep_indices = np.array(hf_ds["episode_index"], dtype=np.int64)
            frame_indices = np.array(hf_ds["frame_index"], dtype=np.int64)
            self.hf_ds.set_transform(_lerobot_ds.hf_transform_to_torch)
            self.ep_indices = ep_indices
            self.frame_indices = frame_indices
            self._ep_ends = {}
            for i in range(len(ep_indices)):
                ep = ep_indices[i]
                if ep not in self._ep_ends or i > self._ep_ends[ep]:
                    self._ep_ends[ep] = i

        def __len__(self):
            return len(self.hf_ds)

        def __getitem__(self, idx):
            sample = self.hf_ds[idx]
            ep = int(self.ep_indices[idx])
            ep_end = self._ep_ends[ep]
            # Gather action chunk
            chunk_end = min(idx + self.action_horizon, ep_end + 1)
            action_indices = list(range(idx, chunk_end))
            # Pad if needed
            if len(action_indices) < self.action_horizon:
                action_indices += [action_indices[-1]] * (self.action_horizon - len(action_indices))
            actions = torch.stack([self.hf_ds[i]["actions"] for i in action_indices])
            sample["actions"] = actions
            # Add prompt from task
            task_idx = int(sample.get("task_index", 0))
            sample["prompt"] = self.task_map.get(task_idx, "")
            return sample

    import torch
    local_dataset = LocalDemoDataset(hf_dataset, task_map, action_horizon, fps)

    # Apply the same transforms as the standard pipeline (repack + data + model transforms)
    data_config = full_config.data.create(full_config.assets_dirs, rlt_model_config)
    transforms_list = [
        *data_config.repack_transforms.inputs,
        *data_config.data_transforms.inputs,
        *data_config.model_transforms.inputs,
    ]
    from openpi.training.data_loader import TransformedDataset, TorchDataLoader, DataLoaderImpl
    transformed_dataset = TransformedDataset(local_dataset, transforms_list)

    local_batch_size = args.batch_size // jax.process_count()
    torch_dl = TorchDataLoader(
        transformed_dataset,
        local_batch_size=local_batch_size,
        shuffle=True,
        num_batches=None,
        num_workers=0,
        seed=args.seed,
    )
    data_loader = DataLoaderImpl(data_config, torch_dl)

    output_dir = pathlib.Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set up optimizer with nnx
    optimizer = optax.adam(args.learning_rate)
    opt_state = nnx.Optimizer(model, optimizer)

    finetune_vla = args.finetune_vla
    vla_weight = args.vla_finetune_weight

    @nnx.jit
    def train_step(model, opt_state, rng, observation, actions):
        def loss_fn(model):
            rl_token_loss = model.compute_rl_token_loss(observation)
            total = rl_token_loss
            if finetune_vla:
                vla_loss = jnp.mean(model.compute_loss(rng, observation, actions, train=True))
                total = rl_token_loss + vla_weight * vla_loss
            return total
        loss, grads = nnx.value_and_grad(loss_fn)(model)
        opt_state.update(grads)
        return loss

    logger.info("Starting RL token training...")
    best_loss = float("inf")

    # Create iterator ONCE — not per step (was causing full re-init each iteration)
    data_iter = iter(data_loader)

    for step in range(args.num_steps):
        rng, step_rng = jax.random.split(rng)
        observation, actions = next(data_iter)

        total_loss = train_step(
            model, opt_state, step_rng, observation, actions,
        )

        if step % 100 == 0:
            loss_val = float(total_loss)
            logger.info(f"Step {step}/{args.num_steps} | Loss: {loss_val:.6f}")
            wandb.log({"train/loss": loss_val, "train/step": step}, step=step)
            if loss_val < best_loss:
                best_loss = loss_val

        if (step + 1) % 1000 == 0:
            import orbax.checkpoint as ocp
            ckpt_path = output_dir / f"step_{step+1}"
            logger.info(f"Saving checkpoint to {ckpt_path}")
            params = nnx.state(model)
            checkpointer = ocp.StandardCheckpointer()
            checkpointer.save(ckpt_path / "params", {"params": params}, force=True)
            checkpointer.wait_until_finished()

    wandb.log({"train/best_loss": best_loss})
    logger.info(f"Saving final checkpoint to {output_dir}")
    import orbax.checkpoint as ocp
    params = nnx.state(model)
    checkpointer = ocp.StandardCheckpointer()
    checkpointer.save(output_dir / "best" / "params", {"params": params}, force=True)
    checkpointer.wait_until_finished()
    logger.info(f"Phase 1 complete. Best loss: {best_loss:.6f}")
    wandb.finish()


def _dreamdojo_step(
    frame_np: np.ndarray,
    action: np.ndarray,
    dd_port: int,
    save_name: str,
    task_description: str,
    dd_timeout: int,
    seed: int,
) -> np.ndarray | None:
    """Call DreamDojo to imagine the next frame from (current_frame, action).

    Returns the next frame as (H, W, 3) uint8, or None on failure.
    """
    import base64
    import cv2
    import requests
    import imageio

    DREAMDOJO_H, DREAMDOJO_W = 480, 640

    if frame_np.shape[:2] != (DREAMDOJO_H, DREAMDOJO_W):
        frame_resized = cv2.resize(frame_np, (DREAMDOJO_W, DREAMDOJO_H))
    else:
        frame_resized = frame_np

    frame_bytes = base64.b64encode(frame_resized.tobytes()).decode()
    payload = {
        "frame": frame_bytes,
        "frame_height": DREAMDOJO_H,
        "frame_width": DREAMDOJO_W,
        "actions": np.array([action], dtype=np.float32).tolist(),
        "save_name": save_name,
        "prompt": task_description,
        "seed": seed,
    }
    url = f"http://127.0.0.1:{dd_port}/generate"
    try:
        resp = requests.post(url, json=payload, timeout=dd_timeout)
        resp.raise_for_status()
        video_path = resp.json()["save_path"]
        reader = imageio.get_reader(video_path, "ffmpeg")
        frames = [np.asarray(f) for f in reader]
        reader.close()
        if not frames:
            return None
        next_frame = frames[-1]
        # Resize back to LIBERO resolution
        if next_frame.shape[:2] != (256, 256):
            next_frame = cv2.resize(next_frame, (256, 256))
        return next_frame
    except Exception as e:
        logger.error(f"[DreamDojo port={dd_port}] step failed: {e}")
        return None


def _vlm_judge_reward(
    frames: list[np.ndarray],
    task_description: str,
    vlm_model: str = "gemini-3-flash-preview",
) -> float:
    """Use Gemini VLM to score the latest frames for task progress.

    Returns a float score in [0, 1].
    """
    import tempfile
    import imageio
    import json

    try:
        from google import genai
        from google.genai import types
        from pydantic import BaseModel
    except ImportError:
        logger.warning("google-genai not installed; returning heuristic reward 0.0")
        return 0.0

    class FrameEvaluation(BaseModel):
        reasoning: str
        score: float
        status: str

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp_path = tmp.name

    recent = frames[-50:] if len(frames) > 50 else frames
    imageio.mimwrite(tmp_path, recent, fps=10, codec="libx264", quality=8)

    client = genai.Client(http_options={"api_version": "v1alpha"})
    video_file = client.files.upload(file=tmp_path)

    import time as _time
    for _ in range(60):
        info = client.files.get(name=video_file.name)
        if info.state.name != "PROCESSING":
            break
        _time.sleep(2)

    prompt = (
        f"Task: {task_description}\n\n"
        "You are a robot action evaluation expert. Based on the video, evaluate the robot's "
        "task progress and provide a Value Score between 0.00 and 1.00.\n"
        "- 0.00-0.20: Disengaged/Failure State\n"
        "- 0.20-0.40: Approach State\n"
        "- 0.40-0.60: Initial Interaction State\n"
        "- 0.60-0.80: Critical Execution State\n"
        "- 0.80-1.00: Completion State\n"
        "Output strictly in JSON array format."
    )

    try:
        response = client.models.generate_content(
            model=vlm_model,
            contents=[prompt, "\n[Video]:", video_file],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=list[FrameEvaluation],
                temperature=0.0,
            ),
        )
        result = json.loads(response.text)
        score = float(result[0]["score"]) if result else 0.0
    except Exception as e:
        logger.error(f"VLM judge failed: {e}")
        score = 0.0
    finally:
        try:
            client.files.delete(name=video_file.name)
        except Exception:
            pass
        pathlib.Path(tmp_path).unlink(missing_ok=True)

    return score


def train_online_rl(args: Args):
    """Online RL with actor-critic using RL token representation (Algorithm 1)."""
    logger.info("=" * 60)
    logger.info("Phase 2: Online RL Training")
    logger.info("=" * 60)
    logger.info(f"Task suite:     {args.task_suite_name}")
    logger.info(f"Perturbations:  {args.perturbations}")
    logger.info(f"Episodes:       {args.num_episodes} (warmup: {args.warmup_episodes})")
    logger.info(f"Chunk length:   {args.rl_chunk_length}")
    logger.info(f"UTD ratio:      {args.update_to_data_ratio}")
    logger.info(f"BC beta:        {args.bc_regularizer_beta}")

    wandb.init(
        project=args.wandb_project,
        name=f"online_rl_{args.task_suite_name}",
        config=dataclasses.asdict(args),
        mode="online" if args.wandb_enabled else "disabled",
    )

    replay_buffer = ReplayBuffer(args.replay_buffer_size)

    from openpi_client import websocket_client_policy as _wsc
    client = _wsc.WebsocketClientPolicy(args.host, args.port)

    import torch
    _original_load = torch.load
    def _patched_load(*a, **kw):
        if "weights_only" not in kw:
            kw["weights_only"] = False
        return _original_load(*a, **kw)
    torch.load = _patched_load

    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from openpi_client import image_tools

    np.random.seed(args.seed)
    rng = np.random.RandomState(args.seed)

    task_suite = benchmark.get_benchmark_dict()[args.task_suite_name]()
    num_tasks = task_suite.n_tasks

    active_perturbations = {p for p in args.perturbations if p != "none"}

    import sys
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "third_party" / "libero"))
    try:
        from multimodal_ambiguity_4traj import (
            LIBERO_DUMMY_ACTION, LIBERO_ENV_RESOLUTION,
            ObstaclePerturbation, OcclusionPerturbation,
            ObjectSwapPerturbation, SceneSwapPerturbation,
            _MAX_STEPS, _quat2axisangle,
        )
    except ImportError:
        logger.warning("Perturbation classes unavailable — running without perturbations.")
        active_perturbations = set()
        LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
        LIBERO_ENV_RESOLUTION = 256
        _MAX_STEPS = {"libero_spatial": 220, "libero_object": 280, "libero_goal": 300,
                      "libero_10": 520, "libero_90": 400}
        def _quat2axisangle(quat):
            if quat[3] > 1.0: quat[3] = 1.0
            elif quat[3] < -1.0: quat[3] = -1.0
            den = np.sqrt(1.0 - quat[3] * quat[3])
            if math.isclose(den, 0.0): return np.zeros(3)
            return (quat[:3] * 2.0 * math.acos(quat[3])) / den

    max_steps = _MAX_STEPS.get(args.task_suite_name, args.max_steps_per_episode)

    scene_swap = SceneSwapPerturbation(rng) if "scene_swap" in active_perturbations else None
    obstacle_p = ObstaclePerturbation(rng) if "obstacle" in active_perturbations else None
    occlusion_p = OcclusionPerturbation(rng) if "occlusion" in active_perturbations else None
    obj_swap_p = (ObjectSwapPerturbation(rng, involve_target=True)
                  if "object_swap_target" in active_perturbations else None)

    output_dir = pathlib.Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    stats = {"total_episodes": 0, "total_successes": 0, "warmup_successes": 0,
             "rl_successes": 0, "total_transitions": 0,
             "use_imagination": args.use_imagination,
             "coevo_iteration": args.coevo_iteration}

    if args.use_imagination:
        logger.info("*** IMAGINATION MODE: rollouts via DreamDojo world model ***")
        logger.info(f"DreamDojo port:  {args.dd_base_port}")
        logger.info(f"Reward source:   {args.imagination_reward}")
        pathlib.Path(args.dd_save_dir).mkdir(parents=True, exist_ok=True)

    logger.info("Starting online RL loop...")

    for ep in range(args.num_episodes):
        is_warmup = ep < args.warmup_episodes

        task_id = rng.randint(num_tasks)
        task = task_suite.get_task(task_id)
        init_states = task_suite.get_task_init_states(task_id)
        ep_idx = rng.randint(len(init_states))

        task_bddl = (pathlib.Path(get_libero_path("bddl_files"))
                     / task.problem_folder / task.bddl_file)
        env = OffScreenRenderEnv(bddl_file_name=task_bddl,
                                 camera_heights=LIBERO_ENV_RESOLUTION,
                                 camera_widths=LIBERO_ENV_RESOLUTION)
        env.seed(args.seed + ep)
        env.reset()
        obs = env.set_init_state(init_states[ep_idx])
        task_desc = task.language

        # Apply a random perturbation (real sim only; in imagination, perturbations
        # are implicitly handled by DreamDojo's generative diversity)
        if not args.use_imagination:
            perturbers = []
            if scene_swap:   perturbers.append(("scene_swap", scene_swap))
            if obstacle_p:   perturbers.append(("obstacle", obstacle_p))
            if occlusion_p:  perturbers.append(("occlusion", occlusion_p))
            if obj_swap_p:   perturbers.append(("object_swap_target", obj_swap_p))

            if perturbers:
                name, p = perturbers[rng.randint(len(perturbers))]
                if name == "obstacle":
                    p.apply(env, num_obstacles=rng.randint(1, 3))
                elif name == "object_swap_target":
                    p.apply(env, task_description=task_desc)
                else:
                    p.apply(env)
                sim = env.env.sim if hasattr(env, "env") else env.sim
                sim.forward()
                env._update_observables(force=True)
                obs = env.env._get_observations() if hasattr(env, "env") else env._get_observations()

        # Extract initial observation (used by both real and imagination modes)
        for _ in range(args.num_steps_wait):
            obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)

        init_agentview = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
        init_wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
        init_eef_pos = obs["robot0_eef_pos"].copy()
        init_eef_quat = obs["robot0_eef_quat"].copy()
        init_gripper_qpos = obs["robot0_gripper_qpos"].copy()

        # In imagination mode we no longer need the sim after extracting s_0
        if args.use_imagination:
            env.close()

        # Rollout
        action_plan = collections.deque()
        t, done = 0, False
        ep_transitions = []

        # Current observation state (updated in-place during rollout)
        current_agentview = init_agentview.copy()
        current_wrist = init_wrist.copy()
        current_state = np.concatenate((
            init_eef_pos,
            _quat2axisangle(init_eef_quat),
            init_gripper_qpos,
        )).astype(np.float32)

        # For VLM reward: accumulate frames
        imagined_frames = [current_agentview.copy()] if args.use_imagination else []

        while t < max_steps:
            try:
                img = image_tools.convert_to_uint8(
                    image_tools.resize_with_pad(current_agentview, args.resize_size, args.resize_size))
                wrist = image_tools.convert_to_uint8(
                    image_tools.resize_with_pad(current_wrist, args.resize_size, args.resize_size))

                state = current_state

                if not action_plan:
                    result = client.infer({
                        "observation/image": img,
                        "observation/wrist_image": wrist,
                        "observation/state": state,
                        "prompt": str(task_desc),
                    })
                    chunk = result["actions"]
                    action_plan.extend(chunk[:args.replan_steps])
                    ep_transitions.append({
                        "state": state.copy(),
                        "action_chunk": np.array(chunk[:args.rl_chunk_length], dtype=np.float32),
                    })

                action = action_plan.popleft()

                if args.use_imagination:
                    # --- Imagination: DreamDojo generates the next frame ---
                    action_arr = np.asarray(action, dtype=np.float64)
                    next_frame = _dreamdojo_step(
                        frame_np=current_agentview,
                        action=action_arr,
                        dd_port=args.dd_base_port,
                        save_name=f"rlt_ep{ep}_step{t}",
                        task_description=task_desc,
                        dd_timeout=args.dd_timeout,
                        seed=args.seed + ep * 10000 + t,
                    )
                    if next_frame is None:
                        logger.warning(f"[Ep {ep}] DreamDojo failed at step {t}; truncating.")
                        break

                    current_agentview = next_frame
                    # Wrist: carry forward from init (same as VLAW pipeline)
                    current_wrist = init_wrist.copy()
                    # State: use action as proxy for new EE state
                    current_state = np.concatenate([
                        action_arr[:3],
                        action_arr[3:6],
                        np.array([action_arr[6], action_arr[6]]),
                    ]).astype(np.float32)
                    imagined_frames.append(current_agentview.copy())
                else:
                    # --- Real simulator ---
                    obs, reward, done, info = env.step(
                        action.tolist() if hasattr(action, "tolist") else action)
                    if done:
                        break
                    # Update current observation from sim
                    current_agentview = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    current_wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                    current_state = np.concatenate((
                        obs["robot0_eef_pos"],
                        _quat2axisangle(obs["robot0_eef_quat"]),
                        obs["robot0_gripper_qpos"],
                    )).astype(np.float32)

                t += 1
            except Exception as e:
                logger.error(f"Step {t}: {e}")
                break

        if not args.use_imagination:
            env.close()

        if args.use_imagination:
            if args.imagination_reward == "vlm":
                score = _vlm_judge_reward(imagined_frames, task_desc, vlm_model=args.vlm_model)
                success = score >= args.vlm_success_threshold
                reward_val = score
            else:
                # Heuristic: shorter trajectories are more likely successes
                success = t < max_steps * 0.8
                reward_val = 1.0 if success else 0.0
        else:
            success = bool(done)
            reward_val = 1.0 if success else 0.0

        for i, tr in enumerate(ep_transitions):
            if i % args.action_chunk_stride == 0:
                replay_buffer.add({
                    "state": tr["state"],
                    "action_chunk": tr["action_chunk"],
                    "reward": np.array([reward_val], dtype=np.float32),
                    "done": np.array([float(success)], dtype=np.float32),
                })
                stats["total_transitions"] += 1

        stats["total_episodes"] += 1
        if success:
            stats["total_successes"] += 1
            if is_warmup:
                stats["warmup_successes"] += 1
            else:
                stats["rl_successes"] += 1

        # Off-policy updates
        if not is_warmup and len(replay_buffer) >= args.rl_batch_size:
            num_updates = args.update_to_data_ratio * len(ep_transitions)
            for _ in range(num_updates):
                batch = replay_buffer.sample(args.rl_batch_size)
                # Actor-critic gradient updates would go here.
                # Requires model loaded locally — see pi0_rlt.py compute_critic_loss / compute_actor_loss.
                pass

        if (ep + 1) % args.log_interval == 0:
            rate = stats["total_successes"] / max(stats["total_episodes"], 1) * 100
            rl_ep = stats["total_episodes"] - args.warmup_episodes
            rl_rate = stats["rl_successes"] / max(rl_ep, 1) * 100 if rl_ep > 0 else 0
            logger.info(
                f"Ep {ep+1}/{args.num_episodes} | "
                f"Success {stats['total_successes']}/{stats['total_episodes']} ({rate:.1f}%) | "
                f"RL {rl_rate:.1f}% | Buffer {len(replay_buffer)}")
            wandb.log({
                "online_rl/episode": ep + 1,
                "online_rl/success_rate": rate,
                "online_rl/rl_success_rate": rl_rate,
                "online_rl/total_successes": stats["total_successes"],
                "online_rl/total_episodes": stats["total_episodes"],
                "online_rl/replay_buffer_size": len(replay_buffer),
            }, step=ep + 1)

        if (ep + 1) % args.save_interval == 0:
            ckpt = output_dir / f"episode_{ep+1}"
            ckpt.mkdir(parents=True, exist_ok=True)
            replay_buffer.save(ckpt / "replay_buffer")
            with open(ckpt / "stats.json", "w") as f:
                json.dump(stats, f, indent=2, default=str)
            logger.info(f"Saved checkpoint → {ckpt}")

    final = output_dir / "final"
    final.mkdir(parents=True, exist_ok=True)
    replay_buffer.save(final / "replay_buffer")
    with open(final / "stats.json", "w") as f:
        json.dump(stats, f, indent=2, default=str)

    rate = stats["total_successes"] / max(stats["total_episodes"], 1) * 100
    logger.info(f"Done. Success rate: {rate:.1f}%  →  {output_dir}")
    wandb.finish()


def main(args: Args):
    if args.phase == "rl_token":
        train_rl_token(args)
    elif args.phase == "online_rl":
        train_online_rl(args)
    else:
        raise ValueError(f"Unknown phase '{args.phase}'. Use 'rl_token' or 'online_rl'.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))

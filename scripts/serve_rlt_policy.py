"""Serve a Pi0.5 policy with RL Token (RLT) actor-critic refinement.

Unlike serve_sde_policy.py (which only swaps the sampling algorithm on the
same checkpoint), RLT adds *separately trained* components:
  - RL token encoder-decoder (trained in Phase 1)
  - Actor-critic heads (trained in Phase 2 online RL)

Therefore this script must load TWO sets of weights:
  1. The base VLA checkpoint (frozen backbone)
  2. The RLT checkpoint (encoder-decoder + actor-critic)

For online RL training, do NOT use this server — run the model locally in
train_rlt.py so that actor-critic gradients can be computed directly.
This server is only for *inference/evaluation* after training is complete.

Usage:
  # Evaluate after full RLT training
  python scripts/serve_rlt_policy.py \\
    --base_checkpoint gs://openpi-assets/checkpoints/pi05_libero \\
    --rlt_checkpoint checkpoints/rlt_libero_perturbed/online_rl/final \\
    --port 8000

  # Evaluate after Phase 1 only (RL token trained, actor untrained → VLA mode)
  python scripts/serve_rlt_policy.py \\
    --base_checkpoint gs://openpi-assets/checkpoints/pi05_libero \\
    --rlt_checkpoint checkpoints/rlt_libero_perturbed/rl_token/best \\
    --mode vla \\
    --port 8000
"""

import dataclasses
import logging
import pathlib
import socket

import jax.numpy as jnp
import flax.nnx as nnx
import tyro

from openpi.models import model as _model
from openpi.models.pi0_rlt import Pi0RLTConfig
from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.serving import websocket_policy_server
from openpi.training import config as _config


@dataclasses.dataclass
class Args:
    """Arguments for serving the RLT policy."""

    # Base training config name.
    base_config: str = "pi05_libero"
    # Base VLA checkpoint (frozen backbone weights).
    base_checkpoint: str = "gs://openpi-assets/checkpoints/pi05_libero"
    # RLT checkpoint (encoder-decoder + actor-critic weights).
    # This is the output of train_rlt.py (Phase 1 or Phase 2).
    rlt_checkpoint: str = "checkpoints/rlt_libero/online_rl/final"

    # Inference mode:
    #   "rlt" — VLA generates reference action, actor refines it deterministically (requires trained actor).
    #   "rlt_tt" — Test-time: actor generates N candidates, critic selects the best one.
    #   "vla" — Standard VLA-only sampling (useful for testing RL token quality before RL).
    mode: str = "rlt"

    # Test-time (rlt_tt) parameters:
    num_tt_samples: int = 64  # Number of candidate actions to sample at test time
    tt_noise_std: float | None = None  # Noise std for test-time sampling (None = use actor_std)

    # Default prompt (if not provided by client).
    default_prompt: str | None = None

    # Server settings.
    port: int = 8000
    record: bool = False

    # RLT architecture (must match what was used during training).
    rl_token_dim: int = 2048
    encoder_num_layers: int = 2
    decoder_num_layers: int = 2
    rl_chunk_length: int = 10
    actor_hidden_dim: int = 256
    actor_num_layers: int = 2
    critic_hidden_dim: int = 256
    critic_num_layers: int = 2
    actor_std: float = 0.1
    vla_num_steps: int = 10


def create_rlt_policy(args: Args) -> _policy.Policy:
    """Load the full RLT policy from base VLA + RLT checkpoints.

    Two-stage weight loading:
      1. Create Pi0RLT model with Pi0RLTConfig
      2. Load base VLA weights (PaliGemma, action projections, etc.)
      3. Load RLT weights (encoder, decoder, actor, critic) on top
    """
    import jax
    from orbax import checkpoint as ocp
    import openpi.shared.download as download

    # Get base training config and extend with RLT config
    full_config = _config.get_config(args.base_config)
    original_model_config = full_config.model
    model_kwargs = dataclasses.asdict(original_model_config)

    rlt_config = Pi0RLTConfig(
        **model_kwargs,
        rl_token_dim=args.rl_token_dim,
        encoder_num_layers=args.encoder_num_layers,
        decoder_num_layers=args.decoder_num_layers,
        rl_chunk_length=args.rl_chunk_length,
        actor_hidden_dim=args.actor_hidden_dim,
        actor_num_layers=args.actor_num_layers,
        critic_hidden_dim=args.critic_hidden_dim,
        critic_num_layers=args.critic_num_layers,
        actor_std=args.actor_std,
        vla_num_steps=args.vla_num_steps,
    )

    full_config = dataclasses.replace(full_config, model=rlt_config)

    # Step 1: Create model (random init)
    logging.info("Creating Pi0RLT model...")
    model = nnx.eval_shape(rlt_config.create, jax.random.key(0))
    graphdef, state = nnx.split(model)

    # Step 2: Load base VLA weights (backbone)
    base_dir = pathlib.Path(download.maybe_download(args.base_checkpoint))
    logging.info("Loading base VLA weights from %s", base_dir)
    vla_params = _model.restore_params(base_dir / "params", dtype=jnp.bfloat16)
    # Intersect: only load params that exist in both VLA checkpoint and our model
    vla_intersect = ocp.transform_utils.intersect_trees(state.to_pure_dict(), vla_params)
    state.replace_by_pure_dict(vla_intersect)

    # Step 3: Load RLT weights (encoder-decoder + actor-critic)
    rlt_dir = pathlib.Path(args.rlt_checkpoint)
    if rlt_dir.exists() and (rlt_dir / "params").exists():
        logging.info("Loading RLT weights from %s", rlt_dir)
        rlt_params = _model.restore_params(rlt_dir / "params", dtype=jnp.bfloat16)
        # Intersect: only load RLT-specific params (encoder, decoder, actor, critic)
        rlt_intersect = ocp.transform_utils.intersect_trees(state.to_pure_dict(), rlt_params)
        state.replace_by_pure_dict(rlt_intersect)
        logging.info("RLT weights loaded successfully.")
    else:
        if args.mode == "rlt":
            logging.warning(
                "RLT checkpoint not found at %s. Actor-critic weights are random! "
                "Consider using --mode vla or providing a valid checkpoint.",
                rlt_dir,
            )
        else:
            logging.info("No RLT checkpoint loaded (mode=%s, VLA-only).", args.mode)

    model = nnx.merge(graphdef, state)

    # Build the data transform pipeline (same as standard policy)
    data_config = full_config.data.create(full_config.assets_dirs, full_config.model)
    from openpi.training import checkpoints as _checkpoints
    import openpi.transforms as transforms
    norm_stats = _checkpoints.load_norm_stats(base_dir / "assets", data_config.asset_id)

    return _policy.Policy(
        model,
        transforms=[
            transforms.InjectDefaultPrompt(args.default_prompt),
            *data_config.data_transforms.inputs,
            transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        output_transforms=[
            *data_config.model_transforms.outputs,
            transforms.Unnormalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.data_transforms.outputs,
        ],
        sample_kwargs={
            "mode": args.mode,
            "num_tt_samples": args.num_tt_samples,
            "tt_noise_std": args.tt_noise_std,
        },
        metadata=full_config.policy_metadata,
    )


def main(args: Args) -> None:
    policy = create_rlt_policy(args)
    policy_metadata = policy.metadata

    if args.record:
        policy = _policy.PolicyRecorder(policy, "policy_records")

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info(
        "RLT server starting (host=%s, ip=%s, mode=%s, port=%d)",
        hostname, local_ip, args.mode, args.port,
    )

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy_metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))

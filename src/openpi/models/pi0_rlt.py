"""Pi0 with RL Token (RLT) extraction.

Implements the RLT method from "RL Token: Bootstrapping Online RL with
Vision-Language-Action Models" (Xu et al., 2025).

The key idea: train an encoder-decoder transformer on top of the frozen VLA
to produce a compact "RL token" representation. This RL token, together with
the VLA's reference action chunk, serves as input to a lightweight actor-critic
for sample-efficient online RL.

Architecture overview:
  1. Frozen VLA (Pi0.5) produces prefix embeddings z_1:M via embed_prefix + LLM forward.
  2. An encoder transformer compresses [z_1:M, e_rl] into a single RL token z_rl (1 x width).
  3. A decoder transformer reconstructs z_1:M from z_rl (autoregressive reconstruction loss).
  4. During online RL, z_rl is used as the state representation for a lightweight actor-critic.
  5. The VLA also produces a reference action chunk a_tilde via standard diffusion sampling.
  6. The RL actor refines a_tilde conditioned on (z_rl, proprioceptive state, a_tilde).
"""

import dataclasses
import logging

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0
from openpi.models import pi0_config
import openpi.models.gemma as _gemma
import openpi.models.siglip as _siglip
from openpi.shared import array_typing as at

logger = logging.getLogger("openpi")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class Pi0RLTConfig(pi0_config.Pi0Config):
    """Configuration for Pi0 with RL Token extraction."""

    # RL token encoder-decoder parameters
    rl_token_dim: int = 2048  # Dimension of the RL token (compressed representation)
    encoder_num_layers: int = 2  # Number of transformer layers in the encoder
    encoder_num_heads: int = 8  # Number of attention heads in the encoder
    decoder_num_layers: int = 2  # Number of transformer layers in the decoder
    decoder_num_heads: int = 8  # Number of attention heads in the decoder

    # RL actor-critic parameters
    actor_hidden_dim: int = 256  # Hidden dimension of the actor MLP
    actor_num_layers: int = 2  # Number of hidden layers in the actor MLP
    critic_hidden_dim: int = 256  # Hidden dimension of the critic MLP
    critic_num_layers: int = 2  # Number of hidden layers in the critic MLP
    rl_chunk_length: int = 10  # Action chunk length for RL (C in the paper, C < H)
    actor_std: float = 0.1  # Fixed standard deviation for the Gaussian actor
    ref_action_dropout: float = 0.5  # Probability of dropping reference action during training

    # RL training parameters
    bc_regularizer_beta: float = 1.0  # BC regularization coefficient (beta in Eq. 5)
    discount_gamma: float = 0.99  # Discount factor for RL

    # VLA sampling parameters (for generating reference actions)
    vla_num_steps: int = 10  # Number of diffusion steps for VLA reference action

    @property
    @override
    def model_type(self) -> _model.ModelType:
        if self.pi05:
            return _model.ModelType.PI05_RLT
        return _model.ModelType.PI0_RLT

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        return super().inputs_spec(batch_size=batch_size)

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0RLT":
        return Pi0RLT(self, rngs=nnx.Rngs(rng))


# ---------------------------------------------------------------------------
# Lightweight Transformer blocks for encoder / decoder
# ---------------------------------------------------------------------------

class TransformerBlock(nnx.Module):
    """A single transformer block with multi-head self-attention and FFN."""

    def __init__(self, width: int, num_heads: int, rngs: nnx.Rngs):
        self.ln1 = nnx.LayerNorm(width, rngs=rngs)
        self.attn = nnx.MultiHeadAttention(
            num_heads=num_heads,
            in_features=width,
            qkv_features=width,
            out_features=width,
            decode=False,
            rngs=rngs,
        )
        self.ln2 = nnx.LayerNorm(width, rngs=rngs)
        self.ffn_up = nnx.Linear(width, width * 4, rngs=rngs)
        self.ffn_down = nnx.Linear(width * 4, width, rngs=rngs)

    def __call__(self, x, mask=None):
        # Self-attention with pre-norm
        h = self.ln1(x)
        h = self.attn(h, mask=mask)
        x = x + h
        # FFN with pre-norm
        h = self.ln2(x)
        h = self.ffn_up(h)
        h = nnx.gelu(h)
        h = self.ffn_down(h)
        x = x + h
        return x


class RLTokenEncoder(nnx.Module):
    """Encoder transformer that compresses VLA embeddings into an RL token.

    Appends a learned [RL] embedding to the VLA embedding sequence and processes
    through transformer layers. The output at the [RL] position is the RL token.

    z_rl = g_phi([z_1:M, e_rl])_{M+1}   (Eq. 1 in the paper)
    """

    def __init__(self, width: int, rl_token_dim: int, num_layers: int, num_heads: int, rngs: nnx.Rngs):
        self.rl_embedding = nnx.Param(jax.random.normal(rngs.params(), (1, 1, width)) * 0.02)
        self.layers = [TransformerBlock(width, num_heads, rngs) for _ in range(num_layers)]
        self.out_proj = nnx.Linear(width, rl_token_dim, rngs=rngs)

    def __call__(self, vla_embeddings: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
            vla_embeddings: (batch, seq_len, width) - VLA prefix embeddings z_1:M

        Returns:
            rl_token: (batch, rl_token_dim) - compressed RL token z_rl
        """
        batch_size = vla_embeddings.shape[0]
        # Append learned RL embedding
        rl_emb = jnp.broadcast_to(self.rl_embedding.value, (batch_size, 1, vla_embeddings.shape[-1]))
        x = jnp.concatenate([vla_embeddings, rl_emb], axis=1)  # (batch, M+1, width)

        for layer in self.layers:
            x = layer(x)

        # Extract the RL token (last position)
        rl_token = x[:, -1, :]  # (batch, width)
        rl_token = self.out_proj(rl_token)  # (batch, rl_token_dim)
        return rl_token


class RLTokenDecoder(nnx.Module):
    """Decoder transformer that reconstructs VLA embeddings from the RL token.

    Autoregressively reconstructs the original VLA embeddings from z_rl.
    Used for training the RL token representation (Eq. 2 in the paper).

    L_ro = E_D [ sum_i || h_phi(d_phi([z_rl, z_bar_1:i-1]))_i - z_bar_i ||^2 ]
    """

    def __init__(self, width: int, rl_token_dim: int, num_layers: int, num_heads: int, rngs: nnx.Rngs):
        self.in_proj = nnx.Linear(rl_token_dim, width, rngs=rngs)
        self.layers = [TransformerBlock(width, num_heads, rngs) for _ in range(num_layers)]
        self.out_proj = nnx.Linear(width, width, rngs=rngs)

    def __call__(self, rl_token: jnp.ndarray, target_embeddings: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
            rl_token: (batch, rl_token_dim) - the RL token z_rl
            target_embeddings: (batch, seq_len, width) - stop-gradient VLA embeddings z_bar_1:M

        Returns:
            reconstructed: (batch, seq_len, width) - reconstructed embeddings
        """
        # Project RL token back to width dimension
        rl_projected = self.in_proj(rl_token)[:, None, :]  # (batch, 1, width)

        # Autoregressive reconstruction: [z_rl, z_bar_1, ..., z_bar_{M-1}] -> predict [z_bar_1, ..., z_bar_M]
        seq_len = target_embeddings.shape[1]
        # Shift target embeddings right (prepend rl_token, remove last)
        shifted = jnp.concatenate([rl_projected, target_embeddings[:, :-1, :]], axis=1)  # (batch, seq_len, width)

        # Create causal mask for autoregressive decoding
        causal_mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=jnp.bool_))

        x = shifted
        for layer in self.layers:
            x = layer(x, mask=causal_mask)

        reconstructed = self.out_proj(x)  # (batch, seq_len, width)
        return reconstructed


# ---------------------------------------------------------------------------
# RL Actor and Critic (lightweight MLP networks)
# ---------------------------------------------------------------------------

class RLActor(nnx.Module):
    """Lightweight actor network for RL.

    pi_theta(a_1:C | x, a_tilde_1:C) = N(mu_theta(x, a_tilde_1:C), sigma^2 I)

    The actor is conditioned on:
      - x = (z_rl, s^p): RL token + proprioceptive state
      - a_tilde_1:C: reference action chunk from the VLA

    It outputs a Gaussian mean over action chunks.
    """

    def __init__(
        self,
        rl_token_dim: int,
        prop_state_dim: int,
        action_dim: int,
        chunk_length: int,
        hidden_dim: int,
        num_layers: int,
        std: float,
        rngs: nnx.Rngs,
    ):
        self.action_dim = action_dim
        self.chunk_length = chunk_length
        self.std = std

        # Input: rl_token + proprioceptive_state + flattened reference action chunk
        input_dim = rl_token_dim + prop_state_dim + action_dim * chunk_length
        layers = []
        for i in range(num_layers):
            in_d = input_dim if i == 0 else hidden_dim
            layers.append(nnx.Linear(in_d, hidden_dim, rngs=rngs))
        self.hidden_layers = layers
        self.output_layer = nnx.Linear(hidden_dim, action_dim * chunk_length, rngs=rngs)

    def __call__(
        self,
        rl_token: jnp.ndarray,
        prop_state: jnp.ndarray,
        ref_action_chunk: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """
        Args:
            rl_token: (batch, rl_token_dim)
            prop_state: (batch, prop_state_dim)
            ref_action_chunk: (batch, chunk_length, action_dim) - VLA reference actions

        Returns:
            mu: (batch, chunk_length, action_dim) - action mean
            std: scalar - fixed standard deviation
        """
        ref_flat = ref_action_chunk.reshape(ref_action_chunk.shape[0], -1)
        x = jnp.concatenate([rl_token, prop_state, ref_flat], axis=-1)

        for layer in self.hidden_layers:
            x = layer(x)
            x = nnx.relu(x)

        mu = self.output_layer(x)
        mu = mu.reshape(-1, self.chunk_length, self.action_dim)
        return mu, self.std

    def sample(
        self,
        rng: at.KeyArrayLike,
        rl_token: jnp.ndarray,
        prop_state: jnp.ndarray,
        ref_action_chunk: jnp.ndarray,
    ) -> jnp.ndarray:
        """Sample an action chunk from the policy."""
        mu, std = self(rl_token, prop_state, ref_action_chunk)
        noise = jax.random.normal(rng, mu.shape)
        return mu + std * noise


class RLCritic(nnx.Module):
    """Lightweight critic network for RL.

    Q_psi(x, a_1:C) -> R

    Takes state (z_rl + proprioceptive) and action chunk as input,
    outputs a scalar Q-value. Uses twin Q-functions (TD3 style).
    """

    def __init__(
        self,
        rl_token_dim: int,
        prop_state_dim: int,
        action_dim: int,
        chunk_length: int,
        hidden_dim: int,
        num_layers: int,
        rngs: nnx.Rngs,
    ):
        input_dim = rl_token_dim + prop_state_dim + action_dim * chunk_length

        # Q1
        q1_layers = []
        for i in range(num_layers):
            in_d = input_dim if i == 0 else hidden_dim
            q1_layers.append(nnx.Linear(in_d, hidden_dim, rngs=rngs))
        self.q1_layers = q1_layers
        self.q1_out = nnx.Linear(hidden_dim, 1, rngs=rngs)

        # Q2 (twin)
        q2_layers = []
        for i in range(num_layers):
            in_d = input_dim if i == 0 else hidden_dim
            q2_layers.append(nnx.Linear(in_d, hidden_dim, rngs=rngs))
        self.q2_layers = q2_layers
        self.q2_out = nnx.Linear(hidden_dim, 1, rngs=rngs)

    def __call__(
        self,
        rl_token: jnp.ndarray,
        prop_state: jnp.ndarray,
        action_chunk: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """
        Args:
            rl_token: (batch, rl_token_dim)
            prop_state: (batch, prop_state_dim)
            action_chunk: (batch, chunk_length, action_dim)

        Returns:
            q1, q2: (batch, 1) each - twin Q-values
        """
        action_flat = action_chunk.reshape(action_chunk.shape[0], -1)
        x = jnp.concatenate([rl_token, prop_state, action_flat], axis=-1)

        # Q1
        h1 = x
        for layer in self.q1_layers:
            h1 = layer(h1)
            h1 = nnx.relu(h1)
        q1 = self.q1_out(h1)

        # Q2
        h2 = x
        for layer in self.q2_layers:
            h2 = layer(h2)
            h2 = nnx.relu(h2)
        q2 = self.q2_out(h2)

        return q1, q2

    def q_min(
        self,
        rl_token: jnp.ndarray,
        prop_state: jnp.ndarray,
        action_chunk: jnp.ndarray,
    ) -> jnp.ndarray:
        """Return the minimum of twin Q-values (for target computation)."""
        q1, q2 = self(rl_token, prop_state, action_chunk)
        return jnp.minimum(q1, q2)


# ---------------------------------------------------------------------------
# Main Pi0RLT Model
# ---------------------------------------------------------------------------

class Pi0RLT(_model.BaseModel):
    """Pi0 model with RL Token extraction and actor-critic heads.

    This model wraps a frozen Pi0/Pi0.5 VLA and adds:
      1. RL Token encoder-decoder (for representation learning)
      2. Lightweight actor-critic (for online RL)

    During inference, it can operate in two modes:
      - "vla": Standard VLA action generation (same as Pi0)
      - "rlt": RL Token extraction + actor refinement of VLA actions
    """

    def __init__(self, config: Pi0RLTConfig, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.config = config
        self.pi05 = config.pi05

        # --- Frozen VLA backbone (same as Pi0/Pi0SDE) ---
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)

        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_config, action_expert_config],
                embed_dtype=config.dtype,
                adarms=config.pi05,
            )
        )
        llm.lazy_init(rngs=rngs, method="init", use_adarms=[False, True] if config.pi05 else [False, False])
        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)
        self.PaliGemma = nnx.Dict(llm=llm, img=img)

        self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        if config.pi05:
            self.time_mlp_in = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        else:
            self.state_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_in = nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)

        # --- RL Token encoder-decoder ---
        vla_width = paligemma_config.width  # Width of VLA prefix embeddings
        self.rl_encoder = RLTokenEncoder(
            width=vla_width,
            rl_token_dim=config.rl_token_dim,
            num_layers=config.encoder_num_layers,
            num_heads=config.encoder_num_heads,
            rngs=rngs,
        )
        self.rl_decoder = RLTokenDecoder(
            width=vla_width,
            rl_token_dim=config.rl_token_dim,
            num_layers=config.decoder_num_layers,
            num_heads=config.decoder_num_heads,
            rngs=rngs,
        )

        # --- RL Actor and Critic ---
        # prop_state_dim = action_dim (padded proprioceptive state, same as model action_dim)
        self.rl_actor = RLActor(
            rl_token_dim=config.rl_token_dim,
            prop_state_dim=config.action_dim,
            action_dim=config.action_dim,
            chunk_length=config.rl_chunk_length,
            hidden_dim=config.actor_hidden_dim,
            num_layers=config.actor_num_layers,
            std=config.actor_std,
            rngs=rngs,
        )
        self.rl_critic = RLCritic(
            rl_token_dim=config.rl_token_dim,
            prop_state_dim=config.action_dim,
            action_dim=config.action_dim,
            chunk_length=config.rl_chunk_length,
            hidden_dim=config.critic_hidden_dim,
            num_layers=config.critic_num_layers,
            rngs=rngs,
        )

        self.deterministic = True

    # ----- VLA forward methods (reused from Pi0) -----

    @at.typecheck
    def embed_prefix(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        input_mask = []
        ar_mask = []
        tokens = []
        for name in obs.images:
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)
            tokens.append(image_tokens)
            input_mask.append(
                einops.repeat(obs.image_masks[name], "b -> b s", s=image_tokens.shape[1])
            )
            ar_mask += [False] * image_tokens.shape[1]

        if obs.tokenized_prompt is not None:
            tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            tokens.append(tokenized_inputs)
            input_mask.append(obs.tokenized_prompt_mask)
            ar_mask += [False] * tokenized_inputs.shape[1]

        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask

    @at.typecheck
    def embed_suffix(
        self, obs: _model.Observation, noisy_actions: _model.Actions, timestep: at.Float[at.Array, " b"]
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
        at.Float[at.Array, "b emb"] | None,
    ]:
        input_mask = []
        ar_mask = []
        tokens = []
        if not self.pi05:
            state_token = self.state_proj(obs.state)[:, None, :]
            tokens.append(state_token)
            input_mask.append(jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_))
            ar_mask += [True]

        action_tokens = self.action_in_proj(noisy_actions)
        time_emb = pi0.posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
        if self.pi05:
            time_emb = self.time_mlp_in(time_emb)
            time_emb = nnx.swish(time_emb)
            time_emb = self.time_mlp_out(time_emb)
            time_emb = nnx.swish(time_emb)
            action_expert_tokens = action_tokens
            adarms_cond = time_emb
        else:
            time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=self.action_horizon)
            action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)
            action_time_tokens = self.action_time_mlp_in(action_time_tokens)
            action_time_tokens = nnx.swish(action_time_tokens)
            action_time_tokens = self.action_time_mlp_out(action_time_tokens)
            action_expert_tokens = action_time_tokens
            adarms_cond = None

        tokens.append(action_expert_tokens)
        input_mask.append(jnp.ones(action_expert_tokens.shape[:2], dtype=jnp.bool_))
        ar_mask += [True] + ([False] * (self.action_horizon - 1))
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask, adarms_cond

    # ----- RL Token extraction -----

    def extract_rl_token(self, observation: _model.Observation) -> jnp.ndarray:
        """Extract the RL token from the frozen VLA's prefix embeddings.

        Args:
            observation: Model observation (images, state, prompt).

        Returns:
            rl_token: (batch, rl_token_dim) - compressed RL representation.
        """
        observation = _model.preprocess_observation(None, observation, train=False)
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)

        # Forward through the VLA backbone (prefix only) to get contextual embeddings
        prefix_attn_mask = pi0.make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        # Only process prefix (no suffix/actions needed for RL token)
        (prefix_out, _), _ = self.PaliGemma.llm(
            [prefix_tokens, None], mask=prefix_attn_mask, positions=positions
        )

        # Stop gradient on VLA embeddings (frozen VLA)
        prefix_out_sg = jax.lax.stop_gradient(prefix_out)

        # Compress through RL token encoder
        rl_token = self.rl_encoder(prefix_out_sg)
        return rl_token

    # ----- VLA reference action sampling (standard diffusion) -----

    def sample_vla_reference(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | None = None,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> _model.Actions:
        """Sample reference action chunk from the frozen VLA (standard flow matching)."""
        if num_steps is None:
            num_steps = self.config.vla_num_steps
        observation = _model.preprocess_observation(None, observation, train=False)
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = pi0.make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        def step(carry):
            x_t, time = carry
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )
            suffix_attn_mask = pi0.make_attn_mask(suffix_mask, suffix_ar_mask)
            prefix_vis_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            full_attn_mask = jnp.concatenate([prefix_vis_mask, suffix_attn_mask], axis=-1)
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (_, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon:])
            return x_t + dt * v_t, time + dt

        def cond(carry):
            _, time = carry
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0

    # ----- RL Token reconstruction loss (for training encoder-decoder) -----

    def compute_rl_token_loss(
        self, observation: _model.Observation
    ) -> at.Float[at.Array, ""]:
        """Compute the autoregressive reconstruction loss for RL token training.

        L_ro = E_D [ sum_i || h_phi(d_phi([z_rl, z_bar_1:i-1]))_i - z_bar_i ||^2 ]
        """
        observation = _model.preprocess_observation(None, observation, train=False)
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)

        prefix_attn_mask = pi0.make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        (prefix_out, _), _ = self.PaliGemma.llm(
            [prefix_tokens, None], mask=prefix_attn_mask, positions=positions
        )

        # Stop gradient on VLA embeddings
        target_embeddings = jax.lax.stop_gradient(prefix_out)

        # Encode -> RL token
        rl_token = self.rl_encoder(target_embeddings)

        # Decode -> reconstruct embeddings
        reconstructed = self.rl_decoder(rl_token, target_embeddings)

        # Reconstruction loss
        loss = jnp.mean(jnp.square(reconstructed - target_embeddings))
        return loss

    # ----- Standard model interface -----

    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> at.Float[at.Array, "*b ah"]:
        """Compute the standard VLA flow matching loss (for joint fine-tuning)."""
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time)
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = pi0.make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions, adarms_cond=[None, adarms_cond]
        )
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon:])
        return jnp.mean(jnp.square(v_t - u_t), axis=-1)

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
        mode: str = "rlt",
        num_tt_samples: int = 64,
        tt_noise_std: float | None = None,
    ) -> _model.Actions:
        """Sample actions using either VLA-only, RLT, or RLT test-time mode.

        Args:
            mode: "vla" for standard VLA sampling, "rlt" for RL Token + actor deterministic
                  refinement, "rlt_tt" for test-time N-sample + critic selection.
            num_tt_samples: Number of candidate actions to generate in rlt_tt mode.
            tt_noise_std: Standard deviation for test-time sampling noise. If None, uses
                          the actor's configured std (actor_std).
        """
        if mode == "vla":
            return self.sample_vla_reference(rng, observation, num_steps=num_steps, noise=noise)

        # --- RLT mode (shared setup for both "rlt" and "rlt_tt") ---
        rng_vla, rng_actor = jax.random.split(rng)

        # 1. Extract RL token from frozen VLA
        rl_token = self.extract_rl_token(observation)

        # 2. Get reference action chunk from VLA
        ref_actions = self.sample_vla_reference(rng_vla, observation, num_steps=num_steps, noise=noise)

        # 3. Take first C steps as the RL chunk
        C = self.config.rl_chunk_length
        ref_chunk = ref_actions[:, :C, :]  # (batch, C, action_dim)

        # 4. Proprioceptive state
        prop_state = observation.state  # (batch, action_dim)

        # 5. Single forward pass through the actor to get the mean
        mu, actor_std = self.rl_actor(rl_token, prop_state, ref_chunk)
        # mu: (batch, C, action_dim)

        if mode == "rlt":
            # Deterministic: use the mean directly
            refined_chunk = mu
        elif mode == "rlt_tt":
            # Test-time: generate N candidates, score with critic, pick best
            sigma = tt_noise_std if tt_noise_std is not None else actor_std
            batch_size = mu.shape[0]

            # Generate N noise samples: (num_tt_samples, batch, C, action_dim)
            noise_samples = jax.random.normal(rng_actor, (num_tt_samples, *mu.shape))
            # Candidate actions: mu + sigma * epsilon_k
            candidates = mu[None] + sigma * noise_samples  # (N, batch, C, action_dim)

            # Score each candidate with the critic (min of twin Qs)
            def score_candidate(candidate):
                # candidate: (batch, C, action_dim)
                return self.rl_critic.q_min(rl_token, prop_state, candidate)  # (batch, 1)

            q_values = jax.vmap(score_candidate)(candidates)  # (N, batch, 1)
            q_values = q_values.squeeze(-1)  # (N, batch)

            # Select best candidate per batch element
            best_idx = jnp.argmax(q_values, axis=0)  # (batch,)
            refined_chunk = candidates[best_idx, jnp.arange(batch_size)]  # (batch, C, action_dim)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        # 6. Construct full action sequence: refined first C steps + remaining from VLA
        if C < self.action_horizon:
            remaining = ref_actions[:, C:, :]
            actions = jnp.concatenate([refined_chunk, remaining], axis=1)
        else:
            actions = refined_chunk[:, :self.action_horizon, :]

        return actions

    # ----- RL training losses -----

    def compute_critic_loss(
        self,
        rl_token: jnp.ndarray,
        prop_state: jnp.ndarray,
        action_chunk: jnp.ndarray,
        reward: jnp.ndarray,
        next_rl_token: jnp.ndarray,
        next_prop_state: jnp.ndarray,
        next_ref_chunk: jnp.ndarray,
        rng: at.KeyArrayLike,
        target_critic: "RLCritic",
    ) -> at.Float[at.Array, ""]:
        """Compute TD3-style critic loss (Eq. 3 in the paper).

        L_Q = E_{(x, a, x')} [ (Q_hat - Q_psi(x, a))^2 ]
        Q_hat = sum_{t'=1}^{C} gamma^{t'-1} r_{t'} + gamma^C E_{a'~pi} [Q_psi'(x', a')]
        """
        gamma = self.config.discount_gamma
        C = self.config.rl_chunk_length

        # Current Q-values
        q1, q2 = self.rl_critic(rl_token, prop_state, action_chunk)

        # Target Q-values (using target network)
        next_action = self.rl_actor.sample(rng, next_rl_token, next_prop_state, next_ref_chunk)
        next_q_min = jax.lax.stop_gradient(
            target_critic.q_min(next_rl_token, next_prop_state, next_action)
        )

        # TD target: reward + gamma^C * Q'(s', a')
        q_target = reward + (gamma ** C) * next_q_min
        q_target = jax.lax.stop_gradient(q_target)

        loss = jnp.mean(jnp.square(q1 - q_target) + jnp.square(q2 - q_target))
        return loss

    def compute_actor_loss(
        self,
        rng: at.KeyArrayLike,
        rl_token: jnp.ndarray,
        prop_state: jnp.ndarray,
        ref_chunk: jnp.ndarray,
    ) -> at.Float[at.Array, ""]:
        """Compute actor loss with BC regularization (Eq. 5 in the paper).

        L_pi(theta) = E_{s~B, a~pi} [ -Q_psi(x, a) + beta * ||a - a_tilde||^2 ]
        """
        beta = self.config.bc_regularizer_beta

        # Sample action from actor
        action = self.rl_actor.sample(rng, rl_token, prop_state, ref_chunk)

        # Q-value (use min of twin Qs)
        q_value = self.rl_critic.q_min(rl_token, prop_state, action)

        # BC regularization toward VLA reference
        bc_reg = jnp.mean(jnp.sum(jnp.square(action - ref_chunk), axis=(-2, -1)))

        loss = -jnp.mean(q_value) + beta * bc_reg
        return loss

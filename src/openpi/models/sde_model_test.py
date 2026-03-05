import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models import pi0_sde
from openpi.shared import nnx_utils

def test_pi05_sde_model_basic():
    key = jax.random.key(42)
    config = pi0_sde.Pi0SDEConfig(
        pi05=True, 
        noise_method="flow_sde", 
        noise_level=0.5,
        num_steps=2
        # num_steps=5
    )
    model = config.create(key)

    batch_size = 2
    obs = config.fake_obs(batch_size)
    actions = nnx_utils.module_jit(model.sample_actions)(key, obs, num_steps=5)
    
    assert actions.shape == (batch_size, config.action_horizon, config.action_dim)
    assert not jnp.any(jnp.isnan(actions)), "NaN"

def test_pi05_sde_stochastic_variance():
    key = jax.random.key(0)
    config = pi0_sde.Pi0SDEConfig(
        pi05=True, 
        noise_method="flow_sde", 
        noise_level=1.0,
        num_steps=3
        # num_steps=10
    )
    model = config.create(key)
    batch_size = 1
    obs = config.fake_obs(batch_size)
    
    init_noise_key = jax.random.key(123)
    fixed_noise = jax.random.normal(init_noise_key, (batch_size, config.action_horizon, config.action_dim))

    key_a, key_b = jax.random.split(key)
    actions_a = model.sample_actions(key_a, obs, noise=fixed_noise, num_steps=10)
    actions_b = model.sample_actions(key_b, obs, noise=fixed_noise, num_steps=10)

    diff = jnp.abs(actions_a - actions_b).mean()
    print(f"\nSDE Mean Difference: {diff}")
    assert diff > 1e-5, f"SDE sampling failed to produce differentiated results (diff: {diff})"

def test_pi05_sde_ode_consistency():
        key = jax.random.key(0)
        config = pi0_sde.Pi0SDEConfig(
            pi05=True,
            noise_method="flow_sde",
            noise_level=0.0,
            num_steps=2
            # num_steps=5
        )
        model = config.create(key)
        obs = config.fake_obs(1)

        key_init, key_a, key_b = jax.random.split(key, 3)
        
        fixed_noise = jax.random.normal(key_init, (1, model.action_horizon, model.action_dim))

        act_a = model.sample_actions(key_a, obs, noise=fixed_noise)
        act_b = model.sample_actions(key_b, obs, noise=fixed_noise)
        
        np.testing.assert_allclose(act_a, act_b, atol=1e-6)

def test_pi05_sde_batch_sampling():
    key = jax.random.key(7)
    num_samples = 5
    config = pi0_sde.Pi0SDEConfig(pi05=True, noise_level=0.5)
    model = config.create(key)

    obs = config.fake_obs(1)
    batched_obs = jax.tree.map(lambda x: jnp.repeat(x, num_samples, axis=0), obs)
    actions = model.sample_actions(key, batched_obs, num_steps=10)
    
    assert actions.shape == (num_samples, config.action_horizon, config.action_dim)
    unique_samples = jnp.unique(actions.reshape(num_samples, -1), axis=0)
    assert len(unique_samples) == num_samples, "Batch sampling results are duplicated; SDE randomness may have failed."

@pytest.mark.parametrize("noise_level", [0.1, 0.8])
def test_pi05_sde_different_noise_levels(noise_level):
    key = jax.random.key(1)
    config = pi0_sde.Pi0SDEConfig(pi05=True, noise_level=noise_level)
    model = config.create(key)
    obs = config.fake_obs(1)
    
    actions = model.sample_actions(key, obs, num_steps=5)
    assert not jnp.any(jnp.isnan(actions))
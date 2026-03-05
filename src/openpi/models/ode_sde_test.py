import jax
import jax.numpy as jnp

BATCH_SIZE = 1000
ACTION_DIM = 7
NUM_STEPS = 10
NOISE_LEVEL = 0.5
DT = -1.0 / NUM_STEPS

def mock_velocity(x, t):
    target = jnp.ones_like(x) * 0.5
    return (x - target) / jnp.maximum(t, 1e-3)

def sample_ode(rng):
    """ODE (Euler)"""
    rng, noise_rng = jax.random.split(rng)
    x_t = jax.random.normal(noise_rng, (BATCH_SIZE, ACTION_DIM))
    
    def step_fn(carry):
        x_t, time = carry
        v_t = mock_velocity(x_t, time)
        x_next = x_t + DT * v_t
        return x_next, time + DT
        
    def cond_fn(carry):
        _, time = carry
        return time >= -DT / 2

    x_0, _ = jax.lax.while_loop(cond_fn, step_fn, (x_t, 1.0))
    return x_0

def sample_flow_sde(rng):
    """rlinf Flow SDE"""
    rng_init, rng_loop = jax.random.split(rng)
    x_t = jax.random.normal(rng_init, (BATCH_SIZE, ACTION_DIM))
    
    def step_fn(carry):
        x_t, time, current_rng = carry
        step_rng, next_rng = jax.random.split(current_rng)
        
        v_t = mock_velocity(x_t, time)
        delta = jnp.abs(DT)
        
        denom = jnp.where(time >= 1.0 - 1e-4, delta, 1.0 - time)
        sigma_i = NOISE_LEVEL * jnp.sqrt(time / denom)
        
        x0_pred = x_t - v_t * time
        x1_pred = x_t + v_t * (1.0 - time)
        
        x0_weight = 1.0 - (time - delta)
        x1_weight = (time - delta) - (sigma_i**2 * delta) / (2.0 * time)
        
        x_t_mean = x0_pred * x0_weight + x1_pred * x1_weight
        x_t_std = jnp.sqrt(delta) * sigma_i
        
        z = jax.random.normal(step_rng, x_t.shape)
        x_next = x_t_mean + x_t_std * z
        
        return x_next, time + DT, next_rng

    def cond_fn(carry):
        _, time, _ = carry
        return time >= -DT / 2

    x_0, _, _ = jax.lax.while_loop(cond_fn, step_fn, (x_t, 1.0, rng_loop))
    return x_0

def sample_flow_sde_fixed(rng):
    """Fixed Flow SDE Sampling Results (zeroed last-step noise)"""
    rng_init, rng_loop = jax.random.split(rng)
    x_t = jax.random.normal(rng_init, (BATCH_SIZE, ACTION_DIM))
    
    def step_fn(carry):
        x_t, time, current_rng = carry
        step_rng, next_rng = jax.random.split(current_rng)
        
        v_t = mock_velocity(x_t, time)
        delta = jnp.abs(DT)
        
        denom = jnp.where(time >= 1.0 - 1e-4, delta, 1.0 - time)
        sigma_i = NOISE_LEVEL * jnp.sqrt(time / denom)
        
        x0_pred = x_t - v_t * time
        x1_pred = x_t + v_t * (1.0 - time)
        
        x0_weight = 1.0 - (time - delta)
        x1_weight = (time - delta) - (sigma_i**2 * delta) / (2.0 * time)
        
        x_t_mean = x0_pred * x0_weight + x1_pred * x1_weight
        
        is_last_step = time < delta * 1.5
        x_t_std = jnp.where(
            is_last_step,
            0.0,
            jnp.sqrt(delta) * sigma_i
        )
        
        z = jax.random.normal(step_rng, x_t.shape)
        x_next = x_t_mean + x_t_std * z
        
        return x_next, time + DT, next_rng

    def cond_fn(carry):
        _, time, _ = carry
        return time >= -DT / 2

    x_0, _, _ = jax.lax.while_loop(cond_fn, step_fn, (x_t, 1.0, rng_loop))
    return x_0

rng = jax.random.PRNGKey(42)
rng_ode, rng_sde, rng_sde_fixed = jax.random.split(rng, 3)

ode_actions = sample_ode(rng_ode)
sde_actions = sample_flow_sde(rng_sde)
sde_fixed_actions = sample_flow_sde_fixed(rng_sde_fixed)

print("=== Pure ODE Sampling Results ===")
print(f"Action Mean: {jnp.mean(ode_actions):.4f}")
print(f"Action Variance: {jnp.var(ode_actions):.4f}")

print("\n=== Flow SDE Sampling Results (with last-step noise) ===")
print(f"Action Mean: {jnp.mean(sde_actions):.4f}")
print(f"Action Variance: {jnp.var(sde_actions):.4f}")

print("\n=== Fixed Flow SDE Sampling Results (zeroed last-step noise) ===")
print(f"Action Mean: {jnp.mean(sde_fixed_actions):.4f}")
print(f"Action Variance: {jnp.var(sde_fixed_actions):.4f}")
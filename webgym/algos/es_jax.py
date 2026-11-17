"""Evolution strategies with the policy in jax: the whole population's forward pass is
one jit-compiled function (XLA), called once per env step."""

import jax
import jax.numpy as jnp
import numpy as np

from webgym.algos import es


def jax_forward(policy):
    """a jitted (params (P, num_params), obs (P, obs_dim)) -> actions (P, act_dim) for
    the layout of the numpy MLP `policy` (same weights, same results)"""
    shapes = policy.shapes

    @jax.jit
    def forward(params, obs):
        x, i = obs, 0
        for a, b in shapes:
            w = params[:, i : i + a * b].reshape(-1, a, b)
            i += a * b
            x = jnp.tanh(jnp.einsum("pa,pab->pb", x, w) + params[:, i : i + b])
            i += b
        return x

    return lambda params, obs: np.asarray(forward(jnp.asarray(params, jnp.float32), jnp.asarray(obs)))


def train(env, policy, **kwargs):
    """es.train with the jitted policy (same parameters, same state)"""
    return es.train(env, policy, forward=jax_forward(policy), **kwargs)

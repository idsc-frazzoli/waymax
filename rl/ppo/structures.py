from typing import Sequence, NamedTuple

import distrax
import flax.linen as nn
import jax.numpy as jnp
import numpy as np
from flax.linen.initializers import constant, orthogonal
from jax import Array
from jaxtyping import Float


class ActorCritic(nn.Module):
    action_dim: Sequence[int]
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x: Float[Array, "b O"]) -> tuple[distrax.Distribution, Float[Array, "b"]]:
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh
        actor_mean = nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(actor_mean)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(actor_mean)
        actor_mean = activation(actor_mean)  ## TODO: check if this is necessary
        actor_logtstd = self.param("log_std", nn.initializers.zeros, (self.action_dim,))
        pi = distrax.MultivariateNormalDiag(actor_mean, jnp.exp(actor_logtstd))

        critic = nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        critic = activation(critic)
        critic = nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(critic)
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(critic)

        return pi, jnp.squeeze(critic, axis=-1)


class Transition(NamedTuple):
    done: Float[Array, "?"]
    action: Float[Array, "?"]
    value: Float[Array, "?"] # todo seems to be a dict[str ,... ]
    reward: Float[Array, "?"]
    log_prob: Float[Array, "?"]
    obs: Float[Array, "?"]
    info: Float[Array, "?"]

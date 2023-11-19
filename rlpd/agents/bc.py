"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""


from functools import partial
from typing import Dict, Optional, Sequence, Tuple

import flax
import gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import struct
from flax.training.train_state import TrainState
from flax.core import FrozenDict

from rlpd.agents.agent import Agent
from rlpd.data.dataset import DatasetDict
from rlpd.distributions import TanhNormal
from rlpd.networks import MLP


class BCAgent(Agent):
    @classmethod
    def create(
        cls,
        seed: int,
        observation_space: gym.Space,
        action_space: gym.Space,
        actor_lr: float = 3e-4,
        hidden_dims: Sequence[int] = (256, 256),
        use_pnorm: bool = False,
    ):
        """
        An implementation of the version of Soft-Actor-Critic described in https://arxiv.org/abs/1812.05905
        """

        action_dim = action_space.shape[-1]
        observations = observation_space.sample()

        rng = jax.random.PRNGKey(seed)
        rng, actor_key = jax.random.split(rng, 2)

        actor_base_cls = partial(
            MLP, hidden_dims=hidden_dims, activate_final=True, use_pnorm=use_pnorm
        )
        actor_def = TanhNormal(actor_base_cls, action_dim)
        actor_params = FrozenDict(actor_def.init(actor_key, observations)["params"])
        actor = TrainState.create(
            apply_fn=actor_def.apply,
            params=actor_params,
            tx=optax.adam(learning_rate=actor_lr),
        )

        return cls(
            rng=rng,
            actor=actor,
        )

    def update_actor(self, batch: DatasetDict) -> Tuple[Agent, Dict[str, float]]:
        def actor_loss_fn(actor_params) -> Tuple[jnp.ndarray, Dict[str, float]]:
            dist = self.actor.apply_fn({"params": actor_params}, batch["observations"])
            actor_loss = -dist.log_prob(batch["actions"]).mean()
            return actor_loss, {"pretrain_bc_loss": actor_loss}

        grads, actor_info = jax.grad(actor_loss_fn, has_aux=True)(self.actor.params)
        actor = self.actor.apply_gradients(grads=grads)

        return self.replace(actor=actor), actor_info

    @partial(jax.jit, static_argnames="utd_ratio")
    def update(self, batch: DatasetDict, utd_ratio: int):

        new_agent = self
        for i in range(utd_ratio):

            def slice(x):
                assert x.shape[0] % utd_ratio == 0
                batch_size = x.shape[0] // utd_ratio
                return x[batch_size * i : batch_size * (i + 1)]

            mini_batch = jax.tree_util.tree_map(slice, batch)
            new_agent, actor_info = new_agent.update_actor(mini_batch)

        return new_agent, actor_info

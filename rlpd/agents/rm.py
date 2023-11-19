
from functools import partial
from typing import Dict, Optional, Sequence, Tuple

import flax
import gym
import jax
import jax.numpy as jnp
import optax
from flax import struct
from flax.training.train_state import TrainState
from flax.core import FrozenDict

from rlpd.data.dataset import DatasetDict
from rlpd.networks import (MLP, StateActionValue,)
from rlpd.types import PRNGKey

class RM(struct.PyTreeNode):
    rng: PRNGKey
    init_r_net: TrainState
    init_m_net: TrainState
    r_net: TrainState
    m_net: TrainState

    @classmethod
    def create(
        cls,
        seed: int,
        observation_space: gym.Space,
        action_space: gym.Space,
        lr: float = 3e-4,
        hidden_dims: Sequence[int] = (256, 256),
    ):

        observations = observation_space.sample()
        actions = action_space.sample()

        rng = jax.random.PRNGKey(seed)
        rng, key1, key2 = jax.random.split(rng, 3)
        
        base_cls = partial(MLP, hidden_dims=hidden_dims, activate_final=True)
        net_def = StateActionValue(base_cls)
        r_params = FrozenDict(net_def.init(key1, observations, actions)["params"])
        r_net = TrainState.create(
            apply_fn=net_def.apply,
            params=r_params,
            tx=optax.adam(learning_rate=lr),
        )

        m_params = FrozenDict(net_def.init(key2, observations, actions)["params"])
        m_net = TrainState.create(
            apply_fn=net_def.apply,
            params=m_params,
            tx=optax.adam(learning_rate=lr),
        )

        return cls(
            rng=rng,
            init_r_net=r_net,
            init_m_net=m_net,
            r_net=r_net,
            m_net=m_net,
        )

    @jax.jit
    def reset(self):
        return self.replace(r_net=self.init_r_net, m_net=self.init_m_net)

    def _update(self, batch: DatasetDict) -> Tuple[struct.PyTreeNode, Dict[str, float]]:
        def r_loss_fn(r_params) -> Tuple[jnp.ndarray, Dict[str, float]]:
            rs = self.r_net.apply_fn({"params": r_params}, batch["observations"], batch["actions"])
            
            loss = ((rs - batch["rewards"]) ** 2.).mean()
            return loss, {"r_loss": loss}

        grads, r_info = jax.grad(r_loss_fn, has_aux=True)(self.r_net.params)
        r_net = self.r_net.apply_gradients(grads=grads)

        def m_loss_fn(m_params) -> Tuple[jnp.ndarray, Dict[str, float]]:
            ms = self.m_net.apply_fn({"params": m_params}, batch["observations"], batch["actions"])
            
            loss = optax.sigmoid_binary_cross_entropy(ms, batch["masks"]).mean()
            return loss, {"m_loss": loss}

        grads, m_info = jax.grad(m_loss_fn, has_aux=True)(self.m_net.params)
        m_net = self.m_net.apply_gradients(grads=grads)

        return self.replace(r_net=r_net, m_net=m_net), {**r_info, **m_info}

    @partial(jax.jit, static_argnames="utd_ratio")
    def update(self, batch: DatasetDict, utd_ratio: int):

        new_self = self
        for i in range(utd_ratio):

            def slice(x):
                assert x.shape[0] % utd_ratio == 0
                batch_size = x.shape[0] // utd_ratio
                return x[batch_size * i : batch_size * (i + 1)]

            mini_batch = jax.tree_util.tree_map(slice, batch)
            new_self, info = new_self._update(mini_batch)

        return new_self, info

    @jax.jit
    def evaluate(self, batch: DatasetDict):
        rewards = self.get_reward(batch["observations"], batch["actions"])
        masks = self.get_mask(batch["observations"], batch["actions"])
        info = {
            "val_r_loss": ((rewards - batch["rewards"]) ** 2.).mean(),
            "val_m_loss": optax.sigmoid_binary_cross_entropy(masks, batch["masks"]).mean()
        }
        return info

    @jax.jit
    def get_reward(self, observations, actions):
        rewards = self.r_net.apply_fn({"params": self.r_net.params}, observations, actions)
        return rewards

    @jax.jit
    def get_mask(self, observations, actions):
        logits = self.m_net.apply_fn({"params": self.m_net.params}, observations, actions)
        return jax.nn.sigmoid(logits)

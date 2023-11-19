"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

from functools import partial
from itertools import zip_longest
from typing import Callable, Optional, Sequence, Tuple, Dict

import flax
import gym
import jax
import jax.numpy as jnp
import optax
from flax import struct
from flax.training.train_state import TrainState
from flax.core import FrozenDict

from rlpd.agents.drq.augmentations import batched_random_crop
from rlpd.data.dataset import DatasetDict
from rlpd.networks import MLP, PixelMultiplexer, StateFeature
from rlpd.types import PRNGKey
from rlpd.networks.encoders import D4PGEncoder

from rlpd.agents.drq.drq_learner import _unpack


class PixelRND(struct.PyTreeNode):
    rng: PRNGKey
    net: TrainState
    frozen_net: TrainState
    coeff: float = struct.field(pytree_node=False)
    data_augmentation_fn: Callable = struct.field(pytree_node=False)

    @classmethod
    def create(
        cls,
        seed: int,
        observation_space: gym.Space,
        action_space: gym.Space,
        lr: float = 3e-4,
        coeff: float = 1.0,
        cnn_features: Sequence[int] = (32, 32, 32, 32),
        cnn_filters: Sequence[int] = (3, 3, 3, 3),
        cnn_strides: Sequence[int] = (2, 1, 1, 1),
        cnn_padding: str = "VALID",
        latent_dim: int = 50,
        feature_dim: int = 256,
        encoder: str = "d4pg",
        hidden_dims: Sequence[int] = (256, 256),
        pixel_keys: Tuple[str, ...] = ("pixels",),
        depth_keys: Tuple[str, ...] = (),
    ):

        observations = observation_space.sample()
        actions = action_space.sample()

        rng = jax.random.PRNGKey(seed)
        rng, key1, key2 = jax.random.split(rng, 3)

        if encoder == "d4pg":
            encoder_cls = partial(
                D4PGEncoder,
                features=cnn_features,
                filters=cnn_filters,
                strides=cnn_strides,
                padding=cnn_padding,
            )
        else:
            raise NotImplementedError
        rnd_base_cls = partial(
            MLP,
            hidden_dims=hidden_dims,
            activate_final=True,
        )
        rnd_cls = partial(StateFeature, base_cls=rnd_base_cls, feature_dim=feature_dim)
        net_def = PixelMultiplexer(
            encoder_cls=encoder_cls,
            network_cls=rnd_cls,
            latent_dim=latent_dim,
            pixel_keys=pixel_keys,
            depth_keys=depth_keys,
        )
        params = FrozenDict(net_def.init(key1, observations)["params"])
        net = TrainState.create(
            apply_fn=net_def.apply,
            params=params,
            tx=optax.adam(learning_rate=lr),
        )
        frozen_params = FrozenDict(net_def.init(key2, observations)["params"])
        frozen_net = TrainState.create(
            apply_fn=net_def.apply,
            params=frozen_params,
            tx=optax.adam(learning_rate=lr),
        )

        def data_augmentation_fn(rng, observations):
            for pixel_key, depth_key in zip_longest(pixel_keys, depth_keys):
                key, rng = jax.random.split(rng)
                observations = batched_random_crop(key, observations, pixel_key)
                if depth_key is not None:
                    observations = batched_random_crop(key, observations, depth_key)
            return observations

        return cls(
            rng=rng,
            net=net,
            frozen_net=frozen_net,
            coeff=coeff,
            data_augmentation_fn=data_augmentation_fn,
        )

    @jax.jit
    def update(self, batch: DatasetDict) -> Tuple[struct.PyTreeNode, Dict[str, float]]:

        rng, key = jax.random.split(self.rng)
        observations = self.data_augmentation_fn(key, batch["observations"])
        rng, key = jax.random.split(rng)
        next_observations = self.data_augmentation_fn(key, batch["next_observations"])
        batch = batch.copy(
            add_or_replace={
                "observations": observations,
                "next_observations": next_observations,
            }
        )
        new_self = self.replace(rng=rng)

        def loss_fn(params) -> Tuple[jnp.ndarray, Dict[str, float]]:
            feats = new_self.net.apply_fn({"params": params}, batch["observations"])
            frozen_feats = new_self.frozen_net.apply_fn(
                {"params": new_self.frozen_net.params}, batch["observations"]
            )

            loss = ((feats - frozen_feats) ** 2.0).mean()
            return loss, {"rnd_loss": loss}

        grads, info = jax.grad(loss_fn, has_aux=True)(new_self.net.params)
        net = new_self.net.apply_gradients(grads=grads)

        return new_self.replace(net=net), info

    @jax.jit
    def get_reward(self, batch):
        if "pixels" not in batch["next_observations"]:
            batch = _unpack(batch)
        feats = self.net.apply_fn({"params": self.net.params}, batch["observations"])
        frozen_feats = self.net.apply_fn(
            {"params": self.frozen_net.params}, batch["observations"]
        )
        return jnp.mean((feats - frozen_feats) ** 2.0, axis=-1) * self.coeff

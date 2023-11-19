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
from rlpd.agents.sac.temperature import Temperature
from rlpd.data.dataset import DatasetDict
from rlpd.distributions import TanhNormal
from rlpd.networks import (
    MLP,
    PixelMultiplexer,
)

from rlpd.networks.encoders import D4PGEncoder
from rlpd.agents.bc import BCAgent


class PixelBCAgent(BCAgent):
    @classmethod
    def create(
        cls,
        seed: int,
        observation_space: gym.Space,
        action_space: gym.Space,
        actor_lr: float = 3e-4,
        cnn_features: Sequence[int] = (32, 32, 32, 32),
        cnn_filters: Sequence[int] = (3, 3, 3, 3),
        cnn_strides: Sequence[int] = (2, 1, 1, 1),
        cnn_padding: str = "VALID",
        latent_dim: int = 50,
        hidden_dims: Sequence[int] = (256, 256),
        pixel_keys: Tuple[str, ...] = ("pixels",),
        depth_keys: Tuple[str, ...] = (),
        encoder: str = "d4pg",
    ):
        assert encoder == "d4pg"
        action_dim = action_space.shape[-1]
        observations = observation_space.sample()

        rng = jax.random.PRNGKey(seed)
        rng, actor_key = jax.random.split(rng, 2)

        encoder_cls = partial(
            D4PGEncoder,
            features=cnn_features,
            filters=cnn_filters,
            strides=cnn_strides,
            padding=cnn_padding,
        )
        actor_base_cls = partial(MLP, hidden_dims=hidden_dims, activate_final=True)
        actor_cls = partial(TanhNormal, base_cls=actor_base_cls, action_dim=action_dim)
        actor_def = PixelMultiplexer(
            encoder_cls=encoder_cls,
            network_cls=actor_cls,
            latent_dim=latent_dim,
            stop_gradient=False,
            pixel_keys=pixel_keys,
            depth_keys=depth_keys,
        )
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

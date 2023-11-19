"""
Modified from https://github.com/ikostrikov/rlpd/blob/main/rlpd/networks/state_action_value.py

Original lincense information:

MIT License

Copyright (c) 2022 Ilya Kostrikov, Philip J. Ball, Laura Smith

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

"""


import flax.linen as nn
import jax.numpy as jnp

from rlpd.networks import default_init


class StateActionValue(nn.Module):
    base_cls: nn.Module

    @nn.compact
    def __call__(
        self, observations: jnp.ndarray, actions: jnp.ndarray, *args, **kwargs
    ) -> jnp.ndarray:
        inputs = jnp.concatenate([observations, actions], axis=-1)
        outputs = self.base_cls()(inputs, *args, **kwargs)

        value = nn.Dense(1, kernel_init=default_init())(outputs)

        return jnp.squeeze(value, -1)


class StateActionFeature(nn.Module):
    base_cls: nn.Module
    feature_dim: int

    @nn.compact
    def __call__(
        self, observations: jnp.ndarray, actions: jnp.ndarray, *args, **kwargs
    ) -> jnp.ndarray:
        inputs = jnp.concatenate([observations, actions], axis=-1)
        outputs = self.base_cls()(inputs, *args, **kwargs)
        feature = nn.Dense(self.feature_dim, kernel_init=default_init())(outputs)
        return feature


class StateValue(nn.Module):
    base_cls: nn.Module

    @nn.compact
    def __call__(
        self, observations: jnp.ndarray, *args, **kwargs
    ) -> jnp.ndarray:
        inputs = observations
        outputs = self.base_cls()(inputs, *args, **kwargs)

        value = nn.Dense(1, kernel_init=default_init())(outputs)

        return jnp.squeeze(value, -1)


class StateFeature(nn.Module):
    base_cls: nn.Module
    feature_dim: int

    @nn.compact
    def __call__(
        self, observations: jnp.ndarray, *args, **kwargs
    ) -> jnp.ndarray:
        inputs = observations
        outputs = self.base_cls()(inputs, *args, **kwargs)
        feature = nn.Dense(self.feature_dim, kernel_init=default_init())(outputs)
        return feature

"""
Modified from https://github.com/ikostrikov/rlpd/blob/main/rlpd/networks/ensemble.py

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

from typing import Type

import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.core import FrozenDict


class Ensemble(nn.Module):
    net_cls: Type[nn.Module]
    num: int = 2

    @nn.compact
    def __call__(self, *args):
        ensemble = nn.vmap(
            self.net_cls,
            variable_axes={"params": 0},
            split_rngs={"params": True, "dropout": True},
            in_axes=None,
            out_axes=0,
            axis_size=self.num,
        )
        return ensemble()(*args)


def subsample_ensemble(key: jax.random.PRNGKey, params, num_sample: int, num_qs: int):
    params = FrozenDict(params)
    if num_sample is not None:
        all_indx = jnp.arange(0, num_qs)
        indx = jax.random.choice(key, a=all_indx, shape=(num_sample,), replace=False)

        if "Ensemble_0" in params:
            ens_params = jax.tree_util.tree_map(
                lambda param: param[indx], params["Ensemble_0"]
            )
            params = params.copy(add_or_replace={"Ensemble_0": ens_params})
        else:
            params = jax.tree_util.tree_map(lambda param: param[indx], params)
    return params

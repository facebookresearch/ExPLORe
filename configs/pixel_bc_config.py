# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import ml_collections

from configs import pixel_config


def get_config():
    config = pixel_config.get_config()

    config.model_cls = "PixelBCAgent"

    config.actor_lr = 3e-4
    config.hidden_dims = (256, 256, 256)

    return config

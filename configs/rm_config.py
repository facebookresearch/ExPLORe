# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import ml_collections

def get_config():
    config = ml_collections.ConfigDict()

    config.model_cls = "RM"
    config.lr = 3e-4
    config.hidden_dims = (256, 256, 256)
    return config

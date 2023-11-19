# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.



from ml_collections.config_dict import config_dict

from configs import pixel_config

def get_config():
    config = pixel_config.get_config()

    config.model_cls = "PixelRND"
    config.lr = 3e-4
    config.hidden_dims = (256, 256)
    config.coeff = 1.
    return config

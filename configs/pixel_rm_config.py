from ml_collections.config_dict import config_dict

from configs import pixel_config

def get_config():
    config = pixel_config.get_config()

    config.model_cls = "PixelRM"
    config.lr = 3e-4
    config.hidden_dims = (256, 256)
    return config


"""Every model in Zoey-TinyWriter, by the name used in config files."""

from .gpt import GPT, GPTConfig

MODELS = {
    "gpt": (GPT, GPTConfig),
}


def build_model(name: str, **config):
    model_cls, config_cls = MODELS[name]
    return model_cls(config_cls(**config))

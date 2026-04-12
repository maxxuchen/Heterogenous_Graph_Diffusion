import torch.nn as nn
from functools import partial


def create_activation(name: str) -> nn.Module:
    mapping = {
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "prelu": nn.PReLU,
        "elu": nn.ELU,
    }
    if name is None:
        return nn.Identity()
    if name not in mapping:
        raise NotImplementedError(f"Activation '{name}' is not supported.")
    return mapping[name]()


def create_norm(name: str):
    if name == "layernorm":
        return nn.LayerNorm
    elif name == "batchnorm":
        return nn.BatchNorm1d
    else:
        return nn.Identity

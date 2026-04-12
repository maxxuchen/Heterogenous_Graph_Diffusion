"""Training utilities: optimizers, poolers, random seeding, checkpointing."""

import os
import random
import shutil

import numpy as np
import torch
import torch.optim as optim
from dgl.nn.pytorch.glob import AvgPooling, MaxPooling, SumPooling


def set_random_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def create_optimizer(opt_name: str, model, lr: float,
                     weight_decay: float) -> torch.optim.Optimizer:
    opt_name = opt_name.lower()
    params = model.parameters()
    kwargs = dict(lr=lr, weight_decay=weight_decay)
    if opt_name == "adam":
        return optim.Adam(params, **kwargs)
    elif opt_name == "adamw":
        return optim.AdamW(params, **kwargs)
    elif opt_name == "sgd":
        return optim.SGD(params, lr=lr, weight_decay=weight_decay, momentum=0.9)
    else:
        raise NotImplementedError(f"Optimizer '{opt_name}' is not supported.")


def create_pooler(name: str):
    if name == "mean":
        return AvgPooling()
    elif name == "max":
        return MaxPooling()
    elif name == "sum":
        return SumPooling()
    else:
        raise NotImplementedError(f"Pooler '{name}' is not supported.")


def adjust_learning_rate(optimizer, epoch: int, lr: float,
                          alpha: float, decay: int):
    new_lr = lr * (alpha ** (epoch // decay))
    for pg in optimizer.param_groups:
        pg["lr"] = new_lr


def save_checkpoint(state: dict, is_best: bool, checkpoint_dir: str):
    os.makedirs(checkpoint_dir, exist_ok=True)
    ckp_path = os.path.join(checkpoint_dir, "checkpoint.pth.tar")
    torch.save(state, ckp_path)
    if is_best:
        shutil.copyfile(ckp_path, os.path.join(checkpoint_dir, "model_best.pth.tar"))

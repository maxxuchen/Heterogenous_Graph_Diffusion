"""
Training script for Heterogeneous Graph Diffusion (HGD).

Usage:
    python train_hgd.py --config configs/MUTAG.yaml
"""

import argparse
import logging
import os
import os.path as osp

import dgl
import numpy as np
import torch
import yaml
from dgl.dataloading import GraphDataLoader
from easydict import EasyDict as edict
from torch.utils.data.sampler import SubsetRandomSampler

from src.datasets import load_graph_classification_dataset
from src.evaluate import evaluate_hgd
from src.models import HeterogeneousGraphDiffusion
from src.utils.training import (
    adjust_learning_rate,
    create_optimizer,
    create_pooler,
    save_checkpoint,
    set_random_seed,
)


def get_args():
    parser = argparse.ArgumentParser(description="HGD Training")
    parser.add_argument("--config", type=str, default="configs/MUTAG.yaml")
    parser.add_argument("--output_dir", type=str, default="logs/hgd")
    parser.add_argument("--resume", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=None,
                        help="Override seed(s) from config")
    return parser.parse_args()


def collate_fn(batch):
    graphs, labels = zip(*batch)
    return dgl.batch(graphs), torch.stack(labels)


def pretrain_epoch(model, loader, optimizer, device, epoch, logger):
    model.train()
    losses = []
    for batch_g, _ in loader:
        batch_g = batch_g.to(device)
        feat = batch_g.ndata["attr"]
        loss, _ = model(batch_g, feat)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    lr = optimizer.param_groups[0]["lr"]
    logger.info(f"Epoch {epoch:4d} | loss={np.mean(losses):.4f} | lr={lr:.2e}")


def main():
    args = get_args()
    with open(args.config) as f:
        cfg = edict(yaml.safe_load(f))

    os.makedirs(args.output_dir, exist_ok=True)
    checkpoint_dir = osp.join(args.output_dir, "checkpoints")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(osp.join(args.output_dir, "train.log")),
            logging.StreamHandler(),
        ],
    )
    logger = logging.getLogger("hgd")

    graphs, (num_features, num_classes) = load_graph_classification_dataset(
        cfg.DATA.data_name, deg4feat=cfg.DATA.deg4feat, PE=False
    )

    train_loader = GraphDataLoader(
        graphs,
        sampler=SubsetRandomSampler(torch.arange(len(graphs))),
        collate_fn=collate_fn,
        batch_size=cfg.DATALOADER.BATCH_SIZE,
        pin_memory=True,
    )
    eval_loader = GraphDataLoader(
        graphs, collate_fn=collate_fn, batch_size=len(graphs), shuffle=False
    )

    pooler = create_pooler(cfg.MODEL.pooler)
    device = cfg.get("DEVICE", "cpu")

    seeds = [args.seed] if args.seed is not None else cfg.seeds
    acc_list = []

    for run_i, seed in enumerate(seeds):
        logger.info(f"===== Run {run_i} | seed={seed} =====")
        set_random_seed(seed)

        model_cfg = dict(cfg.MODEL)
        model_cfg["in_dim"] = num_features
        model = HeterogeneousGraphDiffusion(**model_cfg).to(device)
        logger.info(f"Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

        optimizer = create_optimizer(
            cfg.SOLVER.optim_type, model, cfg.SOLVER.LR, cfg.SOLVER.weight_decay
        )

        best_f1, best_epoch = float("-inf"), 0
        for epoch in range(cfg.SOLVER.MAX_EPOCH):
            adjust_learning_rate(
                optimizer, epoch, cfg.SOLVER.LR,
                cfg.SOLVER.alpha, cfg.SOLVER.decay
            )
            pretrain_epoch(model, train_loader, optimizer, device, epoch, logger)

            if (epoch + 1) % 10 == 0 and epoch > 1:
                model.eval()
                f1 = evaluate_hgd(
                    model, cfg.eval_T, cfg.MODEL.T,
                    pooler, eval_loader, device, logger
                )
                if f1 > best_f1:
                    best_f1, best_epoch = f1, epoch
                    save_checkpoint(
                        {"epoch": epoch + 1, "state_dict": model.state_dict(),
                         "best_f1": best_f1, "optimizer": optimizer.state_dict()},
                        is_best=True, checkpoint_dir=checkpoint_dir,
                    )
                logger.info(f"  f1={f1:.4f}  best={best_f1:.4f} @ epoch {best_epoch}")

        acc_list.append(best_f1)

    mean_f1 = np.mean(acc_list)
    std_f1 = np.std(acc_list)
    logger.info(f"Final: {mean_f1:.4f} ± {std_f1:.4f}")


if __name__ == "__main__":
    main()

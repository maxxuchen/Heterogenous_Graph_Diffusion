"""
Evaluation: extract graph embeddings via HGD or DDM, then classify with SVM.
"""

from multiprocessing import Pool

import numpy as np
import torch
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import SVC

from src.models.hgd import build_laplacian, precompute_laplacian_diffusion


def _svm_fold(args):
    T_list, train_idx, test_idx, embed_list, labels = args
    preds_per_T = []
    for emb in embed_list:
        svc = GridSearchCV(SVC(random_state=42), {"C": [1e-3, 1e-2, 0.1, 1, 10]})
        svc.fit(emb[train_idx], labels[train_idx])
        preds_per_T.append(svc.predict(emb[test_idx]))
    # Majority vote over T values
    preds = np.stack(preds_per_T, axis=0)
    preds = torch.mode(torch.from_numpy(preds), dim=0)[0].long().numpy()
    return f1_score(labels[test_idx], preds, average="micro")


def evaluate_embeddings_svm(embed_list: list, labels: np.ndarray,
                             T_list: list, n_splits: int = 10) -> tuple:
    """
    Parameters
    ----------
    embed_list : list of (N, D) numpy arrays, one per T value
    labels     : (N,) int array
    T_list     : list of timestep values (for logging only)
    n_splits   : number of cross-validation folds

    Returns
    -------
    (mean_f1, std_f1)
    """
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
    fold_args = [
        (T_list, tr, te, embed_list, labels)
        for tr, te in kf.split(embed_list[0], labels)
    ]
    with Pool(n_splits) as pool:
        results = pool.map(_svm_fold, fold_args)
    return float(np.mean(results)), float(np.std(results))


@torch.no_grad()
def evaluate_hgd(model, T_list: list, cfg_T: int,
                 pooler, dataloader, device, logger) -> float:
    """Evaluate HeterogeneousGraphDiffusion via SVM on pooled embeddings."""
    model.eval()
    embed_list = []
    y_all = None

    for t_val in T_list:
        x_list, y_list = [], []
        for batch_g, labels in dataloader:
            batch_g = batch_g.to(device)
            feat = batch_g.ndata["attr"]
            L = build_laplacian(batch_g, device)
            diffused_x = precompute_laplacian_diffusion(feat, L, cfg_T)
            out = model.embed(batch_g, feat, t_val, diffused_x)
            out = pooler(batch_g, out)
            x_list.append(out)
            y_list.append(labels)
        embed_list.append(torch.cat(x_list, dim=0).cpu().numpy())
        y_all = torch.cat(y_list, dim=0).cpu().numpy()

    mean_f1, std_f1 = evaluate_embeddings_svm(embed_list, y_all, T_list)
    logger.info(f"#Test_f1: {mean_f1:.4f}±{std_f1:.4f}")
    return mean_f1


@torch.no_grad()
def evaluate_ddm(model, T_list: list,
                 pooler, dataloader, device, logger) -> float:
    """Evaluate DDM baseline via SVM on pooled embeddings."""
    model.eval()
    embed_list = []
    y_all = None

    for t_val in T_list:
        x_list, y_list = [], []
        for batch_g, labels in dataloader:
            batch_g = batch_g.to(device)
            feat = batch_g.ndata["attr"]
            out = model.embed(batch_g, feat, t_val)
            out = pooler(batch_g, out)
            x_list.append(out)
            y_list.append(labels)
        embed_list.append(torch.cat(x_list, dim=0).cpu().numpy())
        y_all = torch.cat(y_list, dim=0).cpu().numpy()

    mean_f1, std_f1 = evaluate_embeddings_svm(embed_list, y_all, T_list)
    logger.info(f"#Test_f1: {mean_f1:.4f}±{std_f1:.4f}")
    return mean_f1

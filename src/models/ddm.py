"""
DDM – Discrete Diffusion Model baseline.

Standard DDPM-style diffusion over node features with i.i.d. Gaussian noise
(no graph-structure coupling in the forward process).
"""

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .denoising_net import DenoisingUNet
from .hgd import get_beta_schedule, extract, sce_loss


class DDM(nn.Module):
    """
    Baseline graph diffusion model.

    Forward process: x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps,
    where eps is i.i.d. Gaussian with the same sign as x_0.
    """

    def __init__(
        self,
        in_dim: int,
        num_hidden: int,
        num_layers: int,
        nhead: int,
        activation: str = "prelu",
        feat_drop: float = 0.2,
        attn_drop: float = 0.1,
        norm: Optional[str] = "layernorm",
        alpha_l: float = 2.0,
        beta_schedule: str = "linear",
        beta_1: float = 1e-4,
        beta_T: float = 0.02,
        T: int = 1000,
        **kwargs,
    ):
        super().__init__()
        self.T = T
        self.alpha_l = alpha_l

        betas = get_beta_schedule(beta_schedule, beta_1, beta_T, T)
        alphas = 1.0 - betas
        alphas_bar = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("sqrt_alphas_bar", torch.sqrt(alphas_bar))
        self.register_buffer("sqrt_one_minus_alphas_bar", torch.sqrt(1.0 - alphas_bar))

        assert num_hidden % nhead == 0
        self.net = DenoisingUNet(
            in_dim=in_dim,
            num_hidden=num_hidden,
            out_dim=in_dim,
            num_layers=num_layers,
            nhead=nhead,
            activation=activation,
            feat_drop=feat_drop,
            attn_drop=attn_drop,
            negative_slope=0.2,
            norm=norm,
        )
        self.time_embedding = nn.Embedding(T, num_hidden)

    def forward(self, g, x: torch.Tensor):
        with torch.no_grad():
            x = F.layer_norm(x, (x.shape[-1],))

        t = torch.randint(self.T, size=(x.shape[0],), device=x.device)
        x_t, time_embed = self._sample_forward(t, x)

        pred, _ = self.net(g, x_t=x_t, time_embed=time_embed)
        loss = sce_loss(pred, x, self.alpha_l)
        return loss, {"loss": loss.item()}

    def _sample_forward(self, t, x):
        miu, std = x.mean(dim=0), x.std(dim=0)
        noise = torch.randn_like(x)
        noise = F.layer_norm(noise, (noise.shape[-1],))
        noise = noise * std + miu
        noise = torch.sign(x) * torch.abs(noise)
        x_t = (
            extract(self.sqrt_alphas_bar, t, x.shape) * x
            + extract(self.sqrt_one_minus_alphas_bar, t, x.shape) * noise
        )
        time_embed = self.time_embedding(t)
        return x_t, time_embed

    def embed(self, g, x: torch.Tensor, t_embed: int) -> torch.Tensor:
        t = torch.full((x.shape[0],), t_embed, dtype=torch.long, device=x.device)
        with torch.no_grad():
            x = F.layer_norm(x, (x.shape[-1],))
        x_t, time_embed = self._sample_forward(t, x)
        _, hidden = self.net(g, x_t=x_t, time_embed=time_embed)
        return hidden

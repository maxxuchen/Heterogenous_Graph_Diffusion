"""
Heterogeneous Graph Diffusion (HGD)

Forward process uses graph-Laplacian dynamics to diffuse node features
along the graph structure, yielding a structure-aware noising schedule.
The reverse (denoising) process is parameterized by a U-Net over GAT layers.
"""

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .denoising_net import DenoisingUNet


def extract(v, t, x_shape):
    """Index into coefficient buffer at timesteps t and broadcast to x_shape."""
    out = torch.gather(v, index=t, dim=0).float()
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))


def get_beta_schedule(schedule: str, beta_start: float, beta_end: float,
                      num_timesteps: int) -> torch.Tensor:
    """Return a noise schedule as a 1-D tensor of length num_timesteps."""
    def sigmoid(x):
        return 1.0 / (np.exp(-x) + 1.0)

    if schedule == "linear":
        betas = np.linspace(beta_start, beta_end, num_timesteps, dtype=np.float64)
    elif schedule == "quad":
        betas = (np.linspace(beta_start ** 0.5, beta_end ** 0.5,
                             num_timesteps, dtype=np.float64)) ** 2
    elif schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    elif schedule == "const":
        betas = beta_end * np.ones(num_timesteps, dtype=np.float64)
    elif schedule == "jsd":
        betas = 1.0 / np.linspace(num_timesteps, 1, num_timesteps, dtype=np.float64)
    else:
        raise NotImplementedError(f"Unknown beta schedule: {schedule}")
    assert betas.shape == (num_timesteps,)
    return torch.from_numpy(betas)


def sce_loss(pred: torch.Tensor, target: torch.Tensor, alpha: float = 2) -> torch.Tensor:
    """Scaled Cosine Error loss."""
    pred = F.normalize(pred, p=2, dim=-1)
    target = F.normalize(target, p=2, dim=-1)
    return (1 - (pred * target).sum(dim=-1)).pow_(alpha).mean()


def build_laplacian(g, device: torch.device) -> torch.Tensor:
    """Compute the normalized graph Laplacian L = D^{-1}A / n  (as in the paper)."""
    src, dst = g.edges()
    n = g.num_nodes()
    adj = torch.zeros((n, n), device=device)
    adj[src, dst] = 1.0
    epsilon = adj / n
    D = torch.diag(epsilon.sum(dim=1))
    return D - epsilon  # shape (n, n)


def precompute_laplacian_diffusion(x: torch.Tensor, L: torch.Tensor,
                                   T: int) -> list:
    """
    Precompute the Laplacian-diffused versions of x at each timestep.

    Returns a list of length T where entry t holds  (-L)^t @ x_0.
    Index 0 is the identity: Laplace_x[0] = x_0.
    """
    laplace_powers = [torch.eye(L.shape[0], device=L.device)]
    diffused = [x]
    for _ in range(1, T):
        M = torch.matmul(-L, laplace_powers[-1])
        laplace_powers.append(M)
        diffused.append(torch.matmul(M, diffused[-1]))
    return diffused


class HeterogeneousGraphDiffusion(nn.Module):
    """
    Heterogeneous Graph Diffusion model.

    The forward process mixes the Laplacian-diffused signal with Gaussian noise:

        x_t = sqrt(alpha_bar_t) * L^t(x_0)  +  sqrt(1 - alpha_bar_t) * eps

    where  eps ~ N(0, 2 I)  and  L^t(x_0) = (-L)^t x_0.

    The denoising network is a U-Net built from GAT layers.
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
        beta_schedule: str = "sigmoid",
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

        assert num_hidden % nhead == 0, "num_hidden must be divisible by nhead"
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

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def forward(self, g, x: torch.Tensor):
        """Return (loss, loss_dict) for one training step."""
        with torch.no_grad():
            x = F.layer_norm(x, (x.shape[-1],))

        # Build Laplacian and precompute diffused feature sequence
        L = build_laplacian(g, x.device)
        diffused_x = precompute_laplacian_diffusion(x, L, self.T)

        t = torch.randint(self.T, size=(x.shape[0],), device=x.device)
        x_t, time_embed = self._sample_forward(t, x, diffused_x)
        loss = self._denoising_loss(x, x_t, time_embed, g)
        return loss, {"loss": loss.item()}

    def _sample_forward(self, t: torch.Tensor, x: torch.Tensor,
                        diffused_x: list):
        """Sample x_t given the Laplacian-diffused sequence."""
        noise = torch.randn_like(x)
        noise = np.sqrt(2) * F.layer_norm(noise, (noise.shape[-1],))
        noise = torch.sign(x) * torch.abs(noise)
        # Select the diffused feature at timestep t for each node
        diffused = torch.stack([diffused_x[t[i]][i] for i in range(len(t))], dim=0)
        x_t = (
            extract(self.sqrt_alphas_bar, t, x.shape) * diffused
            + extract(self.sqrt_one_minus_alphas_bar, t, x.shape) * noise
        )
        time_embed = self.time_embedding(t)
        return x_t, time_embed

    def _denoising_loss(self, x, x_t, time_embed, g) -> torch.Tensor:
        pred, _ = self.net(g, x_t=x_t, time_embed=time_embed)
        return sce_loss(pred, x, self.alpha_l)

    # ------------------------------------------------------------------
    # Inference / embedding
    # ------------------------------------------------------------------

    def embed(self, g, x: torch.Tensor, t_embed: int,
              diffused_x: list) -> torch.Tensor:
        """
        Produce a graph-diffusion embedding at timestep t_embed.

        Parameters
        ----------
        g        : DGL graph
        x        : node features, shape (N, F)
        t_embed  : diffusion timestep used for embedding
        diffused_x : precomputed list from precompute_laplacian_diffusion
        """
        t = torch.full((x.shape[0],), t_embed, dtype=torch.long, device=x.device)
        with torch.no_grad():
            x = F.layer_norm(x, (x.shape[-1],))
        x_t, time_embed = self._sample_forward(t, x, diffused_x)
        _, hidden = self.net(g, x_t=x_t, time_embed=time_embed)
        return hidden

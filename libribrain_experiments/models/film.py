import torch
import torch.nn as nn


class FiLM(nn.Module):
    """Feature-wise Linear Modulation conditioned on a scalar signal c.

    Applies per-channel affine: FiLM(F | c) = (1 + delta_gamma(c)) * F + beta(c)
    where gamma and beta are broadcast over the time dimension.

    The last linear layer is zero-initialised so the layer starts as identity.
    """

    def __init__(self, num_channels: int, hidden_dim: int = 64, conditioning_dim: int = 1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(conditioning_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * num_channels),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T), c: (B, conditioning_dim)
        params = self.mlp(c)                        # (B, 2C)
        delta_gamma, beta = params.chunk(2, dim=1)  # (B, C) each
        gamma = 1.0 + delta_gamma                   # identity init
        return gamma.unsqueeze(-1) * x + beta.unsqueeze(-1)

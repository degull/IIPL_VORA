import math

import torch
from torch import nn


class LoRANLinear(nn.Module):
    """LoRA-style adapter with a nonlinear transform after the low-rank branch."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float = 1.0,
        dropout: float = 0.0,
        bias: bool = True,
        freeze_base: bool = True,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("rank must be positive.")
        self.scaling = alpha / rank
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.A_l = nn.Linear(in_features, rank, bias=False)
        self.B_l = nn.Linear(rank, out_features, bias=False)
        self.reset_parameters()
        if freeze_base:
            self.freeze_base()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.A_l.weight, a=math.sqrt(5))
        nn.init.zeros_(self.B_l.weight)

    def freeze_base(self) -> None:
        for param in self.linear.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branch = self.B_l(self.A_l(self.dropout(x))) * self.scaling
        return self.linear(x) + torch.tanh(branch)


class AuroRALinear(nn.Module):
    """LoRA-style adapter with a nonlinear hidden mapping between A and B."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float = 1.0,
        dropout: float = 0.0,
        bias: bool = True,
        freeze_base: bool = True,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("rank must be positive.")
        self.scaling = alpha / rank
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.A_l = nn.Linear(in_features, rank, bias=False)
        self.B_l = nn.Linear(rank, out_features, bias=False)
        self.reset_parameters()
        if freeze_base:
            self.freeze_base()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.A_l.weight, a=math.sqrt(5))
        nn.init.zeros_(self.B_l.weight)

    def freeze_base(self) -> None:
        for param in self.linear.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = torch.nn.functional.silu(self.A_l(self.dropout(x)))
        return self.linear(x) + self.B_l(hidden) * self.scaling


class StructuredNonlinearLoRALinear(nn.Module):
    """LoRA-style adapter with a learnable structured nonlinear hidden transform."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float = 1.0,
        dropout: float = 0.0,
        bias: bool = True,
        freeze_base: bool = True,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("rank must be positive.")
        self.scaling = alpha / rank
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.A_l = nn.Linear(in_features, rank, bias=False)
        self.B_l = nn.Linear(rank, out_features, bias=False)
        self.mix_logits = nn.Parameter(torch.zeros(3, rank))
        self.reset_parameters()
        if freeze_base:
            self.freeze_base()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.A_l.weight, a=math.sqrt(5))
        nn.init.zeros_(self.B_l.weight)

    def freeze_base(self) -> None:
        for param in self.linear.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.A_l(self.dropout(x))
        weights = torch.softmax(self.mix_logits, dim=0)
        hidden = weights[0] * z + weights[1] * torch.tanh(z) + weights[2] * torch.sin(z)
        return self.linear(x) + self.B_l(hidden) * self.scaling


class NEATLinear(nn.Module):
    """NEAT-style adapter with a small neural generator over low-rank features."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float = 1.0,
        dropout: float = 0.0,
        bias: bool = True,
        freeze_base: bool = True,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("rank must be positive.")
        self.scaling = alpha / rank
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.A_l = nn.Linear(in_features, rank, bias=False)
        self.B_l = nn.Linear(rank, out_features, bias=False)
        self.generator = nn.Sequential(
            nn.Linear(rank, rank, bias=False),
            nn.SiLU(),
            nn.Linear(rank, rank, bias=False),
        )
        self.reset_parameters()
        if freeze_base:
            self.freeze_base()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.A_l.weight, a=math.sqrt(5))
        nn.init.zeros_(self.B_l.weight)
        for module in self.generator:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)

    def freeze_base(self) -> None:
        for param in self.linear.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.A_l(self.dropout(x))
        gate = torch.sigmoid(self.generator(z))
        return self.linear(x) + self.B_l(z * gate) * self.scaling

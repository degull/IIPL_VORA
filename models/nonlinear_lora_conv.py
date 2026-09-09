import math

import torch
from torch import nn


class LoRANConv2d(nn.Module):
    """1x1 Conv LoRA adapter with a nonlinear transform after the branch."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        rank: int = 8,
        alpha: float = 1.0,
        bias: bool = True,
        freeze_base: bool = True,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("rank must be positive.")
        self.scaling = alpha / rank
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.A_l = nn.Conv2d(in_channels, rank, kernel_size=1, bias=False)
        self.B_l = nn.Conv2d(rank, out_channels, kernel_size=1, bias=False)
        self.reset_parameters()
        if freeze_base:
            self.freeze_base()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.A_l.weight, a=math.sqrt(5))
        nn.init.zeros_(self.B_l.weight)

    def freeze_base(self) -> None:
        for param in self.conv.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branch = self.B_l(self.A_l(x)) * self.scaling
        return self.conv(x) + torch.tanh(branch)


class AuroRAConv2d(nn.Module):
    """1x1 Conv LoRA adapter with a nonlinear hidden mapping between A and B."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        rank: int = 8,
        alpha: float = 1.0,
        bias: bool = True,
        freeze_base: bool = True,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("rank must be positive.")
        self.scaling = alpha / rank
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.A_l = nn.Conv2d(in_channels, rank, kernel_size=1, bias=False)
        self.B_l = nn.Conv2d(rank, out_channels, kernel_size=1, bias=False)
        self.reset_parameters()
        if freeze_base:
            self.freeze_base()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.A_l.weight, a=math.sqrt(5))
        nn.init.zeros_(self.B_l.weight)

    def freeze_base(self) -> None:
        for param in self.conv.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = torch.nn.functional.silu(self.A_l(x))
        return self.conv(x) + self.B_l(hidden) * self.scaling


class StructuredNonlinearLoRAConv2d(nn.Module):
    """1x1 Conv LoRA adapter with a learnable structured nonlinear hidden transform."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        rank: int = 8,
        alpha: float = 1.0,
        bias: bool = True,
        freeze_base: bool = True,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("rank must be positive.")
        self.scaling = alpha / rank
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.A_l = nn.Conv2d(in_channels, rank, kernel_size=1, bias=False)
        self.B_l = nn.Conv2d(rank, out_channels, kernel_size=1, bias=False)
        self.mix_logits = nn.Parameter(torch.zeros(3, rank, 1, 1))
        self.reset_parameters()
        if freeze_base:
            self.freeze_base()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.A_l.weight, a=math.sqrt(5))
        nn.init.zeros_(self.B_l.weight)

    def freeze_base(self) -> None:
        for param in self.conv.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.A_l(x)
        weights = torch.softmax(self.mix_logits, dim=0)
        hidden = weights[0] * z + weights[1] * torch.tanh(z) + weights[2] * torch.sin(z)
        return self.conv(x) + self.B_l(hidden) * self.scaling


class NEATConv2d(nn.Module):
    """NEAT-style 1x1 Conv adapter with a small neural generator over rank channels."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        rank: int = 8,
        alpha: float = 1.0,
        bias: bool = True,
        freeze_base: bool = True,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("rank must be positive.")
        self.scaling = alpha / rank
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.A_l = nn.Conv2d(in_channels, rank, kernel_size=1, bias=False)
        self.B_l = nn.Conv2d(rank, out_channels, kernel_size=1, bias=False)
        self.generator = nn.Sequential(
            nn.Conv2d(rank, rank, kernel_size=1, bias=False),
            nn.SiLU(),
            nn.Conv2d(rank, rank, kernel_size=1, bias=False),
        )
        self.reset_parameters()
        if freeze_base:
            self.freeze_base()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.A_l.weight, a=math.sqrt(5))
        nn.init.zeros_(self.B_l.weight)
        for module in self.generator:
            if isinstance(module, nn.Conv2d):
                nn.init.xavier_uniform_(module.weight)

    def freeze_base(self) -> None:
        for param in self.conv.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.A_l(x)
        gate = torch.sigmoid(self.generator(z))
        return self.conv(x) + self.B_l(z * gate) * self.scaling

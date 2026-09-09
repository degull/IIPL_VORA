from __future__ import annotations

from dataclasses import dataclass

from torch import nn

from models.ablation_conv import LinearVolterraConv2d, VolterraOnlyConv2d
from models.ablation_linear import LinearVolterraLinear, VolterraOnlyLinear
from models.lora_conv import LoRAConv2d
from models.lora_linear import LoRALinear
from models.nonlinear_lora_conv import AuroRAConv2d, LoRANConv2d, NEATConv2d, StructuredNonlinearLoRAConv2d
from models.nonlinear_lora_linear import AuroRALinear, LoRANLinear, NEATLinear, StructuredNonlinearLoRALinear
from models.vora_conv import VoRAConv2d
from models.vora_full_linear import VoRAFullLinear, VoRATokenLinear
from models.vora_linear import VoRALinear


@dataclass
class AdapterStats:
    replaced: int = 0
    skipped: int = 0


def freeze_module(module: nn.Module) -> None:
    for param in module.parameters():
        param.requires_grad = False


def _copy_linear(
    source: nn.Linear,
    target: (
        LoRALinear
        | VoRALinear
        | VoRATokenLinear
        | VoRAFullLinear
        | VolterraOnlyLinear
        | LinearVolterraLinear
        | LoRANLinear
        | AuroRALinear
        | NEATLinear
        | StructuredNonlinearLoRALinear
    ),
) -> None:
    target.linear.weight.data.copy_(source.weight.data)
    if source.bias is not None and target.linear.bias is not None:
        target.linear.bias.data.copy_(source.bias.data)


def _copy_conv1x1(
    source: nn.Conv2d,
    target: (
        LoRAConv2d
        | VoRAConv2d
        | VolterraOnlyConv2d
        | LinearVolterraConv2d
        | LoRANConv2d
        | AuroRAConv2d
        | NEATConv2d
        | StructuredNonlinearLoRAConv2d
    ),
) -> None:
    target.conv.weight.data.copy_(source.weight.data)
    if source.bias is not None and target.conv.bias is not None:
        target.conv.bias.data.copy_(source.bias.data)


def replace_linear_adapters(
    module: nn.Module,
    method: str,
    target_keywords: tuple[str, ...] = ("qkv", "proj", "mlp"),
    rank: int = 4,
    prefix: str = "",
    include_conv1x1: bool = True,
) -> AdapterStats:
    stats = AdapterStats()
    for name, child in list(module.named_children()):
        full_name = f"{prefix}.{name}" if prefix else name
        if isinstance(child, nn.Linear):
            if any(keyword in full_name for keyword in target_keywords):
                if method == "lora":
                    replacement = LoRALinear(child.in_features, child.out_features, rank=rank, bias=child.bias is not None)
                elif method == "loran":
                    replacement = LoRANLinear(child.in_features, child.out_features, rank=rank, bias=child.bias is not None)
                elif method == "aurora":
                    replacement = AuroRALinear(child.in_features, child.out_features, rank=rank, bias=child.bias is not None)
                elif method == "neat":
                    replacement = NEATLinear(child.in_features, child.out_features, rank=rank, bias=child.bias is not None)
                elif method == "structured_nonlinear_lora":
                    replacement = StructuredNonlinearLoRALinear(
                        child.in_features,
                        child.out_features,
                        rank=rank,
                        bias=child.bias is not None,
                    )
                elif method == "vora_v1":
                    replacement = VoRALinear(
                        child.in_features,
                        child.out_features,
                        lora_rank=rank,
                        volterra_rank=rank,
                        bias=child.bias is not None,
                    )
                elif method == "vora_token":
                    replacement = VoRATokenLinear(
                        child.in_features,
                        child.out_features,
                        lora_rank=rank,
                        volterra_rank=rank,
                        bias=child.bias is not None,
                    )
                elif method == "vora_full":
                    replacement = VoRAFullLinear(
                        child.in_features,
                        child.out_features,
                        lora_rank=rank,
                        volterra_rank=rank,
                        num_experts=4,
                        bias=child.bias is not None,
                    )
                elif method == "volterra_only":
                    replacement = VolterraOnlyLinear(
                        child.in_features,
                        child.out_features,
                        rank=rank,
                        bias=child.bias is not None,
                    )
                elif method == "lora_linear_volterra":
                    replacement = LinearVolterraLinear(
                        child.in_features,
                        child.out_features,
                        lora_rank=rank,
                        volterra_rank=rank,
                        bias=child.bias is not None,
                    )
                else:
                    raise ValueError(f"Unsupported adapter method: {method}")
                _copy_linear(child, replacement)
                setattr(module, name, replacement)
                stats.replaced += 1
            else:
                stats.skipped += 1
            continue

        if include_conv1x1 and isinstance(child, nn.Conv2d) and child.kernel_size == (1, 1):
            if any(keyword in full_name for keyword in target_keywords):
                if method == "lora":
                    conv_replacement = LoRAConv2d(
                        child.in_channels,
                        child.out_channels,
                        rank=rank,
                        bias=child.bias is not None,
                    )
                elif method == "loran":
                    conv_replacement = LoRANConv2d(
                        child.in_channels,
                        child.out_channels,
                        rank=rank,
                        bias=child.bias is not None,
                    )
                elif method == "aurora":
                    conv_replacement = AuroRAConv2d(
                        child.in_channels,
                        child.out_channels,
                        rank=rank,
                        bias=child.bias is not None,
                    )
                elif method == "neat":
                    conv_replacement = NEATConv2d(
                        child.in_channels,
                        child.out_channels,
                        rank=rank,
                        bias=child.bias is not None,
                    )
                elif method == "structured_nonlinear_lora":
                    conv_replacement = StructuredNonlinearLoRAConv2d(
                        child.in_channels,
                        child.out_channels,
                        rank=rank,
                        bias=child.bias is not None,
                    )
                elif method == "vora_v1":
                    conv_replacement = VoRAConv2d(
                        child.in_channels,
                        child.out_channels,
                        lora_rank=rank,
                        volterra_rank=rank,
                        bias=child.bias is not None,
                    )
                elif method == "volterra_only":
                    conv_replacement = VolterraOnlyConv2d(
                        child.in_channels,
                        child.out_channels,
                        rank=rank,
                        bias=child.bias is not None,
                    )
                elif method == "lora_linear_volterra":
                    conv_replacement = LinearVolterraConv2d(
                        child.in_channels,
                        child.out_channels,
                        lora_rank=rank,
                        volterra_rank=rank,
                        bias=child.bias is not None,
                    )
                else:
                    stats.skipped += 1
                    continue
                _copy_conv1x1(child, conv_replacement)
                setattr(module, name, conv_replacement)
                stats.replaced += 1
            else:
                stats.skipped += 1
            continue

        child_stats = replace_linear_adapters(
            child,
            method,
            target_keywords=target_keywords,
            rank=rank,
            prefix=full_name,
            include_conv1x1=include_conv1x1,
        )
        stats.replaced += child_stats.replaced
        stats.skipped += child_stats.skipped
    return stats

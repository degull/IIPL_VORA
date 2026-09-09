from __future__ import annotations

import importlib.util
from pathlib import Path

from torch import nn


def _load_swinir_class():
    repo_root = Path(__file__).resolve().parents[2]
    network_path = repo_root / "external" / "SwinIR" / "models" / "network_swinir.py"
    if not network_path.exists():
        raise FileNotFoundError(
            f"Official SwinIR code was not found at {network_path}. "
            "Clone https://github.com/JingyunLiang/SwinIR.git into external/SwinIR first."
        )

    spec = importlib.util.spec_from_file_location("official_swinir_network", network_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load SwinIR module from {network_path}.")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.SwinIR


def build_official_swinir(size: str = "tiny", img_size: int = 64) -> nn.Module:
    """Build official SwinIR for same-resolution image restoration.

    `tiny` is for smoke tests and development. `small` and `base` are closer to
    paper-scale experiments, but should be trained on GPU.
    """

    SwinIR = _load_swinir_class()

    configs = {
        "tiny": dict(embed_dim=48, depths=[2, 2], num_heads=[3, 3], window_size=8, mlp_ratio=2.0),
        "small": dict(embed_dim=60, depths=[4, 4, 4, 4], num_heads=[4, 4, 4, 4], window_size=8, mlp_ratio=2.0),
        "base": dict(embed_dim=96, depths=[6, 6, 6, 6], num_heads=[6, 6, 6, 6], window_size=8, mlp_ratio=4.0),
        "color_dn": dict(
            embed_dim=180,
            depths=[6, 6, 6, 6, 6, 6],
            num_heads=[6, 6, 6, 6, 6, 6],
            window_size=8,
            mlp_ratio=2.0,
            img_size=128,
        ),
    }
    if size not in configs:
        choices = ", ".join(sorted(configs))
        raise ValueError(f"Unknown SwinIR size '{size}'. Available: {choices}.")

    config = configs[size].copy()
    resolved_img_size = config.pop("img_size", img_size)

    return SwinIR(
        img_size=resolved_img_size,
        patch_size=1,
        in_chans=3,
        upscale=1,
        img_range=1.0,
        upsampler="",
        resi_connection="1conv",
        **config,
    )

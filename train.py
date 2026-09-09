import argparse
import csv
import os
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.composite_degradation_dataset import COMPOSITE_CHOICES, CompositeDegradationDataset
from datasets.paired_image_dataset import PairedImageDataset
from engine.metrics import psnr, simple_ssim
from models.restoration.official_adair import build_official_adair
from models.restoration.official_dfpir_restormer import build_official_dfpir_restormer
from models.restoration.official_hat import build_official_hat
from models.restoration.official_mambairv2 import build_official_mambairv2
from models.restoration.official_swinir import build_official_swinir
from models.restoration.official_uformer import build_official_uformer
from models.restoration.swinir_lite import SwinIRLite
from utils.adapter_utils import freeze_module, replace_linear_adapters
from utils.model_utils import count_parameters


DATASET_CHOICES = ("rain100h", "csd", "gopro", "reside6k", "sidd")
METHOD_CHOICES = (
    "full_ft",
    "frozen",
    "lora",
    "vora_v1",
    "vora_token",
    "vora_full",
    "volterra_only",
    "lora_linear_volterra",
    "loran",
    "aurora",
    "neat",
    "structured_nonlinear_lora",
)
BACKBONE_CHOICES = (
    "swinir_lite",
    "swinir_official",
    "uformer",
    "hat",
    "mambairv2",
    "adair",
    "dfpir_restormer",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train VoRA adapters on paired image restoration datasets.")
    parser.add_argument("--dataset", default="rain100h", choices=DATASET_CHOICES)
    parser.add_argument("--data-root", default=r"E:\restormer+volterra\data")
    parser.add_argument("--method", default="vora_v1", choices=METHOD_CHOICES)
    parser.add_argument("--backbone", default="swinir_lite", choices=BACKBONE_CHOICES)
    parser.add_argument("--swinir-size", default="tiny", choices=("tiny", "small", "base", "color_dn"))
    parser.add_argument("--backbone-size", default="", choices=("", "tiny", "base"))
    parser.add_argument("--target", default="all", choices=("attn", "mlp", "all"))
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--source-dataset", default="", help="Source dataset used to train the frozen base model.")
    parser.add_argument("--base-checkpoint", default="", help="Load these base-model weights before inserting adapters.")
    parser.add_argument("--added-degradation", default="none", choices=COMPOSITE_CHOICES)
    parser.add_argument("--degradation-intensity", type=float, default=-1.0)
    parser.add_argument("--crop-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max-train-samples", type=int, default=16, help="Use 0 for the full train split.")
    parser.add_argument("--max-val-samples", type=int, default=4, help="Use 0 for the full validation split.")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--results-csv", default="outputs/logs/table1_main_comparison.csv")
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--save-interval", type=int, default=5000, help="Save latest checkpoint every N steps. Use 0 to disable.")
    parser.add_argument("--eval-interval", type=int, default=0, help="Evaluate and save best checkpoint every N steps. Use 0 for final eval only.")
    parser.add_argument("--resume", default="", help="Path to a checkpoint to resume from.")
    parser.add_argument("--auto-resume", action="store_true", help="Resume from this run's latest checkpoint if it exists.")
    return parser.parse_args()


def target_keywords(target: str) -> tuple[str, ...]:
    if target == "attn":
        return ("attn.qkv", "attn.proj")
    if target == "mlp":
        return ("mlp",)
    return ("attn.qkv", "attn.proj", "mlp")


def backbone_size(args: argparse.Namespace) -> str:
    return args.backbone_size or args.swinir_size


def adapter_keywords_for_backbone(args: argparse.Namespace) -> tuple[str, ...]:
    if args.backbone == "adair":
        if args.target == "attn":
            return ("attn", "qkv", "project_out")
        if args.target == "mlp":
            return ("project_in", "project_out", "reduce_chan")
        return ("project_in", "project_out", "reduce_chan", "attn", "qkv")
    if args.backbone == "dfpir_restormer":
        if args.target == "attn":
            return ("qkv", "project_out")
        if args.target == "mlp":
            return ("project_in", "project_out", "reduce_chan")
        return ("qkv", "project_in", "project_out", "reduce_chan")
    return target_keywords(args.target)


def build_restoration_model(args: argparse.Namespace) -> tuple[nn.Module, int]:
    if args.backbone == "swinir_lite":
        model = SwinIRLite(embed_dim=48, depth=2, num_heads=4, window_size=8)
    elif args.backbone == "swinir_official":
        model = build_official_swinir(size=args.swinir_size, img_size=args.crop_size)
    elif args.backbone == "uformer":
        model = build_official_uformer(size=backbone_size(args), img_size=args.crop_size)
    elif args.backbone == "hat":
        model = build_official_hat(size=backbone_size(args), img_size=args.crop_size)
    elif args.backbone == "mambairv2":
        model = build_official_mambairv2(size=backbone_size(args), img_size=args.crop_size)
    elif args.backbone == "adair":
        model = build_official_adair(size=backbone_size(args), img_size=args.crop_size)
    elif args.backbone == "dfpir_restormer":
        model = build_official_dfpir_restormer(size=backbone_size(args), img_size=args.crop_size)
    else:
        raise ValueError(f"Unsupported backbone: {args.backbone}")
    replaced = 0

    base_checkpoint = getattr(args, "base_checkpoint", "")
    if base_checkpoint:
        missing, unexpected = load_model_weights(Path(base_checkpoint), model, args.device)
        print(f"Loaded base checkpoint: {base_checkpoint}")
        if missing:
            print(f"Base checkpoint missing keys: {len(missing)}")
        if unexpected:
            print(f"Base checkpoint unexpected keys: {len(unexpected)}")

    if args.method == "frozen":
        freeze_module(model)
    elif args.method in {
        "lora",
        "vora_v1",
        "vora_token",
        "vora_full",
        "volterra_only",
        "lora_linear_volterra",
        "loran",
        "aurora",
        "neat",
        "structured_nonlinear_lora",
    }:
        freeze_module(model)
        stats = replace_linear_adapters(
            model,
            method=args.method,
            target_keywords=adapter_keywords_for_backbone(args),
            rank=args.rank,
        )
        replaced = stats.replaced
    elif args.method == "full_ft":
        pass
    else:
        raise ValueError(f"Unsupported method: {args.method}")

    return model, replaced


def _checkpoint_state_dict(checkpoint: object) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in ("model", "state_dict", "params"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value
        if all(isinstance(key, str) for key in checkpoint):
            return checkpoint  # type: ignore[return-value]
    raise ValueError("Checkpoint does not contain a model state dict.")


def _strip_known_prefixes(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    stripped = {}
    for key, value in state_dict.items():
        new_key = key
        for prefix in ("module.", "model."):
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix) :]
        stripped[new_key] = value
    return stripped


def load_model_weights(path: Path, model: nn.Module, device: str) -> tuple[list[str], list[str]]:
    checkpoint = torch.load(path, map_location=device)
    state_dict = _strip_known_prefixes(_checkpoint_state_dict(checkpoint))
    incompatible = model.load_state_dict(state_dict, strict=False)
    return list(incompatible.missing_keys), list(incompatible.unexpected_keys)


def evaluate(model: nn.Module, loader: DataLoader, device: str) -> tuple[float, float]:
    model.eval()
    psnr_values = []
    ssim_values = []
    with torch.no_grad():
        for batch in loader:
            degraded = batch["input"].to(device)
            clean = batch["target"].to(device)
            pred = model(degraded)
            psnr_values.append(psnr(pred, clean))
            ssim_values.append(simple_ssim(pred, clean))
    return sum(psnr_values) / len(psnr_values), sum(ssim_values) / len(ssim_values)


def append_result(args: argparse.Namespace, row: dict[str, str | int | float]) -> None:
    csv_path = Path(args.results_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset",
        "source_dataset",
        "base_checkpoint",
        "backbone",
        "swinir_size",
        "method",
        "target",
        "rank",
        "added_degradation",
        "degradation_intensity",
        "steps",
        "train_pairs",
        "val_pairs",
        "replaced_layers",
        "trainable_params",
        "psnr",
        "ssim",
        "gpu_mem_mb",
        "best_checkpoint",
        "latest_checkpoint",
    ]
    write_header = not csv_path.exists()
    if csv_path.exists():
        with csv_path.open("r", newline="") as handle:
            reader = csv.reader(handle)
            existing_header = next(reader, None)
        if existing_header:
            fieldnames = existing_header
    with csv_path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in fieldnames})


def print_result_table(row: dict[str, str | int | float]) -> None:
    print("")
    print("Method | Dataset | Trainable Params | PSNR (higher) | SSIM (higher) | GPU Mem MB (lower)")
    print("--- | --- | ---: | ---: | ---: | ---:")
    print(
        f"{row['method']} | {row['dataset']} | {row['trainable_params']:,} | "
        f"{row['psnr']:.4f} | {row['ssim']:.4f} | {row['gpu_mem_mb']}"
    )


def checkpoint_prefix(args: argparse.Namespace) -> Path:
    degradation_suffix = "" if args.added_degradation == "none" else f"_{args.added_degradation}"
    if args.added_degradation != "none" and args.degradation_intensity >= 0:
        intensity_text = str(args.degradation_intensity).replace(".", "p")
        degradation_suffix += f"_i{intensity_text}"
    name = (
        f"{args.backbone}_{args.swinir_size}_{args.dataset}_"
        f"{args.method}_{args.target}_r{args.rank}{degradation_suffix}"
    )
    source_dataset = getattr(args, "source_dataset", "")
    if source_dataset:
        name = (
            f"{args.backbone}_{args.swinir_size}_{source_dataset}_to_{args.dataset}_"
            f"{args.method}_{args.target}_r{args.rank}{degradation_suffix}"
        )
    return Path(args.checkpoint_dir) / name


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    args: argparse.Namespace,
    step: int,
    psnr_value: float | None = None,
    ssim_value: float | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "args": vars(args),
        "step": step,
        "psnr": psnr_value,
        "ssim": ssim_value,
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    tmp_path = path.with_name(path.name + ".tmp")
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    device: str = "cpu",
) -> int:
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    return int(checkpoint.get("step", 0))


def main() -> None:
    args = parse_args()
    torch.manual_seed(42)

    dataset_cls = CompositeDegradationDataset if args.added_degradation != "none" else PairedImageDataset
    intensity = None if args.degradation_intensity < 0 else args.degradation_intensity
    train_kwargs = (
        {"added_degradation": args.added_degradation, "degradation_intensity": intensity}
        if args.added_degradation != "none"
        else {}
    )
    train_set = dataset_cls(
        dataset=args.dataset,
        data_root=Path(args.data_root),
        split="train",
        crop_size=args.crop_size,
        max_samples=None if args.max_train_samples == 0 else args.max_train_samples,
        **train_kwargs,
    )
    val_kwargs = (
        {"added_degradation": args.added_degradation, "degradation_intensity": intensity}
        if args.added_degradation != "none"
        else {}
    )
    val_set = dataset_cls(
        dataset=args.dataset,
        data_root=Path(args.data_root),
        split="test",
        crop_size=args.crop_size,
        max_samples=None if args.max_val_samples == 0 else args.max_val_samples,
        **val_kwargs,
    )
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=args.num_workers)

    model, replaced = build_restoration_model(args)
    model = model.to(args.device)
    trainable_params = count_parameters(model, trainable_only=True)
    ckpt_base = checkpoint_prefix(args)
    latest_checkpoint = ckpt_base.with_name(ckpt_base.name + "_latest.pth")
    best_checkpoint = ckpt_base.with_name(ckpt_base.name + "_best.pth")

    print(f"Dataset: {args.dataset}")
    if args.source_dataset:
        print(f"Source dataset: {args.source_dataset}")
    print(f"Backbone: {args.backbone}")
    print(f"Method: {args.method}")
    print(f"Target: {args.target}")
    print(f"Added degradation: {args.added_degradation}")
    print(f"Degradation intensity: {intensity if intensity is not None else 'default'}")
    print(f"Train pairs: {len(train_set)} | Val pairs: {len(val_set)}")
    print(f"Replaced Linear layers: {replaced}")
    print(f"Trainable params: {trainable_params:,}")

    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()

    if trainable_params == 0:
        val_psnr, val_ssim = evaluate(model, val_loader, args.device)
        print(f"Eval only | PSNR {val_psnr:.4f} | SSIM {val_ssim:.4f}")
        gpu_mem = torch.cuda.max_memory_allocated() / (1024**2) if args.device.startswith("cuda") else 0.0
        row = {
            "dataset": args.dataset,
            "source_dataset": args.source_dataset,
            "base_checkpoint": args.base_checkpoint,
            "backbone": args.backbone,
            "swinir_size": args.swinir_size,
            "method": args.method,
            "target": args.target,
            "rank": args.rank,
            "added_degradation": args.added_degradation,
            "degradation_intensity": args.degradation_intensity,
            "steps": 0,
            "train_pairs": len(train_set),
            "val_pairs": len(val_set),
            "replaced_layers": replaced,
            "trainable_params": trainable_params,
            "psnr": val_psnr,
            "ssim": val_ssim,
            "gpu_mem_mb": round(gpu_mem, 2),
            "best_checkpoint": "",
            "latest_checkpoint": "",
        }
        append_result(args, row)
        print_result_table(row)
        return

    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr)
    loss_fn = nn.L1Loss()
    model.train()

    step = 0
    resume_path = Path(args.resume) if args.resume else None
    if args.auto_resume and latest_checkpoint.exists():
        resume_path = latest_checkpoint
    if resume_path is not None and resume_path.exists():
        step = load_checkpoint(resume_path, model, optimizer, args.device)
        print(f"Resumed from {resume_path} at step {step}")

    best_psnr = float("-inf")
    if step >= args.steps:
        print(f"Checkpoint step {step} already reached requested steps {args.steps}. Evaluating only.")
        val_psnr, val_ssim = evaluate(model, val_loader, args.device)
        gpu_mem = torch.cuda.max_memory_allocated() / (1024**2) if args.device.startswith("cuda") else 0.0
        row = {
            "dataset": args.dataset,
            "source_dataset": args.source_dataset,
            "base_checkpoint": args.base_checkpoint,
            "backbone": args.backbone,
            "swinir_size": args.swinir_size,
            "method": args.method,
            "target": args.target,
            "rank": args.rank,
            "added_degradation": args.added_degradation,
            "degradation_intensity": args.degradation_intensity,
            "steps": args.steps,
            "train_pairs": len(train_set),
            "val_pairs": len(val_set),
            "replaced_layers": replaced,
            "trainable_params": trainable_params,
            "psnr": val_psnr,
            "ssim": val_ssim,
            "gpu_mem_mb": round(gpu_mem, 2),
            "best_checkpoint": str(best_checkpoint),
            "latest_checkpoint": str(latest_checkpoint),
        }
        append_result(args, row)
        print_result_table(row)
        return

    progress = tqdm(total=args.steps, initial=step, desc="training")
    while step < args.steps:
        for batch in train_loader:
            degraded = batch["input"].to(args.device)
            clean = batch["target"].to(args.device)

            optimizer.zero_grad(set_to_none=True)
            pred = model(degraded)
            loss = loss_fn(pred, clean)
            loss.backward()
            optimizer.step()

            step += 1
            progress.update(1)
            progress.set_postfix(loss=f"{loss.item():.4f}")
            if args.save_interval > 0 and step % args.save_interval == 0:
                save_checkpoint(latest_checkpoint, model, optimizer, args, step)
            if args.eval_interval > 0 and step % args.eval_interval == 0:
                val_psnr, val_ssim = evaluate(model, val_loader, args.device)
                model.train()
                if val_psnr > best_psnr:
                    best_psnr = val_psnr
                    save_checkpoint(best_checkpoint, model, optimizer, args, step, val_psnr, val_ssim)
            if step >= args.steps:
                break
    progress.close()

    val_psnr, val_ssim = evaluate(model, val_loader, args.device)
    print(f"Validation | PSNR {val_psnr:.4f} | SSIM {val_ssim:.4f}")
    save_checkpoint(latest_checkpoint, model, optimizer, args, step, val_psnr, val_ssim)
    if val_psnr > best_psnr:
        best_psnr = val_psnr
        save_checkpoint(best_checkpoint, model, optimizer, args, step, val_psnr, val_ssim)
    gpu_mem = torch.cuda.max_memory_allocated() / (1024**2) if args.device.startswith("cuda") else 0.0
    row = {
        "dataset": args.dataset,
        "source_dataset": args.source_dataset,
        "base_checkpoint": args.base_checkpoint,
        "backbone": args.backbone,
        "swinir_size": args.swinir_size,
        "method": args.method,
        "target": args.target,
        "rank": args.rank,
        "added_degradation": args.added_degradation,
        "degradation_intensity": args.degradation_intensity,
        "steps": args.steps,
        "train_pairs": len(train_set),
        "val_pairs": len(val_set),
        "replaced_layers": replaced,
        "trainable_params": trainable_params,
        "psnr": val_psnr,
        "ssim": val_ssim,
        "gpu_mem_mb": round(gpu_mem, 2),
        "best_checkpoint": str(best_checkpoint),
        "latest_checkpoint": str(latest_checkpoint),
    }
    append_result(args, row)
    print_result_table(row)


if __name__ == "__main__":
    main()

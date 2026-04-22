from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from efficientnet_b0_model import GenderEfficientNet
from gender_dataset import EvalTransform, GenderDataset, TrainTransform
from training_utils import (
    EarlyStopping,
    TrainingConfig,
    create_loss_fn,
    create_optimizer,
    create_scheduler,
    evaluate,
    get_device,
    load_checkpoint,
    save_checkpoint,
    train_one_epoch,
)


@dataclass
class Phase2Config:
    manifest_path: str = "outputs/data_pipeline/train_val_test_split.csv"
    phase1_checkpoint_path: str = "outputs/checkpoints/phase1/phase1_best.pt"
    output_dir: str = "outputs/training/phase2"
    checkpoint_dir: str = "outputs/checkpoints/phase2"
    epochs: int = 15
    batch_size: int = 16
    num_workers: int = 0
    seed: int = 42
    unfreeze_from_feature_block: int = 6
    training: TrainingConfig = field(
        default_factory=lambda: TrainingConfig(
            lr=1e-4,
            weight_decay=1e-4,
            gradient_clip_norm=1.0,
            scheduler_name="reduce_on_plateau",
            scheduler_patience=2,
            scheduler_factor=0.5,
            scheduler_min_lr=1e-6,
            early_stopping_patience=5,
            early_stopping_min_delta=1e-4,
        )
    )


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_indices(df: object, split_name: str) -> list[int]:
    split_series = df["split"].astype(str)
    indices = split_series[split_series == split_name].index.tolist()
    if not indices:
        raise ValueError(f"No samples found for split='{split_name}'")
    return indices


def build_dataloaders(
    config: Phase2Config,
) -> tuple[DataLoader, DataLoader, float, dict[str, int]]:
    train_dataset = GenderDataset(
        manifest_path=config.manifest_path,
        transform=TrainTransform(),
        include_only_quality_pass=True,
    )
    val_dataset = GenderDataset(
        manifest_path=config.manifest_path,
        transform=EvalTransform(),
        include_only_quality_pass=True,
    )

    train_indices = _build_indices(train_dataset.df, "train")
    val_indices = _build_indices(val_dataset.df, "val")

    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(val_dataset, val_indices)

    train_loader = DataLoader(
        train_subset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    train_labels = train_dataset.df.iloc[train_indices]["label"].to_numpy(
        dtype=np.int64
    )
    pos_count = int((train_labels == 1).sum())
    neg_count = int((train_labels == 0).sum())
    if pos_count == 0:
        raise ValueError("Train split has no positive samples")
    pos_weight = neg_count / pos_count

    split_counts = {
        "train": len(train_indices),
        "val": len(val_indices),
        "train_male": pos_count,
        "train_female": neg_count,
    }
    return train_loader, val_loader, float(pos_weight), split_counts


def unfreeze_top_backbone_layers(
    model: GenderEfficientNet, unfreeze_from_feature_block: int
) -> int:
    for param in model.backbone.features.parameters():
        param.requires_grad = False

    features = list(model.backbone.features.children())
    if unfreeze_from_feature_block < 0 or unfreeze_from_feature_block >= len(features):
        raise ValueError(
            f"unfreeze_from_feature_block must be in [0, {len(features) - 1}]"
        )

    trainable_params = 0
    for idx, block in enumerate(features):
        if idx >= unfreeze_from_feature_block:
            for param in block.parameters():
                param.requires_grad = True
                trainable_params += param.numel()

    for param in model.backbone.classifier.parameters():
        param.requires_grad = True
        trainable_params += param.numel()

    return trainable_params


def main() -> None:
    config = Phase2Config()
    set_seed(config.seed)

    output_dir = Path(config.output_dir)
    ckpt_dir = Path(config.checkpoint_dir)
    phase1_checkpoint_path = Path(config.phase1_checkpoint_path)

    if not phase1_checkpoint_path.exists():
        raise FileNotFoundError(
            f"Missing Phase 1 checkpoint: {phase1_checkpoint_path}. "
            "Run train_phase1.py first."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    train_loader, val_loader, pos_weight, split_counts = build_dataloaders(config)

    model = GenderEfficientNet(pretrained=True, dropout=0.3, freeze_backbone=True).to(
        device
    )
    phase1_checkpoint = load_checkpoint(
        path=phase1_checkpoint_path,
        model=model,
        optimizer=None,
        scheduler=None,
        map_location=device,
    )

    trainable_params = unfreeze_top_backbone_layers(
        model=model,
        unfreeze_from_feature_block=config.unfreeze_from_feature_block,
    )

    loss_fn = create_loss_fn(pos_weight=pos_weight, device=device)
    optimizer = create_optimizer(
        model,
        lr=config.training.lr,
        weight_decay=config.training.weight_decay,
    )
    scheduler = create_scheduler(
        optimizer,
        scheduler_name=config.training.scheduler_name,
        scheduler_patience=config.training.scheduler_patience,
        scheduler_factor=config.training.scheduler_factor,
        scheduler_min_lr=config.training.scheduler_min_lr,
        cosine_t_max=config.training.cosine_t_max,
    )
    early_stopper = EarlyStopping(
        patience=config.training.early_stopping_patience,
        min_delta=config.training.early_stopping_min_delta,
        mode="min",
    )

    history: list[dict[str, float | int | bool]] = []
    best_val_loss = float("inf")
    best_epoch = 0

    print(f"Starting Phase 2 fine-tuning on device={device}")
    print(f"Loaded Phase 1 checkpoint: {phase1_checkpoint_path}")
    print(
        "Split counts: "
        f"train={split_counts['train']} val={split_counts['val']} "
        f"train_male={split_counts['train_male']} train_female={split_counts['train_female']}"
    )
    print(
        f"Unfreezing feature blocks >= {config.unfreeze_from_feature_block}; "
        f"trainable parameters={trainable_params}"
    )
    print(f"Using pos_weight={pos_weight:.6f}")

    for epoch in range(1, config.epochs + 1):
        train_metrics = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            gradient_clip_norm=config.training.gradient_clip_norm,
        )
        val_metrics = evaluate(
            model=model,
            dataloader=val_loader,
            loss_fn=loss_fn,
            device=device,
            return_outputs=False,
        )

        if scheduler is not None:
            if config.training.scheduler_name == "reduce_on_plateau":
                scheduler.step(val_metrics["loss"])
            else:
                scheduler.step()

        improved = early_stopper.step(float(val_metrics["loss"]))
        is_best = float(val_metrics["loss"]) < best_val_loss
        if is_best:
            best_val_loss = float(val_metrics["loss"])
            best_epoch = epoch
            save_checkpoint(
                path=ckpt_dir / "phase2_best.pt",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                metrics={
                    "train_loss": float(train_metrics["loss"]),
                    "train_accuracy": float(train_metrics["accuracy"]),
                    "val_loss": float(val_metrics["loss"]),
                    "val_accuracy": float(val_metrics["accuracy"]),
                },
                extra={
                    "phase": "phase2",
                    "freeze_backbone": False,
                    "unfreeze_from_feature_block": config.unfreeze_from_feature_block,
                },
            )

        save_checkpoint(
            path=ckpt_dir / "phase2_last.pt",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            metrics={
                "train_loss": float(train_metrics["loss"]),
                "train_accuracy": float(train_metrics["accuracy"]),
                "val_loss": float(val_metrics["loss"]),
                "val_accuracy": float(val_metrics["accuracy"]),
            },
            extra={
                "phase": "phase2",
                "freeze_backbone": False,
                "unfreeze_from_feature_block": config.unfreeze_from_feature_block,
            },
        )

        epoch_row: dict[str, float | int | bool] = {
            "epoch": epoch,
            "train_loss": float(train_metrics["loss"]),
            "train_accuracy": float(train_metrics["accuracy"]),
            "val_loss": float(val_metrics["loss"]),
            "val_accuracy": float(val_metrics["accuracy"]),
            "lr": float(train_metrics["lr"]),
            "is_best": is_best,
            "early_stopping_improved": improved,
        }
        history.append(epoch_row)

        print(
            f"Epoch {epoch:02d}/{config.epochs} "
            f"train_loss={epoch_row['train_loss']:.4f} "
            f"train_acc={epoch_row['train_accuracy']:.4f} "
            f"val_loss={epoch_row['val_loss']:.4f} "
            f"val_acc={epoch_row['val_accuracy']:.4f} "
            f"lr={epoch_row['lr']:.6f}"
        )

        if early_stopper.should_stop:
            print(f"Early stopping triggered at epoch {epoch}")
            break

    phase1_metrics = phase1_checkpoint.get("metrics", {})
    history_payload = {
        "phase": "phase2",
        "device": str(device),
        "seed": config.seed,
        "frozen_backbone": False,
        "unfreeze_from_feature_block": config.unfreeze_from_feature_block,
        "config": {
            "manifest_path": config.manifest_path,
            "phase1_checkpoint_path": config.phase1_checkpoint_path,
            "output_dir": config.output_dir,
            "checkpoint_dir": config.checkpoint_dir,
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "num_workers": config.num_workers,
            "training": asdict(config.training),
        },
        "phase1_reference": {
            "source_checkpoint": config.phase1_checkpoint_path,
            "epoch": int(phase1_checkpoint.get("epoch", 0)),
            "metrics": phase1_metrics,
        },
        "split_counts": split_counts,
        "pos_weight": pos_weight,
        "trainable_params": trainable_params,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "history": history,
    }

    history_path = output_dir / "phase2_history.json"
    with history_path.open("w", encoding="utf-8") as f:
        json.dump(history_payload, f, indent=2)

    print(f"Saved history: {history_path}")
    print(f"Saved best checkpoint: {ckpt_dir / 'phase2_best.pt'}")
    print(f"Saved last checkpoint: {ckpt_dir / 'phase2_last.pt'}")


if __name__ == "__main__":
    main()

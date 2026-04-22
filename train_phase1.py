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
    save_checkpoint,
    train_one_epoch,
)


@dataclass
class Phase1Config:
    manifest_path: str = "outputs/data_pipeline/train_val_test_split.csv"
    output_dir: str = "outputs/training/phase1"
    checkpoint_dir: str = "outputs/checkpoints/phase1"
    epochs: int = 5
    batch_size: int = 16
    num_workers: int = 0
    seed: int = 42
    training: TrainingConfig = field(
        default_factory=lambda: TrainingConfig(
            lr=1e-3,
            weight_decay=1e-4,
            gradient_clip_norm=1.0,
            scheduler_name="reduce_on_plateau",
            scheduler_patience=1,
            scheduler_factor=0.5,
            scheduler_min_lr=1e-6,
            early_stopping_patience=3,
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
    config: Phase1Config,
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


def main() -> None:
    config = Phase1Config()
    set_seed(config.seed)

    output_dir = Path(config.output_dir)
    ckpt_dir = Path(config.checkpoint_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    train_loader, val_loader, pos_weight, split_counts = build_dataloaders(config)

    model = GenderEfficientNet(pretrained=True, dropout=0.3, freeze_backbone=True).to(
        device
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

    print(f"Starting Phase 1 training on device={device}")
    print(
        "Split counts: "
        f"train={split_counts['train']} val={split_counts['val']} "
        f"train_male={split_counts['train_male']} train_female={split_counts['train_female']}"
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
                path=ckpt_dir / "phase1_best.pt",
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
                extra={"phase": "phase1", "freeze_backbone": True},
            )

        save_checkpoint(
            path=ckpt_dir / "phase1_last.pt",
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
            extra={"phase": "phase1", "freeze_backbone": True},
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

    history_payload = {
        "phase": "phase1",
        "device": str(device),
        "seed": config.seed,
        "frozen_backbone": True,
        "config": {
            "manifest_path": config.manifest_path,
            "output_dir": config.output_dir,
            "checkpoint_dir": config.checkpoint_dir,
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "num_workers": config.num_workers,
            "training": asdict(config.training),
        },
        "split_counts": split_counts,
        "pos_weight": pos_weight,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "history": history,
    }

    history_path = output_dir / "phase1_history.json"
    with history_path.open("w", encoding="utf-8") as f:
        json.dump(history_payload, f, indent=2)

    print(f"Saved history: {history_path}")
    print(f"Saved best checkpoint: {ckpt_dir / 'phase1_best.pt'}")
    print(f"Saved last checkpoint: {ckpt_dir / 'phase1_last.pt'}")


if __name__ == "__main__":
    main()

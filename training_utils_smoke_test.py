from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from efficientnet_b0_model import GenderEfficientNet
from gender_dataset import EvalTransform, GenderDataset
from training_utils import (
    EarlyStopping,
    create_loss_fn,
    create_optimizer,
    create_scheduler,
    evaluate,
    get_device,
    save_checkpoint,
    train_one_epoch,
)


def build_split_subsets(
    manifest_path: str | Path,
    max_train: int = 16,
    max_val: int = 16,
) -> tuple[Subset, Subset]:
    dataset = GenderDataset(
        manifest_path=manifest_path,
        transform=EvalTransform(),
        include_only_quality_pass=True,
    )

    split_series = dataset.df["split"].astype(str)
    train_indices = split_series[split_series == "train"].index.tolist()[:max_train]
    val_indices = split_series[split_series == "val"].index.tolist()[:max_val]

    if not train_indices or not val_indices:
        raise SystemExit("Could not build train/val subsets from manifest split column")

    return Subset(dataset, train_indices), Subset(dataset, val_indices)


def main() -> None:
    device = get_device()
    train_subset, val_subset = build_split_subsets(
        manifest_path="outputs/data_pipeline/train_val_test_split.csv",
    )

    train_loader = DataLoader(train_subset, batch_size=8, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_subset, batch_size=8, shuffle=False, num_workers=0)

    model = GenderEfficientNet(pretrained=True, dropout=0.3, freeze_backbone=True).to(
        device
    )
    loss_fn = create_loss_fn()
    optimizer = create_optimizer(model, lr=1e-3, weight_decay=1e-4)
    scheduler = create_scheduler(
        optimizer,
        scheduler_name="reduce_on_plateau",
        scheduler_patience=1,
        scheduler_factor=0.5,
        scheduler_min_lr=1e-6,
    )
    early_stopper = EarlyStopping(patience=2, min_delta=1e-4, mode="min")

    train_metrics = train_one_epoch(
        model=model,
        dataloader=train_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        gradient_clip_norm=1.0,
    )
    val_metrics = evaluate(
        model=model,
        dataloader=val_loader,
        loss_fn=loss_fn,
        device=device,
        return_outputs=True,
    )

    scheduler.step(val_metrics["loss"])
    improved = early_stopper.step(float(val_metrics["loss"]))

    checkpoint_path = Path("outputs/checkpoints/training_utils_smoke.pt")
    save_checkpoint(
        path=checkpoint_path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=1,
        metrics={
            "train_loss": float(train_metrics["loss"]),
            "val_loss": float(val_metrics["loss"]),
            "train_accuracy": float(train_metrics["accuracy"]),
            "val_accuracy": float(val_metrics["accuracy"]),
        },
        extra={"device": str(device)},
    )

    if not torch.isfinite(torch.tensor(train_metrics["loss"])):
        raise SystemExit("Train loss is non-finite")
    if not torch.isfinite(torch.tensor(val_metrics["loss"])):
        raise SystemExit("Val loss is non-finite")

    print("Training utilities smoke test passed.")
    print(f"Device: {device}")
    print(
        f"Train: loss={train_metrics['loss']:.6f}, "
        f"acc={train_metrics['accuracy']:.4f}, lr={train_metrics['lr']:.6f}"
    )
    print(
        f"Val: loss={val_metrics['loss']:.6f}, "
        f"acc={val_metrics['accuracy']:.4f}, samples={int(val_metrics['num_samples'])}"
    )
    print(f"EarlyStopping improved={improved}, should_stop={early_stopper.should_stop}")
    print(f"Checkpoint saved: {checkpoint_path}")


if __name__ == "__main__":
    main()

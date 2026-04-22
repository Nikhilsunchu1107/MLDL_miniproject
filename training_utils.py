from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.data import DataLoader


@dataclass
class TrainingConfig:
    lr: float = 1e-3
    weight_decay: float = 1e-4
    gradient_clip_norm: float | None = None
    scheduler_name: str | None = None
    scheduler_patience: int = 2
    scheduler_factor: float = 0.5
    scheduler_min_lr: float = 1e-6
    cosine_t_max: int = 10
    early_stopping_patience: int = 5
    early_stopping_min_delta: float = 0.0


class EarlyStopping:
    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 0.0,
        mode: str = "min",
    ) -> None:
        if mode not in {"min", "max"}:
            raise ValueError("mode must be 'min' or 'max'")

        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_value: float | None = None
        self.num_bad_epochs = 0
        self.should_stop = False

    def _is_improvement(self, value: float) -> bool:
        if self.best_value is None:
            return True
        if self.mode == "min":
            return value < (self.best_value - self.min_delta)
        return value > (self.best_value + self.min_delta)

    def step(self, value: float) -> bool:
        if self._is_improvement(value):
            self.best_value = value
            self.num_bad_epochs = 0
            self.should_stop = False
            return True

        self.num_bad_epochs += 1
        self.should_stop = self.num_bad_epochs >= self.patience
        return False


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_loss_fn(
    pos_weight: float | None = None, device: torch.device | None = None
) -> nn.Module:
    if pos_weight is None:
        return nn.BCEWithLogitsLoss()

    weight_tensor = torch.tensor([pos_weight], dtype=torch.float32)
    if device is not None:
        weight_tensor = weight_tensor.to(device)
    return nn.BCEWithLogitsLoss(pos_weight=weight_tensor)


def create_optimizer(
    model: nn.Module,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
) -> Optimizer:
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise ValueError("No trainable parameters found for optimizer")
    return AdamW(params, lr=lr, weight_decay=weight_decay)


def create_scheduler(
    optimizer: Optimizer,
    scheduler_name: str | None,
    scheduler_patience: int = 2,
    scheduler_factor: float = 0.5,
    scheduler_min_lr: float = 1e-6,
    cosine_t_max: int = 10,
) -> ReduceLROnPlateau | CosineAnnealingLR | None:
    if scheduler_name is None:
        return None

    normalized = scheduler_name.strip().lower()
    if normalized == "reduce_on_plateau":
        return ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=scheduler_factor,
            patience=scheduler_patience,
            min_lr=scheduler_min_lr,
        )
    if normalized == "cosine":
        return CosineAnnealingLR(
            optimizer, T_max=cosine_t_max, eta_min=scheduler_min_lr
        )

    raise ValueError(
        "Unsupported scheduler_name. Use one of: None, 'reduce_on_plateau', 'cosine'"
    )


def binary_accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    probs = torch.sigmoid(logits)
    preds = (probs >= 0.5).long()
    truth = targets.long()
    return float((preds == truth).float().mean().item())


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    gradient_clip_norm: float | None = None,
) -> dict[str, float]:
    model.train()

    running_loss = 0.0
    running_correct = 0
    total_samples = 0

    for batch_x, batch_y in dataloader:
        batch_x = batch_x.to(device, non_blocking=True)
        targets = batch_y.float().unsqueeze(1).to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(batch_x)
        loss = loss_fn(logits, targets)
        loss.backward()

        if gradient_clip_norm is not None:
            clip_grad_norm_(model.parameters(), max_norm=gradient_clip_norm)

        optimizer.step()

        batch_size = batch_x.size(0)
        running_loss += float(loss.item()) * batch_size
        running_correct += int(
            ((torch.sigmoid(logits) >= 0.5).long() == targets.long()).sum().item()
        )
        total_samples += batch_size

    if total_samples == 0:
        raise ValueError("Empty dataloader: cannot compute training metrics")

    current_lr = float(optimizer.param_groups[0]["lr"])
    return {
        "loss": running_loss / total_samples,
        "accuracy": running_correct / total_samples,
        "lr": current_lr,
        "num_samples": float(total_samples),
    }


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    return_outputs: bool = False,
) -> dict[str, Any]:
    model.eval()

    running_loss = 0.0
    running_correct = 0
    total_samples = 0
    logits_list: list[torch.Tensor] = []
    targets_list: list[torch.Tensor] = []

    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device, non_blocking=True)
            targets = batch_y.float().unsqueeze(1).to(device, non_blocking=True)

            logits = model(batch_x)
            loss = loss_fn(logits, targets)

            batch_size = batch_x.size(0)
            running_loss += float(loss.item()) * batch_size
            running_correct += int(
                ((torch.sigmoid(logits) >= 0.5).long() == targets.long()).sum().item()
            )
            total_samples += batch_size

            if return_outputs:
                logits_list.append(logits.detach().cpu())
                targets_list.append(targets.detach().cpu())

    if total_samples == 0:
        raise ValueError("Empty dataloader: cannot compute evaluation metrics")

    metrics: dict[str, Any] = {
        "loss": running_loss / total_samples,
        "accuracy": running_correct / total_samples,
        "num_samples": float(total_samples),
    }
    if return_outputs:
        metrics["logits"] = torch.cat(logits_list, dim=0)
        metrics["targets"] = torch.cat(targets_list, dim=0)
    return metrics


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: Optimizer | None,
    scheduler: ReduceLROnPlateau | CosineAnnealingLR | None,
    epoch: int,
    metrics: dict[str, float],
    extra: dict[str, Any] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "metrics": metrics,
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler_state_dict"] = scheduler.state_dict()
    if extra is not None:
        payload["extra"] = extra

    torch.save(payload, path)


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: Optimizer | None = None,
    scheduler: ReduceLROnPlateau | CosineAnnealingLR | None = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    checkpoint = torch.load(Path(path), map_location=map_location)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return checkpoint

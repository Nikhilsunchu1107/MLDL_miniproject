from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from efficientnet_b0_model import GenderEfficientNet
from gender_dataset import EvalTransform, GenderDataset


def main() -> None:
    dataset = GenderDataset(
        manifest_path="outputs/data_pipeline/train_val_test_split.csv",
        transform=EvalTransform(),
        include_only_quality_pass=True,
    )
    loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)

    model = GenderEfficientNet(pretrained=True, dropout=0.3, freeze_backbone=False)
    model.train()

    batch_x, batch_y = next(iter(loader))
    logits = model(batch_x)

    shape_ok = tuple(logits.shape) == (batch_x.shape[0], 1)
    finite_ok = bool(torch.isfinite(logits).all())

    if not shape_ok:
        raise SystemExit(
            f"Unexpected logits shape: {tuple(logits.shape)} expected ({batch_x.shape[0]}, 1)"
        )
    if not finite_ok:
        raise SystemExit("Non-finite values found in logits")

    target = batch_y.float().unsqueeze(1)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
    loss.backward()

    trainable_grads = [
        p.grad is not None for p in model.parameters() if p.requires_grad
    ]
    grad_ok = any(trainable_grads)
    if not grad_ok:
        raise SystemExit("No gradients found on trainable parameters")

    print("Model smoke test passed.")
    print(f"Input batch shape: {tuple(batch_x.shape)}")
    print(f"Logits shape: {tuple(logits.shape)}")
    print(f"Loss: {loss.item():.6f}")
    print(
        f"Batch labels distribution (0=female,1=male): "
        f"{batch_y.bincount(minlength=2).tolist()}"
    )


if __name__ == "__main__":
    main()

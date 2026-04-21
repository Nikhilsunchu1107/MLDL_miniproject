# AGENTS.md

## Reality Check (Current Repo State)
- `main.py` is a placeholder and does not run the ML pipeline.
- `README.md` is empty; treat `masterplan.md` and `dataset_audit_report.md` as the project source of truth.
- There are no configured lint/test/typecheck tasks yet (no pytest/ruff/mypy config, no CI workflows).
- Phase 0.1 and 0.2 are implemented in standalone scripts/modules; current active work starts at `progress.md` item 0.3.

## Environment and Commands
- Python version is pinned to 3.11 (`.python-version`, `mise.toml`, `pyproject.toml`).
- Dependency manager is `uv` (`uv.lock` present). Use:
- `uv sync`
- `uv run jupyter notebook`
- `uv run python main.py` (only for scaffold sanity-check; not the training pipeline)
- Installed ML stack includes `torch` and `torchvision` in `pyproject.toml`.

## Current Entrypoints (Do Not Rebuild)
- TIFF-in-JPG robust decoder is already implemented in `image_decode.py` (`robust_decode_image`).
- Phase 0.1 verification command: `uv run python image_decode.py`.
- Preprocessing primitives are implemented in `preprocessing.py`:
- `apply_circular_mask`
- `apply_clahe_on_green_channel`
- `resize_to_224`
- `normalize_imagenet`
- `preprocess_image`
- Phase 0.2 verification command: `uv run python preprocessing_smoke_test.py`.
- Visual sanity outputs are written to `outputs/preprocessing_sanity/`.

## Data Scope (Do Not Guess)
- Use `datasets/Dataset` as the only labeled gender-training set (700 images; 364 male, 336 female).
- Labels come from `datasets/Dataset/Demographics of the participants.xlsx` (image ID to gender mapping).
- Quality filtering metadata is in `datasets/Dataset/Ground Truth.xlsx` and `datasets/Dataset/Individual Quality Assessment.xlsx`.

## Critical Loader Gotcha
- In `datasets/Dataset`, 200 files have `.jpg` names but TIFF encoding (all AMD + DR subset per `dataset_audit_report.md`).
- Do content-based decoding (PIL/OpenCV fallback), not extension-based decoding.

## Preprocessing Validation Rule
- Before touching labeled training data, validate preprocessing on `datasets/retinal-disease-detection-002` (unlabeled; 2254 images).
- This validation dataset is for pipeline robustness only; do not use it for train/val/test metrics.
- Next required task is exactly `progress.md` item 0.3 (run preprocessing on 50-100 images from `retinal-disease-detection-002`).

## Dataset Exclusions for This Project
- Ignore these sources for gender model training/evaluation: `retinal-colorized-oct-images-003`, `fundus-image-registration`, `retina-blood-vessel`, `retinal-vessel-segmentation`, `UK_Biobank_Dataset`, `papila-retinal-fundus-images`.
- Reason: wrong modality, registration duplicates, segmentation assets, low quality/sample size, or unresolved gender code mapping.

## Expected Pipeline Targets (from masterplan)
- Split labeled data stratified 70/15/15 (train/val/test).
- Backbone: EfficientNet-B0 transfer learning with two phases:
- freeze backbone, train head (~5 epochs)
- unfreeze top layers, fine-tune (~15 epochs, lower LR)
- Report metrics: Accuracy, ROC-AUC, F1, confusion matrix; include Grad-CAM visuals and training curves.

## Progress Tracking
- Source of truth for task state is `progress.md`; update it immediately after each completed step with concrete evidence (command + artifact path).

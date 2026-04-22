# Project Progress

Track work in strict `masterplan.md` order. Do not start a later phase until the current phase exit criteria are met.

Status legend: `[ ]` not started, `[-]` in progress, `[x]` done, `[!]` blocked

## Phase 0 - Preprocessing Validation (Gate)

- [x] 0.1 Build a standalone image decode utility with content-based handling for TIFF-in-JPG files.
  - Scope: loading only, no labels or training.
  - Done when: utility successfully opens sample files from `datasets/Dataset` including known TIFF-in-JPG cases.
  - Evidence: `image_decode.py` and `uv run python image_decode.py` smoke test passed on `AMD-001.jpg` (TIFF), `DR-001.jpg` (TIFF), `Healthy-001.jpg` (JPEG).

- [x] 0.2 Implement preprocessing primitives (circular mask, CLAHE on green channel, resize to 224x224, ImageNet normalization).
  - Scope: pure preprocessing functions with visual sanity checks.
  - Done when: functions run end-to-end on single images and output expected tensor/image shapes.
  - Evidence: `preprocessing.py` + `preprocessing_smoke_test.py`; `uv run python preprocessing_smoke_test.py` passed with output tensor shape `(3, 224, 224)` for `AMD-001.jpg`, `DR-001.jpg`, `Healthy-001.jpg`; visuals saved in `outputs/preprocessing_sanity/`.

- [x] 0.3 Validate preprocessing on 50-100 images from `datasets/retinal-disease-detection-002`.
  - Scope: robustness check across mixed resolutions.
  - Done when: no decode/preprocess failures and visual outputs look clinically plausible.
  - Evidence: `validate_preprocessing_batch.py`; `uv run python validate_preprocessing_batch.py` processed 75 images with 0 failures; report at `outputs/preprocessing_validation/validation_report.md`, summary at `outputs/preprocessing_validation/validation_summary.json`, and 10 visual panels in `outputs/preprocessing_validation/`.

- [x] 0.4 Record validation findings and fixes in notebook markdown.
  - Scope: short notes and representative before/after visuals.
  - Done when: Phase 0 evidence is documented and Phase 1 gate is explicitly marked passed.
  - Evidence: `notebook.ipynb` created with sections for 0.1/0.2/0.3, embedded representative visuals from `outputs/preprocessing_sanity/` and `outputs/preprocessing_validation/`, links to `validation_report.md`/`validation_summary.json`, and explicit markdown marker `Phase 0 Gate: PASSED`.

## Phase 1 - Data Pipeline

- [x] 1.1 Parse `datasets/Dataset/Demographics of the participants.xlsx` into `image_id -> gender` mapping.
  - Scope: mapping + basic label integrity checks.
  - Done when: class counts match expected distribution (364 male / 336 female).
  - Evidence: `parse_demographics.py`; `uv run python parse_demographics.py` passed integrity checks with total `700` samples and class counts `male=364`, `female=336`; outputs saved to `outputs/data_pipeline/demographics_mapping.csv`, `outputs/data_pipeline/demographics_mapping.json`, and `outputs/data_pipeline/demographics_mapping_summary.json`.

- [x] 1.2 Parse quality metadata (`Ground Truth.xlsx`, optionally `Individual Quality Assessment.xlsx`) and define filtering rule.
  - Scope: deterministic quality inclusion/exclusion logic.
  - Done when: filtered image list is reproducible and documented.
  - Evidence: `parse_quality_metadata.py`; `uv run python parse_quality_metadata.py` passed integrity checks and produced deterministic filter outputs with `700` total images, `583` included and `117` excluded by `Ground Truth Overall quality`; artifacts at `outputs/data_pipeline/quality_filter_manifest.csv`, `outputs/data_pipeline/quality_filter_summary.json`, and `outputs/data_pipeline/quality_filter_rule.md`.

- [x] 1.3 Build dataset manifest joining file paths, labels, and quality filter outcome.
  - Scope: one tabular source of truth for downstream split/training.
  - Done when: manifest has no missing labels for included samples.
  - Evidence: `build_dataset_manifest.py`; `uv run python build_dataset_manifest.py` produced `700` total rows and `583` quality-passed rows with `0` missing labels among included samples; artifacts at `outputs/data_pipeline/dataset_manifest.csv`, `outputs/data_pipeline/dataset_manifest_included.csv`, and `outputs/data_pipeline/dataset_manifest_summary.json`.

- [x] 1.4 Implement PyTorch `Dataset` class using robust decoder + preprocessing pipeline.
  - Scope: train/eval transform wiring and sample retrieval.
  - Done when: random sample pulls work and visual spot checks (10-20 images) look correct.
  - Evidence: `gender_dataset.py` (`GenderDataset`, `TrainTransform`, `EvalTransform`) and `gender_dataset_smoke_test.py`; `uv run python gender_dataset_smoke_test.py` passed on 20 random eval + train samples with tensor shape `(3, 224, 224)`, dtype `float32`, labels in `{0,1}`, finite values, and visual panels saved to `outputs/dataset_sanity/`.

- [x] 1.5 Create stratified 70/15/15 train/val/test split and persist split assignments.
  - Scope: split only labeled, quality-passed samples.
  - Done when: class balance is preserved across all splits and counts are logged.
  - Evidence: `create_train_val_test_split.py`; `uv run python create_train_val_test_split.py` created deterministic stratified splits for `583` quality-passed samples with counts `train=408`, `val=87`, `test=88` and balanced class distribution (`train: male=213 female=195`, `val: male=46 female=41`, `test: male=46 female=42`); artifacts at `outputs/data_pipeline/train_val_test_split.csv` and `outputs/data_pipeline/split_summary.json`.

## Phase 2 - Model + Training

- [x] 2.1 Implement EfficientNet-B0 model with custom binary head.
  - Scope: model definition only.
  - Done when: forward pass works on a batch from dataloader.
  - Evidence: `efficientnet_b0_model.py` (`GenderEfficientNet`) and `model_smoke_test.py`; `uv run python model_smoke_test.py` passed with input batch shape `(8, 3, 224, 224)`, logits shape `(8, 1)`, finite outputs, and successful backward pass (`BCEWithLogitsLoss` gradient flow confirmed).

- [x] 2.2 Implement training/eval loop utilities (loss, optimizer, scheduler optional, checkpointing, early stopping hooks).
  - Scope: reusable loop code with metric logging.
  - Done when: one short smoke-run epoch completes without errors.
  - Evidence: `training_utils.py` (`TrainingConfig`, `EarlyStopping`, `create_loss_fn`, `create_optimizer`, `create_scheduler`, `train_one_epoch`, `evaluate`, `save_checkpoint`, `load_checkpoint`) and `training_utils_smoke_test.py`; `uv run python training_utils_smoke_test.py` passed on CUDA with one train + val epoch over split subsets, finite metrics (`train_loss=0.727721`, `val_loss=0.670207`), and checkpoint artifact `outputs/checkpoints/training_utils_smoke.pt`.

- [x] 2.3 Run Phase 1 training (frozen backbone, ~5 epochs).
  - Scope: train head only and save best checkpoint.
  - Done when: phase metrics and checkpoint are recorded.
  - Evidence: `train_phase1.py`; `prime-run uv run python train_phase1.py` completed on CUDA with early stopping at epoch 4/5, best validation metrics at epoch 1 (`val_loss=0.6634`, `val_acc=0.4598`), and artifacts saved to `outputs/checkpoints/phase1/phase1_best.pt`, `outputs/checkpoints/phase1/phase1_last.pt`, and `outputs/training/phase1/phase1_history.json`.

- [x] 2.4 Run Phase 2 fine-tuning (unfreeze top layers, ~15 epochs, lower LR).
  - Scope: controlled unfreezing + lower learning rate.
  - Done when: best fine-tuned checkpoint and full training history are saved.
  - Evidence: `train_phase2.py`; `prime-run uv run python train_phase2.py` completed on CUDA with top-block unfreezing (`unfreeze_from_feature_block=6`, trainable params `3157021`), early stopping at epoch 6/15, best validation metrics at epoch 1 (`val_loss=0.6733`, `val_acc=0.5172`), and artifacts saved to `outputs/checkpoints/phase2/phase2_best.pt`, `outputs/checkpoints/phase2/phase2_last.pt`, and `outputs/training/phase2/phase2_history.json`.

## Phase 3 - Evaluation

- [ ] 3.1 Evaluate best model on test split.
  - Scope: Accuracy, ROC-AUC, F1, confusion matrix.
  - Done when: all required metrics are computed and logged in notebook.

- [ ] 3.2 Generate training curves (loss + accuracy for train/val).
  - Scope: clear plots for both phases.
  - Done when: plots are visible in notebook and interpretable.

- [ ] 3.3 Generate Grad-CAM visualizations for representative test samples.
  - Scope: qualitative explainability outputs.
  - Done when: multiple sample visualizations are produced with short interpretation notes.

## Phase 4 - Report / Notebook Finalization

- [ ] 4.1 Structure notebook into clear sections (setup, data, preprocessing, training, evaluation, conclusions).
  - Scope: markdown cleanup and reproducible cell order.
  - Done when: notebook can be run top-to-bottom without narrative gaps.

- [ ] 4.2 Add final results tables and figures.
  - Scope: consolidated metrics table + key visuals.
  - Done when: primary goal status (>75% val accuracy) is explicitly stated.

- [ ] 4.3 Add limitations and future work notes aligned with `masterplan.md`.
  - Scope: short, evidence-based discussion (no over-claiming).
  - Done when: final section is complete and consistent with observed results.

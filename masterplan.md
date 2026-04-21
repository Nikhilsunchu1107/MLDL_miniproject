# masterplan.md
# Ocular Gender Classification via Retinal Fundus Imaging
### Deep Learning Pipeline — Miniproject Blueprint

---

## 1. App Overview & Objectives

Build a binary gender classification model (Male/Female) from retinal fundus photographs using transfer learning. The model will learn subtle sub-visual anatomical differences in the retina — vascular patterns, optic disc size, macular structure — that correlate with biological sex.

**Primary Goal:** Train a model that achieves >75% validation accuracy on the available dataset.
**Secondary Goal:** Produce a fully documented Jupyter notebook pipeline with evaluation metrics and a complete project report.

---

## 2. Target Dataset

**Primary Dataset:** `datasets/Dataset` — 700 labeled fundus images
- Male: 364 images
- Female: 336 images
- Resolution: 3900x3072 (high quality)
- Labels: `Demographics of the participants.xlsx`
- Quality metadata: `Ground Truth.xlsx`, `Individual Quality Assessment.xlsx`
- ⚠️ Known issue: 200 AMD/DR images are TIFF encoded with .jpg extension — must be handled in loader

**Preprocessing Validation Dataset:** `datasets/retinal-disease-detection-002` — 2,254 unlabelled images
- Used exclusively to stress-test the preprocessing pipeline before touching labeled training data
- Validates: TIFF detection edge cases, CLAHE behavior on pathological retinas, circular mask robustness across resolutions, resize quality
- NOT used for training or evaluation — labels are irrelevant here
- Chosen over `fundus-image-registration` which has 53% duplicate rate (144/270) and is a registration dataset with intentionally paired images — not representative of real clinical diversity

**What to ignore entirely:**
- retinal-colorized-oct-images-003 (no labels, wrong modality — OCT not fundus)
- fundus-image-registration (53% duplicates, registration pairs — not useful even for preprocessing validation)
- retina-blood-vessel, retinal-vessel-segmentation (segmentation mask assets, not raw fundus images)
- UK_Biobank_Dataset (only 40 images, 28 near-black — too small and too noisy)
- papila-retinal-fundus-images (unresolved 0/1 gender codes — risky to assume mapping)

---

## 3. Core Features of the Pipeline

1. **Data Loading & Label Parsing** — Read Demographics Excel, map image IDs to Male/Female
2. **TIFF-in-JPG Detection** — Use python-magic or PIL to detect actual encoding
3. **Image Preprocessing** — Resize, normalize, apply circular mask
4. **Data Augmentation** — Horizontal flip, rotation, brightness/contrast jitter
5. **Train/Val/Test Split** — 70/15/15 stratified by gender
6. **Transfer Learning Model** — EfficientNet-B0 pretrained on ImageNet, fine-tuned
7. **Training Loop** — Adam optimizer, binary cross-entropy, early stopping
8. **Evaluation** — Accuracy, ROC-AUC, Confusion Matrix, Precision, Recall, F1
9. **Visualization** — Training curves, Grad-CAM saliency maps, sample predictions

---

## 4. High-Level Technical Stack

| Component | Choice | Why |
|---|---|---|
| Language | Python 3.10+ | Standard for ML |
| Deep Learning | PyTorch + torchvision | Flexible, great pretrained models |
| Data Handling | Pandas, OpenPyXL | For Excel label parsing |
| Image Processing | Pillow, OpenCV | Preprocessing + TIFF detection |
| Visualization | Matplotlib, Seaborn | Curves, confusion matrix |
| Explainability | pytorch-grad-cam | Grad-CAM saliency maps |
| Environment | Jupyter Notebook | Required deliverable |
| Hardware | RTX 4050 (local) | Sufficient for EfficientNet-B0 on 700 images |

---

## 5. Model Architecture

**Backbone:** EfficientNet-B0 (pretrained on ImageNet)

**Why EfficientNet-B0:**
- Lightweight enough for RTX 4050 with high-res inputs
- Strong transfer learning performance on medical images
- Balances accuracy vs. compute perfectly for 700 image dataset
- Research paper confirms CNN-based approaches work well for 2D fundus

**Architecture Flow:**
```
Input Image (224x224x3)
       ↓
EfficientNet-B0 Feature Extractor (frozen initially, then unfrozen)
       ↓
Global Average Pooling
       ↓
Dropout (0.3)
       ↓
Dense Layer (256 units, ReLU)
       ↓
Dropout (0.2)
       ↓
Output Neuron (1 unit, Sigmoid)
       ↓
Binary Prediction (Male/Female)
```

**Training Strategy (Two-Phase):**
- Phase 1: Freeze backbone, train only the head (5 epochs) — fast convergence
- Phase 2: Unfreeze top layers, fine-tune end-to-end (15 epochs, lower LR) — refinement

---

## 6. Preprocessing Pipeline

1. Read image → detect actual encoding (handle TIFF-in-JPG)
2. Apply circular retinal mask (remove black background)
3. CLAHE (Contrast Limited Adaptive Histogram Equalization) on green channel
4. Resize to 224x224 using bilinear interpolation
5. Normalize to ImageNet mean/std ([0.485, 0.456, 0.406] / [0.229, 0.224, 0.225])

**Augmentation (training only):**
- Random horizontal flip (p=0.5)
- Random rotation (±15°)
- Random brightness/contrast jitter (±0.2)
- Random crop + resize

---

## 7. Conceptual Data Model

```
Demographics.xlsx
    └── image_id (string) → gender (0/1 → Male/Female)
    
Dataset/
    ├── image_001.jpg  ← actual JPEG
    ├── image_002.jpg  ← actual TIFF (must decode with PIL)
    └── ...

Ground Truth.xlsx
    └── image_id → quality_score (use to filter low quality)
```

**Recommended:** Filter out images with poor quality scores before training.

---

## 8. Evaluation Plan

| Metric | Target | Tool |
|---|---|---|
| Validation Accuracy | >75% | sklearn |
| ROC-AUC | >0.78 | sklearn |
| F1-Score | >0.75 | sklearn |
| Confusion Matrix | Visualized | seaborn heatmap |
| Grad-CAM | Qualitative | pytorch-grad-cam |
| Training Curves | Loss + Accuracy | matplotlib |

**Note on expectations:** With 700 images and transfer learning, 75-85% accuracy is realistic. The research paper achieved 87-99% AUC on much larger datasets (thousands of images). Don't over-claim results in the report.

---

## 9. Development Phases — ONE DAY PLAN

### Phase 0: Preprocessing Validation (30-45 mins) ← NEW
- [ ] Run preprocessing pipeline on 50-100 images from `retinal-disease-detection-002`
- [ ] Visually inspect CLAHE output, mask application, resize quality
- [ ] Confirm TIFF detection handles varied resolutions (1440x960 to 2000x1333)
- [ ] Fix any issues before touching the labeled training data
- [ ] Only proceed to Phase 1 once preprocessing looks clean

### Phase 1: Data Pipeline (2-3 hours)
- [ ] Parse Demographics Excel, build image_id → label map
- [ ] Write dataset class with TIFF detection
- [ ] Verify labels visually (spot check 10-20 images)
- [ ] Apply quality filter using Ground Truth.xlsx
- [ ] Set up train/val/test splits

### Phase 2: Model + Training (2-3 hours)
- [ ] Build EfficientNet-B0 model with custom head
- [ ] Implement two-phase training strategy
- [ ] Add early stopping + model checkpointing
- [ ] Train on RTX 4050 locally

### Phase 3: Evaluation (1-2 hours)
- [ ] Generate all metrics (accuracy, AUC, F1, confusion matrix)
- [ ] Plot training curves
- [ ] Generate Grad-CAM visualizations on test samples

### Phase 4: Report (2-3 hours)
- [ ] Fill report sections using masterplan content
- [ ] Add result screenshots and metric tables
- [ ] Clean up Jupyter notebook with markdown cells

---

## 10. Potential Challenges & Solutions

| Challenge | Solution |
|---|---|
| TIFF files with .jpg extension | Use PIL to attempt open; fallback to TIFF loader |
| Small dataset (700 images) | Transfer learning + aggressive augmentation |
| Class imbalance (364M vs 336F) | Weighted sampler or class weights in loss |
| Disease confounding (DR/AMD patients) | Note as limitation; ideally filter healthy-only subset |
| Overfitting | Dropout, early stopping, data augmentation |
| High resolution images (3900x3072) | Resize to 224x224 during loading |

---

## 11. Future Scope (for report section)

- **Self-Supervised Pretraining (SimCLR/MoCo):** Pretrain EfficientNet-B0 backbone on ALL available unlabelled fundus images (~3,000+ from retinal-disease-detection-002 and others) before fine-tuning on labeled data. This inserts cleanly as a Phase 0 before the current training pipeline — backbone learns retina-specific features without needing gender labels, then labeled fine-tuning benefits from a stronger starting point. Estimated improvement: 3-7% accuracy gain. Requires ~3-4 additional hours of compute on RTX 4050.
- Scale to larger labeled datasets (UK Biobank full release)
- Add interpretability with BagNet architecture
- Multi-task learning: predict gender + disease simultaneously
- Fairness-aware training with FaMI framework
- PAPILA dataset integration once gender codes are resolved
- Clinical deployment via DICOM/FHIR integration

---

## 12. Key References (for report)

1. Poplin et al. (2018) — Predicting cardiovascular risk factors from retinal fundus
2. BagNet gender classification paper (ResearchGate, 2021)
3. Gender prediction across multiethnic populations — NIH PMC8408758
4. EfficientNet: Rethinking Model Scaling for CNNs (Tan & Le, 2019)
5. PAPILA Dataset paper
6. Your attached research document: "Comprehensive Design of a Deep Learning Pipeline for Gender Classification via Retinal and Ocular Imaging"

---

*Generated as part of miniproject planning session. One-day execution plan. Focus on a clean, working pipeline over complexity.*

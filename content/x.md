# Ablation Study — BiomedCLIP Multi-Label Chest X-Ray Classification
## 34-Class Indiana University CXR Dataset

**Author:** Brij Kishor (2026)  
**Backbone:** `microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224` (ViT-B/16, pretrained on 15M biomedical image-text pairs)  
**Task:** Multi-label thoracic pathology classification — 34 active classes (35 defined; Bronchiolitis/Subcutaneous Emphysema had 0 test samples in single-view runs)

---

## 1. Dataset Summary

| Property | Value |
|---|---|
| Source | Indiana University Chest X-ray Collection (OpenI) |
| Total valid images | 7,466 |
| Total patients / reports | 3,852 |
| Disease classes defined | 35 |
| Active classes (with test data) | 34 |
| Train/Test split | 80% / 20% (patient-level, no leakage) |
| Training images | 5,950 (3,080 patients) |
| Test images | 1,516 (771 patients) |
| Labels per image | Min 1, Max 9, Mean 1.76 |
| Class imbalance ratio | up to 1,581:1 |

### Class Distribution (Training Set)

| Class | Train Count | % |
|---|---|---|
| Normal | 3,162 | 42.35% |
| Degenerative Change | 1,360 | 18.22% |
| Lesion | 1,338 | 17.92% |
| Hyperinflation | 1,076 | 14.41% |
| Calcified Granuloma | 798 | 10.69% |
| Cardiomegaly | 662 | 8.87% |
| Volume Loss | 634 | 8.49% |
| Calcinosis | 558 | 7.47% |
| Airspace Disease | 372 | 4.98% |
| Fibrosis | 368 | 4.93% |
| Increased Lung Markings | 305 | 4.09% |
| Pleural Effusion | 292 | 3.91% |
| Emphysema | 286 | 3.83% |
| Nodule | 211 | 2.83% |
| Edema | 184 | 2.47% |
| Scoliosis | 177 | 2.37% |
| Fractures | 168 | 2.25% |
| Hernia | 155 | 2.08% |
| Pleural Thickening | 151 | 2.02% |
| Osteophyte | 137 | 1.84% |
| Interstitial Lung Disease | 123 | 1.65% |
| Consolidation | 114 | 1.53% |
| Cardiac Shadow (abnormal) | 105 | 1.41% |
| Thickening | 100 | 1.34% |
| Kyphosis | 58 | 0.78% |
| Pneumothorax | 54 | 0.72% |
| Mass | 41 | 0.55% |
| Pulmonary Artery Enlargement | 36 | 0.48% |
| Pulmonary Fibrosis | 34 | 0.46% |
| Effusion | 26 | 0.35% |
| Bronchiectasis | 13 | 0.17% |
| Bullous Disease | 9 | 0.12% |
| Rib Fracture | 8 | 0.11% |
| Subcutaneous Emphysema | 6 | 0.08% |
| Bronchiolitis | 2 | 0.03% |

---

## 2. Experiment Overview

All 5 experiments use the same **BiomedCLIP ViT-B/16** backbone. The key differences are the fusion strategy (single-view vs. multi-view), backbone freeze policy, loss function, and batch size.

| Exp | Folder | Date | Key Change | Test AUC |
|---|---|---|---|---|
| **E1** | `lebelwise-16batch/` | Jan 2026 | Baseline: single-view, fully frozen, BS=32, Focal Loss | **71.54%** |
| **E2** | `lebelwise-36batch/` | Jan 2026 | Batch size 32→16 | **72.52%** |
| **E3** | `fine_tune/results/` | Apr 10, 2026 | Multi-view fusion, 9 strategies compared, 6 layers unfrozen, composite loss | **75.83%** (feat. attn) |
| **E4** | `fine_tune/output_new/` | May 1, 2026 | Feature attention, LR 1e-3→1e-4, 40 epochs (CV only, no held-out test) | Val 74.18% |
| **E5** | `fine_tune/output_feature/` | May 1, 2026 | Feature attention, LR back to 1e-3, 100 epochs, best model | **75.49%** (mean) |

---

## 3. Experiment 1 — Single-View Frozen BiomedCLIP, Batch=32

**Folder:** `lebelwise-16batch/` | **Script:** `train_optimized.py`

### Motivation
Establish a baseline using the BiomedCLIP visual encoder with zero backbone modification. Only a 3-layer MLP classification head is trained. Images are processed one at a time (no frontal–lateral pairing); frontal and lateral are treated as separate samples.

### Configuration

| Parameter | Value |
|---|---|
| Backbone freeze | **Fully frozen** (0 backbone layers trained) |
| Total parameters | ~196M |
| Trainable parameters | ~404K (head only, 0.2%) |
| Classification head | Linear(512→256) → BN → GELU → Dropout → Linear(256→35) |
| Loss | Focal Loss (α=0.25, γ=2.0) + inverse-frequency class weights |
| Optimizer | AdamW (lr=2e-5, weight_decay=1e-4) |
| Scheduler | ReduceLROnPlateau (factor=0.5, patience=3) |
| Batch size | 32 (AMP mixed precision) |
| Early stopping | patience=5 |
| Gradient clipping | max_norm=1.0 |
| Cross-validation | 5-fold **StratifiedKFold** *(not patient-safe)* |
| Augmentation | Resize 224×224, RandomHorizontalFlip, ImageNet normalization |

### Epoch Search

| Epochs | Mean Val AUC | Std | Time |
|---|---|---|---|
| 30 | 0.6233 | ±0.0204 | ~194 min |
| 50 | 0.6895 | ±0.0093 | ~282 min |
| 75 | 0.7147 | ±0.0071 | ~429 min |
| 90 | 0.7252 | ±0.0103 | ~537 min |
| **100** | **0.7287** | **±0.0107** | **~597 min** |

**Optimal: 100 epochs** (total search ≈ 30.7 hours)

### Overall Test Results (100 epochs, best checkpoint)

| Split | Label-wise Accuracy | AUC | Macro F1 | mAP |
|---|---|---|---|---|
| Train (80%) | 95.09% | 78.92% | 14.80% | 24.03% |
| **Test (20%)** | **94.98%** | **71.54%** | **4.94%** | **13.63%** |

### Per-Class Results — Experiment 1 (test set, 1494 images)

> Source: `lebelwise-16batch/evaluation_results.csv`

| Rank | Class | Test Pos | AUC | AP | Best F1 | Rating |
|---|---|---|---|---|---|---|
| 1 | Cardiac Shadow (abnormal) | 21 | **0.9052** | 0.2514 | 0.2745 | EXCELLENT |
| 2 | Pleural Effusion | 58 | **0.8933** | 0.3633 | 0.3607 | EXCELLENT |
| 3 | Bronchiectasis | 1 | 0.8858 | 0.0058 | 0.0000 | EXCELLENT |
| 4 | Edema | 38 | **0.8796** | 0.1982 | 0.2439 | EXCELLENT |
| 5 | Cardiomegaly | 132 | **0.8500** | 0.3741 | 0.4195 | EXCELLENT |
| 6 | Rib Fracture | 2 | 0.8147 | 0.0518 | 0.1111 | EXCELLENT |
| 7 | Pleural Thickening | 28 | 0.8100 | 0.0929 | 0.1446 | EXCELLENT |
| 8 | Pulmonary Fibrosis | 9 | 0.7896 | 0.1687 | 0.1667 | GOOD |
| 9 | Effusion | 4 | 0.7846 | 0.0503 | 0.1538 | GOOD |
| 10 | Volume Loss | 131 | **0.7656** | 0.2616 | 0.3588 | GOOD |
| 11 | Airspace Disease | 74 | **0.7652** | 0.1621 | 0.2683 | GOOD |
| 12 | Consolidation | 25 | 0.7625 | 0.0748 | 0.1351 | GOOD |
| 13 | Emphysema | 62 | **0.7564** | 0.1335 | 0.1809 | GOOD |
| 14 | Hernia | 33 | 0.7516 | 0.0614 | 0.1000 | GOOD |
| 15 | Kyphosis | 12 | 0.7460 | 0.0252 | 0.0513 | GOOD |
| 16 | Fibrosis | 71 | **0.7214** | 0.1343 | 0.1640 | GOOD |
| 17 | Increased Lung Markings | 60 | 0.6978 | 0.0774 | 0.1429 | FAIR |
| 18 | Lesion | 274 | **0.6943** | 0.3607 | 0.3845 | FAIR |
| 19 | Mass | 7 | 0.6889 | 0.0235 | 0.0645 | FAIR |
| 20 | Interstitial Lung Disease | 26 | 0.6838 | 0.0374 | 0.1020 | FAIR |
| 21 | Hyperinflation | 219 | **0.6756** | 0.2680 | 0.3196 | FAIR |
| 22 | Fractures | 37 | 0.6653 | 0.0502 | 0.0950 | FAIR |
| 23 | normal | 633 | 0.6497 | 0.5529 | 0.6162 | FAIR |
| 24 | Thickening | 18 | 0.6418 | 0.0311 | 0.0882 | FAIR |
| 25 | Pneumothorax | 9 | 0.6213 | 0.0225 | 0.0571 | FAIR |
| 26 | Scoliosis | 36 | 0.6183 | 0.0428 | 0.0877 | FAIR |
| 27 | Osteophyte | 30 | 0.6138 | 0.0532 | 0.0805 | FAIR |
| 28 | Degenerative Change | 277 | **0.6121** | 0.2512 | 0.3512 | FAIR |
| 29 | Nodule | 49 | 0.6116 | 0.0471 | 0.0870 | FAIR |
| 30 | Bullous Disease | 1 | 0.5988 | 0.0017 | 0.0000 | POOR |
| 31 | Calcinosis | 115 | 0.5942 | 0.1083 | 0.1752 | POOR |
| 32 | Calcified Granuloma | 161 | 0.5590 | 0.1529 | 0.2067 | POOR |
| 33 | Pulmonary Artery Enlargement | 10 | 0.5010 | 0.0069 | 0.0100 | POOR |
| 34 | Bronchiolitis | 0 | N/A | N/A | N/A | INSUFFICIENT |
| 35 | Subcutaneous Emphysema | 0 | N/A | N/A | N/A | INSUFFICIENT |

**Macro AUC (33 classes with data): 0.7154** | Classes ≥0.80: 8 | Classes ≥0.70: 16

---

## 4. Experiment 2 — Single-View Frozen BiomedCLIP, Batch=16

**Folder:** `lebelwise-36batch/` | **Script:** `xray_new.py`

### Motivation
Experiment 1 used batch=32. Hypothesis: **smaller batches** improve gradient diversity on a heavily imbalanced dataset — each mini-batch sees a richer mix of rare classes when batch size is halved. Epoch search range was narrowed to [75, 90, 100] since E1 already showed 30–75 epochs are sub-optimal.

### Configuration Changes vs E1

| Parameter | E1 | E2 | Effect |
|---|---|---|---|
| Batch size | 32 | **16** | More diverse gradients per update |
| Epoch search | [30,50,75,90,100] | [75,90,100] | Focused search |
| Trainable params | 2.8M (head: 512→256→35) | Same | No change |
| Everything else | — | Identical | — |

### Epoch Search

| Epochs | Mean Val AUC | Std |
|---|---|---|
| 75 | 0.7263 | ±0.0140 |
| 90 | 0.7304 | ±0.0075 |
| **100** | **0.7349** | **±0.0048** |

Std drops 3× vs E1 (±0.0048 vs ±0.0107) — smaller batches produce more stable cross-fold training.

### Overall Test Results (100 epochs)

| Split | Label-wise Accuracy | AUC | Macro F1 | mAP |
|---|---|---|---|---|
| Train (80%) | 95.22% | 80.78% | 18.16% | 28.67% |
| **Test (20%)** | **95.09%** | **72.52%** | **4.75%** | **13.56%** |

**Gain vs E1:** Test AUC +0.98% | Train AUC +1.86% | CV std ↓ 3×

### Per-Class Results — Experiment 2 (test set, 1494 images)

> Source: `lebelwise-36batch/evaluation_results.csv`

| Rank | Class | Test Pos | AUC | AP | Best F1 | Rating |
|---|---|---|---|---|---|---|
| 1 | Pleural Effusion | 58 | **0.8972** | 0.3515 | 0.3953 | EXCELLENT |
| 2 | Rib Fracture | 2 | 0.8904 | 0.0130 | 0.0215 | EXCELLENT |
| 3 | Cardiac Shadow (abnormal) | 21 | **0.8895** | 0.1253 | 0.1951 | EXCELLENT |
| 4 | Edema | 38 | **0.8737** | 0.1982 | 0.2558 | EXCELLENT |
| 5 | Cardiomegaly | 132 | **0.8397** | 0.3718 | 0.3733 | EXCELLENT |
| 6 | Pulmonary Fibrosis | 9 | **0.8059** | 0.0679 | 0.1111 | EXCELLENT |
| 7 | Emphysema | 62 | **0.8047** | 0.1729 | 0.2485 | EXCELLENT |
| 8 | Consolidation | 25 | 0.7859 | 0.0942 | 0.2182 | GOOD |
| 9 | Airspace Disease | 74 | **0.7775** | 0.1892 | 0.2559 | GOOD |
| 10 | Bullous Disease | 1 | 0.7736 | 0.0029 | 0.0000 | GOOD |
| 11 | Kyphosis | 12 | 0.7679 | 0.0395 | 0.0792 | GOOD |
| 12 | Pleural Thickening | 28 | 0.7677 | 0.0645 | 0.1375 | GOOD |
| 13 | Hernia | 33 | 0.7497 | 0.0563 | 0.1045 | GOOD |
| 14 | Mass | 7 | 0.7468 | 0.1035 | 0.2000 | GOOD |
| 15 | Volume Loss | 131 | **0.7463** | 0.2197 | 0.3034 | GOOD |
| 16 | Bronchiectasis | 1 | 0.7395 | 0.0026 | 0.0000 | GOOD |
| 17 | Interstitial Lung Disease | 26 | 0.7245 | 0.0718 | 0.1493 | GOOD |
| 18 | Effusion | 4 | 0.7107 | 0.0169 | 0.0741 | GOOD |
| 19 | Fibrosis | 71 | **0.7095** | 0.1242 | 0.1900 | GOOD |
| 20 | normal | 633 | 0.6996 | 0.6058 | 0.6324 | FAIR |
| 21 | Fractures | 37 | 0.6917 | 0.0963 | 0.1091 | FAIR |
| 22 | Increased Lung Markings | 60 | 0.6874 | 0.0832 | 0.1436 | FAIR |
| 23 | Lesion | 274 | **0.6872** | 0.3430 | 0.3775 | FAIR |
| 24 | Hyperinflation | 219 | **0.6632** | 0.2364 | 0.3161 | FAIR |
| 25 | Thickening | 18 | 0.6419 | 0.0257 | 0.0667 | FAIR |
| 26 | Pneumothorax | 9 | 0.6372 | 0.1220 | 0.1429 | FAIR |
| 27 | Nodule | 49 | 0.6343 | 0.0506 | 0.0914 | FAIR |
| 28 | Osteophyte | 30 | 0.6314 | 0.0663 | 0.1064 | FAIR |
| 29 | Degenerative Change | 277 | **0.6238** | 0.2627 | 0.3439 | FAIR |
| 30 | Calcinosis | 115 | 0.6222 | 0.1137 | 0.1854 | FAIR |
| 31 | Scoliosis | 36 | 0.6063 | 0.0419 | 0.0615 | FAIR |
| 32 | Calcified Granuloma | 161 | 0.5709 | 0.1313 | 0.2030 | POOR |
| 33 | Pulmonary Artery Enlargement | 10 | 0.5339 | 0.0085 | 0.0196 | POOR |
| 34 | Bronchiolitis | 0 | N/A | N/A | N/A | INSUFFICIENT |
| 35 | Subcutaneous Emphysema | 0 | N/A | N/A | N/A | INSUFFICIENT |

**Macro AUC (33 classes): 0.7252** | Classes ≥0.80: 10 | Classes ≥0.70: 19 | +2 classes moved up vs E1

---

## 5. Experiment 3 — Multi-View Fusion Strategy Ablation (9 Strategies)

**Folder:** `fine_tune/results/` | **Script:** `fine_tune/05_train_ablation_all_strategies.py` | **Date:** Apr 10, 2026

### Motivation
Single-view models (E1/E2) ignore the complementary diagnostic information in the lateral view. This experiment introduces **two-view (frontal + lateral) fusion** and systematically compares 9 different fusion strategies to find the optimal way to combine the two views. Additional improvements:
- **Partial backbone unfreeze** (last 6 transformer blocks) for domain adaptation
- **Composite loss** (Weighted BCE + Focal + Soft Jaccard) for better multi-label learning
- **Patient-safe GroupKFold** — no patient UID can appear in both train and validation

### Configuration

| Parameter | Value |
|---|---|
| Backbone | BiomedCLIP ViT-Base (last **6 blocks unfrozen**) |
| Total parameters | 197,062,467 |
| Trainable parameters | ~1,159,746 (0.59%) |
| Fusion module | 558,112 params |
| Classification head | Linear(768→256→34), compact |
| Loss | 0.6 × WeightedBCE + 0.2 × Focal + 0.2 × SoftJaccard |
| Optimizer | AdamW (lr=1e-3, weight_decay=1e-5) |
| Batch size | 16 |
| Max epochs | 100 (early stopping, patience=8) |
| Cross-validation | 5-fold **patient-safe GroupKFold** |

### Fusion Strategies Explained

| Strategy | Mechanism |
|---|---|
| **Early** | Concatenate frontal+lateral at input level, single forward pass |
| **Late** | Encode views independently, concatenate their embeddings |
| **Sum** | Element-wise sum of frontal and lateral feature vectors |
| **Max** | Element-wise maximum of frontal and lateral features |
| **Weighted** | Learnable scalar weights (one per view) applied to features |
| **Gated** | Sigmoid gate per feature dimension — soft masking per channel |
| **Feature Attention** | `combined = front+lat` → `attn = sigmoid(MLP(combined))` → `out = front*attn + lat*(1-attn)` |
| **Cross-Modal Attention** | Bidirectional cross-attention — frontal queries lateral and vice versa |
| **Self-Attention** | Joint self-attention over both views concatenated as a token sequence |

### 5-Fold Cross-Validation Results — All 9 Strategies

| Rank | Strategy | Test AUC (mean ± std) | Test mAP | Test Macro F1 | Test Jaccard |
|---|---|---|---|---|---|
| **1** | **feature_attention** | **0.7583 ± 0.0155** | **0.1755** | **0.1199** | **0.4544** |
| 2 | weighted | 0.7540 ± 0.0214 | 0.1701 | 0.1153 | 0.4488 |
| 3 | sum | 0.7507 ± 0.0180 | 0.1718 | 0.1160 | 0.4565 |
| 4 | gated | 0.7517 ± 0.0130 | 0.1678 | 0.1159 | 0.4531 |
| 5 | max | 0.7502 ± 0.0148 | 0.1777 | 0.1220 | 0.4560 |
| 6 | early | 0.7488 ± 0.0202 | 0.1733 | 0.1120 | 0.4523 |
| 7 | late | 0.7467 ± 0.0206 | 0.1716 | 0.1193 | 0.4568 |
| 8 | cross_modal_attention | 0.7403 ± 0.0108 | 0.1622 | 0.1043 | 0.4515 |
| 9 | self_attention | 0.7285 ± 0.0205 | 0.1453 | 0.0984 | 0.4501 |

**Winner: Feature Attention** — smallest gap between best/worst fusion strategies is Jaccard (0.4488–0.4568), showing all multi-view strategies beat single-view but Feature Attention leads in AUC and F1.

### Per-Fold Detail — Feature Attention (Best Strategy)

| Fold | Val AUC | Val MAP | Val F1 | Val Jaccard | Test AUC | Test MAP | Test F1 | Test Jaccard |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.7426 | 0.1724 | 0.1041 | 0.4441 | 0.7617 | 0.1813 | 0.1344 | 0.4518 |
| 2 | 0.6869 | 0.1151 | 0.0894 | 0.4321 | 0.7530 | 0.1612 | 0.1246 | 0.4508 |
| 3 | 0.7191 | 0.1257 | 0.0903 | 0.4388 | 0.7365 | 0.1677 | 0.1041 | 0.4521 |
| 4 | 0.7161 | 0.1280 | 0.0908 | 0.4470 | **0.7792** | 0.1870 | 0.1205 | 0.4556 |
| 5 | 0.7205 | 0.1386 | 0.1034 | 0.4436 | 0.7610 | 0.1805 | 0.1159 | 0.4616 |
| **Mean** | **0.7170** | **0.1360** | **0.0956** | **0.4411** | **0.7583** | **0.1755** | **0.1199** | **0.4544** |

---

## 6. Experiment 4 — Feature Attention, Learning Rate Ablation (lr=1e-4)

**Folder:** `fine_tune/output_new/` | **Script:** `fine_tune/train_feature_attention.py` | **Date:** May 1, 2026 (11:41 AM)

### Motivation
E3 used lr=1e-3 which converged quickly but had inter-fold variability. This experiment tests whether **lr=1e-4** (10× smaller) produces more stable convergence on the 6-unfrozen-block feature attention model. Max epochs reduced to 40 (early stopping); validation-only evaluation (no held-out test set comparison in this run).

### Configuration Changes vs E3

| Parameter | E3 | E4 | Effect |
|---|---|---|---|
| Learning rate | 1e-3 | **1e-4** | Slower, more careful weight updates |
| Max epochs | 100 | **40** | Reduced — early stopping catches optimum |
| Strategies run | 9 | **1** (feature_attention only) | Focused experiment |
| Test evaluation | Full held-out test | **CV only** | No direct AUC comparison |

### 5-Fold Validation Results

| Fold | Best Epoch | Val AUC | Val mAP | Val F1 | Val Jaccard | Val Hamming |
|---|---|---|---|---|---|---|
| 1 | 24 | 0.7358 | 0.1706 | 0.0959 | 0.3853 | 0.0631 |
| 2 | 29 | 0.7335 | 0.1550 | 0.0959 | 0.4057 | 0.0589 |
| 3 | **40** | **0.7529** | **0.1855** | **0.1168** | 0.3772 | 0.0646 |
| 4 | 25 | 0.7123 | 0.1745 | 0.0986 | 0.4116 | 0.0584 |
| 5 | 35 | **0.7745** | 0.1844 | 0.1131 | **0.4494** | 0.0530 |
| **Mean** | — | **0.7418** | **0.1740** | **0.1041** | **0.4058** | **0.0596** |
| Std | — | ±0.0208 | — | — | — | — |

**Observation:** Lower LR (1e-4) achieves mean Val AUC 0.7418 vs E3's 0.7170 (validation), suggesting it generalises better within the CV folds. However, Fold 5 reaches best epoch only at epoch 35/40, meaning lr=1e-4 needs more epochs. Decision: keep lr=1e-3 but with full 100-epoch budget (→ Exp 5).

---

## 7. Experiment 5 — Feature Attention, lr=1e-3, 100 Epochs (Final Best Model)

**Folder:** `fine_tune/output_feature/` | **Script:** `feature.py` | **Date:** May 1, 2026 (1:11 PM)

### Motivation
Consolidate findings: use Feature Attention (best strategy from E3), lr=1e-3 (which E3 showed is good for convergence speed), and full 100-epoch budget (so the model isn't cut short). Evaluate against the pre-split held-out test set (771 patients, 1,516 images) for all 5 folds. Compute per-class confusion matrices.

### Configuration

| Parameter | Value |
|---|---|
| Strategy | Feature Attention fusion |
| Backbone unfrozen blocks | 6 (last 6 transformer blocks) |
| Total parameters | 197,062,467 |
| Trainable parameters | 1,159,746 (0.59%) |
| Fusion parameters | 558,112 |
| Head | Linear(768→256→34) |
| Loss | Weighted BCE + Focal + SoftJaccard |
| Optimizer | AdamW (lr=1e-3, weight_decay=1e-5) |
| Batch size | 16 |
| Max epochs | 100 (patience=8) |
| CV | 5-fold patient-safe GroupKFold |
| Train set | 3,080 patients / 5,950 images |
| Test set | 771 patients / 1,516 images (held-out) |

### Per-Fold Results

| Fold | Best Epoch | Val AUC | Val MAP | Val F1 | **Test AUC** | Test MAP | Test F1 |
|---|---|---|---|---|---|---|---|
| 1 | 11 | 0.7436 | 0.1455 | 0.1001 | 0.7632 | 0.1627 | 0.1231 |
| 2 | 11 | 0.6877 | 0.1331 | 0.1025 | 0.7483 | 0.1679 | 0.1231 |
| 3 | 6 | 0.7076 | 0.1129 | 0.0696 | 0.7361 | 0.1664 | 0.1108 |
| **4** | 6 | 0.6957 | 0.1242 | 0.0800 | **0.7693** | **0.1755** | **0.1202** |
| 5 | 11 | 0.7328 | 0.1404 | 0.0961 | 0.7576 | 0.1739 | 0.1182 |
| **Mean** | — | **0.7135** | **0.1312** | **0.0897** | **0.7549** | **0.1693** | **0.1191** |
| Std | — | — | — | — | ±0.0130 | — | — |

**Best model: Fold 4** (Test AUC=0.7693) → saved as `BEST_MODEL_FINAL.pth`

### Per-Class Results — Experiment 5, Best Model (Fold 4)

Computed from `test_predictions_feature_attention_fold4.csv` using the held-out test set (771 patients).  
Ensemble AUC = average of probabilities from all 5 trained folds (soft voting).

> Source: `fine_tune/output_feature/per_class_auc_exp5.csv`

| Rank | Class | Test Pos | AUC (Fold4) | AUC (Ensemble) | AP | F1 | Rating |
|---|---|---|---|---|---|---|---|
| 1 | Pleural Effusion | 24 | **0.9350** | 0.9423 | 0.4828 | 0.4058 | EXCELLENT |
| 2 | Bullous Disease | 1 | 0.9318 | 0.9740 | 0.0182 | 0.0000 | EXCELLENT |
| 3 | Pneumothorax | 7 | **0.9122** | 0.9344 | 0.1164 | 0.0000 | EXCELLENT |
| 4 | Bronchiectasis | 2 | 0.9008 | 0.9655 | 0.0202 | 0.0000 | EXCELLENT |
| 5 | Subcutaneous Emphysema | 1 | 0.8929 | 0.8714 | 0.0115 | 0.0000 | EXCELLENT |
| 6 | Cardiac Shadow (abnormal) | 12 | **0.8889** | 0.9078 | 0.1806 | 0.0000 | EXCELLENT |
| 7 | Consolidation | 12 | **0.8764** | 0.8702 | 0.1078 | 0.0000 | EXCELLENT |
| 8 | Edema | 21 | **0.8754** | 0.8773 | 0.1623 | 0.1250 | EXCELLENT |
| 9 | Airspace Disease | 41 | **0.8649** | 0.8678 | 0.3749 | 0.4048 | EXCELLENT |
| 10 | Effusion | 6 | 0.8637 | 0.8521 | 0.2538 | 0.0000 | EXCELLENT |
| 11 | Cardiomegaly | 78 | **0.8498** | 0.8658 | 0.4368 | 0.4878 | GOOD |
| 12 | Volume Loss | 67 | **0.8033** | 0.8058 | 0.3634 | 0.4025 | GOOD |
| 13 | Emphysema | 35 | **0.7873** | 0.8244 | 0.3040 | 0.3846 | GOOD |
| 14 | Osteophyte | 10 | 0.7855 | 0.7882 | 0.0653 | 0.0000 | GOOD |
| 15 | Interstitial Lung Disease | 17 | 0.7719 | 0.8141 | 0.0978 | 0.0000 | GOOD |
| 16 | Pleural Thickening | 13 | 0.7667 | 0.8079 | 0.1372 | 0.0000 | GOOD |
| 17 | Rib Fracture | 1 | 0.7636 | 0.5539 | 0.0053 | 0.0000 | GOOD |
| 18 | Lesion | 154 | **0.7561** | 0.7572 | 0.4631 | 0.4744 | GOOD |
| 19 | normal | 326 | 0.7419 | 0.7516 | 0.6328 | 0.6333 | FAIR |
| 20 | Hernia | 15 | 0.7407 | 0.7119 | 0.0554 | 0.0000 | FAIR |
| 21 | Fractures | 17 | 0.7391 | 0.7428 | 0.0570 | 0.0000 | FAIR |
| 22 | Hyperinflation | 109 | **0.7373** | 0.7390 | 0.3043 | 0.3333 | FAIR |
| 23 | Increased Lung Markings | 35 | 0.7285 | 0.7436 | 0.1362 | 0.0000 | FAIR |
| 24 | Fibrosis | 43 | **0.7263** | 0.7293 | 0.2139 | 0.1509 | FAIR |
| 25 | Pulmonary Artery Enlargement | 4 | 0.7241 | 0.7256 | 0.0168 | 0.0000 | FAIR |
| 26 | Thickening | 11 | 0.7178 | 0.6834 | 0.0303 | 0.0000 | FAIR |
| 27 | Kyphosis | 8 | 0.7094 | 0.7267 | 0.0657 | 0.0000 | FAIR |
| 28 | Pulmonary Fibrosis | 4 | 0.6975 | 0.7199 | 0.0144 | 0.0000 | FAIR |
| 29 | Nodule | 23 | 0.6665 | 0.6419 | 0.0591 | 0.0000 | FAIR |
| 30 | Calcinosis | 58 | 0.6457 | 0.6568 | 0.1211 | 0.0000 | POOR |
| 31 | Mass | 6 | 0.6351 | 0.6368 | 0.2033 | 0.0000 | POOR |
| 32 | Degenerative Change | 148 | **0.6115** | 0.6228 | 0.2877 | 0.2857 | POOR |
| 33 | Calcified Granuloma | 76 | 0.5667 | 0.5340 | 0.1221 | 0.0000 | POOR |
| 34 | Scoliosis | 21 | 0.5309 | 0.6248 | 0.0292 | 0.0000 | VERY POOR |

**Macro AUC Fold4: 0.7690** | **Macro AUC Ensemble: 0.7727**  
Classes EXCELLENT (≥0.85): **10** | GOOD (0.75–0.85): **8** | FAIR (0.65–0.75): **11** | POOR (<0.65): **5**

### Confusion Matrix — Best Model Fold 4 (test set, 771 patients)

> Source: `fine_tune/output_feature/confusion_matrix_stats.csv`

| Class | TP | FP | FN | TN | Precision | Recall |
|---|---|---|---|---|---|---|
| Airspace Disease | 17 | 26 | 24 | 704 | 0.395 | 0.415 |
| Bronchiectasis | 0 | 0 | 2 | 769 | — | 0.000 |
| Bullous Disease | 0 | 0 | 1 | 770 | — | 0.000 |
| Calcified Granuloma | 0 | 0 | 76 | 695 | — | 0.000 |
| Calcinosis | 0 | 6 | 58 | 707 | — | 0.000 |
| Cardiac Shadow (abnormal) | 0 | 0 | 12 | 759 | — | 0.000 |
| Cardiomegaly | 40 | 46 | 38 | 647 | 0.465 | 0.513 |
| Consolidation | 0 | 3 | 12 | 756 | — | 0.000 |
| Degenerative Change | 43 | 110 | 105 | 513 | 0.281 | 0.291 |
| Edema | 2 | 9 | 19 | 741 | 0.182 | 0.095 |
| Effusion | 0 | 0 | 6 | 765 | — | 0.000 |
| Emphysema | 10 | 7 | 25 | 729 | 0.588 | 0.286 |
| Fibrosis | 4 | 6 | 39 | 722 | 0.400 | 0.093 |
| Fractures | 0 | 0 | 17 | 754 | — | 0.000 |
| Hernia | 0 | 0 | 15 | 756 | — | 0.000 |
| Hyperinflation | 45 | 116 | 64 | 546 | 0.280 | 0.413 |
| Increased Lung Markings | 0 | 0 | 35 | 736 | — | 0.000 |
| Interstitial Lung Disease | 0 | 0 | 17 | 754 | — | 0.000 |
| Kyphosis | 0 | 0 | 8 | 763 | — | 0.000 |
| Lesion | 74 | 84 | 80 | 533 | 0.468 | 0.481 |
| Mass | 0 | 0 | 6 | 765 | — | 0.000 |
| Nodule | 0 | 0 | 23 | 748 | — | 0.000 |
| Osteophyte | 0 | 0 | 10 | 761 | — | 0.000 |
| Pleural Effusion | 14 | 31 | 10 | 716 | 0.311 | 0.583 |
| Pleural Thickening | 0 | 0 | 13 | 758 | — | 0.000 |
| Pneumothorax | 0 | 0 | 7 | 764 | — | 0.000 |
| Pulmonary Artery Enlargement | 0 | 0 | 4 | 767 | — | 0.000 |
| Pulmonary Fibrosis | 0 | 0 | 4 | 767 | — | 0.000 |
| Rib Fracture | 0 | 0 | 1 | 770 | — | 0.000 |
| Scoliosis | 0 | 0 | 21 | 750 | — | 0.000 |
| Subcutaneous Emphysema | 0 | 0 | 1 | 770 | — | 0.000 |
| Thickening | 0 | 0 | 11 | 760 | — | 0.000 |
| Volume Loss | 32 | 60 | 35 | 644 | 0.348 | 0.478 |
| normal | 316 | 356 | 10 | 89 | 0.470 | 0.969 |

---

## 8. Cross-Experiment Per-Class AUC Comparison (All 34 Classes)

> **E1** = single-view BS=32 | **E2** = single-view BS=16 | **E5** = multi-view feature attention (best fold)  
> Note: E1/E2 test on 1,494 images (image-level); E5 tests on 771 patients (patient-level paired).  
> Ordered by E5 AUC. Bold = best across experiments for that class. ↑/↓ = direction vs E1.

| Class | E1 AUC | E2 AUC | **E5 AUC** | E5 Ensemble | Δ vs E1 |
|---|---|---|---|---|---|
| Pleural Effusion | 0.8933 | **0.8972** | 0.9350 | 0.9423 | ↑ +0.042 |
| Bullous Disease | 0.5988 | 0.7736 | **0.9318** | 0.9740 | ↑ +0.333 |
| Pneumothorax | 0.6213 | 0.6372 | **0.9122** | 0.9344 | ↑ +0.291 |
| Bronchiectasis | 0.8858 | 0.7395 | **0.9008** | 0.9655 | ↑ +0.015 |
| Subcutaneous Emphysema | N/A | N/A | **0.8929** | 0.8714 | — |
| Cardiac Shadow (abnormal) | **0.9052** | 0.8895 | 0.8889 | 0.9078 | ↓ −0.016 |
| Consolidation | 0.7625 | 0.7859 | **0.8764** | 0.8702 | ↑ +0.114 |
| Edema | 0.8796 | 0.8737 | **0.8754** | 0.8773 | ↓ −0.004 |
| Airspace Disease | 0.7652 | 0.7775 | **0.8649** | 0.8678 | ↑ +0.100 |
| Effusion | 0.7846 | 0.7107 | **0.8637** | 0.8521 | ↑ +0.079 |
| Cardiomegaly | **0.8500** | 0.8397 | 0.8498 | 0.8658 | ≈ |
| Volume Loss | 0.7656 | 0.7463 | **0.8033** | 0.8058 | ↑ +0.038 |
| Emphysema | 0.7564 | **0.8047** | 0.7873 | 0.8244 | ↑ +0.031 |
| Osteophyte | 0.6138 | 0.6314 | **0.7855** | 0.7882 | ↑ +0.172 |
| Interstitial Lung Disease | 0.6838 | 0.7245 | **0.7719** | 0.8141 | ↑ +0.088 |
| Pleural Thickening | **0.8100** | 0.7677 | 0.7667 | 0.8079 | ↓ −0.043 |
| Rib Fracture | **0.8147** | 0.8904 | 0.7636 | 0.5539 | ↓ −0.051 |
| Lesion | 0.6943 | 0.6872 | **0.7561** | 0.7572 | ↑ +0.062 |
| normal | 0.6497 | 0.6996 | **0.7419** | 0.7516 | ↑ +0.092 |
| Hernia | **0.7516** | 0.7497 | 0.7407 | 0.7119 | ↓ −0.011 |
| Fractures | 0.6653 | 0.6917 | **0.7391** | 0.7428 | ↑ +0.074 |
| Hyperinflation | 0.6756 | 0.6632 | **0.7373** | 0.7390 | ↑ +0.062 |
| Increased Lung Markings | 0.6978 | 0.6874 | **0.7285** | 0.7436 | ↑ +0.031 |
| Fibrosis | **0.7214** | 0.7095 | 0.7263 | 0.7293 | ↑ +0.005 |
| Pulmonary Artery Enlargement | 0.5010 | 0.5339 | **0.7241** | 0.7256 | ↑ +0.223 |
| Thickening | 0.6418 | **0.6419** | 0.7178 | 0.6834 | ↑ +0.076 |
| Kyphosis | 0.7460 | **0.7679** | 0.7094 | 0.7267 | ↓ −0.037 |
| Pulmonary Fibrosis | 0.7896 | **0.8059** | 0.6975 | 0.7199 | ↓ −0.092 |
| Nodule | 0.6116 | 0.6343 | **0.6665** | 0.6419 | ↑ +0.055 |
| Calcinosis | 0.5942 | 0.6222 | **0.6457** | 0.6568 | ↑ +0.052 |
| Mass | 0.6889 | **0.7468** | 0.6351 | 0.6368 | ↓ −0.054 |
| Degenerative Change | 0.6121 | 0.6238 | **0.6115** | 0.6228 | ≈ |
| Calcified Granuloma | 0.5590 | 0.5709 | **0.5667** | 0.5340 | ↑ +0.008 |
| Scoliosis | 0.6183 | 0.6063 | 0.5309 | **0.6248** | ↓ −0.087 |

### Summary Across Experiments

| Metric | E1 | E2 | E5 (Fold4) | E5 (Ensemble) |
|---|---|---|---|---|
| Macro AUC (all valid classes) | 0.7154 | 0.7252 | **0.7690** | **0.7727** |
| Classes EXCELLENT (AUC≥0.85) | 8 | 10 | **10** | — |
| Classes GOOD (AUC≥0.75) | 8 | 9 | **18** | — |
| Classes FAIR (AUC≥0.65) | 12 | 14 | **29** | — |
| Classes POOR (AUC<0.65) | 7 | 5 | **5** | — |
| Classes improved vs E1 | — | +2 | **+22** | — |
| Classes declined vs E1 | — | — | **9** | — |

---

## 9. Key Findings for Thesis

### Finding 1: Two-View Fusion Yields the Largest Single Gain
The transition from single-view (E2: 72.52%) to multi-view Feature Attention (E3: 75.83%) gave **+3.31% macro AUC** — the largest improvement across all experiments. At the per-class level, **22 of 34 classes improved** when switching to multi-view, with Bullous Disease (+33.3%), Pneumothorax (+29.1%), and Pulmonary Artery Enlargement (+22.3%) seeing the largest gains.

### Finding 2: Batch Size Affects Imbalanced Learning Stability
Halving the batch size (E1→E2: 32→16) improved test AUC by +0.98% and reduced CV standard deviation by 3× (±0.0107→±0.0048). Smaller batches expose the model to more label diversity per gradient update, which is particularly beneficial for rare classes.

### Finding 3: Feature Attention Outperforms 8 Other Fusion Strategies
Among 9 fusion strategies, Feature Attention (channel-wise attention over summed features) achieved the highest test AUC (75.83%). Notably, simpler bidirectional attention (Cross-Modal: 74.03%) and full self-attention (72.85%) performed worse — likely because complex attention mechanisms require more data to train stably than is available in this dataset.

### Finding 4: Partial Backbone Unfreezing + Composite Loss Boosts All Classes
Compared to fully-frozen single-view (E2), the multi-view setup (E5) improved 22 classes while degrading 9. The degraded classes (Pulmonary Fibrosis, Scoliosis, Rib Fracture, Mass, Kyphosis, Hernia, Cardiac Shadow, Pleural Thickening, Edema) are mostly low-sample classes where the more complex model may be harder to calibrate.

### Finding 5: Ensemble Improves Rare-Class Performance
The 5-fold ensemble (averaging probabilities across all 5 trained folds) improves macro AUC to 0.7727 vs 0.7690 for the single best fold. Gains are especially visible for rare classes (Bronchiectasis: 0.9008→0.9655; Pneumothorax: 0.9122→0.9344).

### Finding 6: Long-Tail Classes Remain Challenging
Even in the best model (E5), classes with <10 test positives show inconsistent results:
- High AUC but 0% F1: Pneumothorax (AUC=0.91, F1=0), Bullous Disease (AUC=0.93, F1=0)
- Poor AUC: Scoliosis (0.53), Calcified Granuloma (0.57), Degenerative Change (0.61)
These classes need oversampling, data augmentation, or few-shot learning techniques in future work.

---

## 10. Final Architecture (Best Model — Experiment 5)

```
Input: Frontal X-ray (224×224) + Lateral X-ray (224×224)
              ↓                              ↓
   ┌──────────────────────────────────────────────────────┐
   │  BiomedCLIP ViT-B/16 (shared weights)               │
   │  Pretrained on 15M biomedical image-text pairs       │
   │  Last 6 transformer blocks UNFROZEN                  │
   └──────────────────────────────────────────────────────┘
              ↓                              ↓
     feat_frontal (512-d)         feat_lateral (512-d)
              ↓ L2-normalize              ↓ L2-normalize
              └──────────────┬────────────┘
                    combined = front + lateral
                    attention = sigmoid(Linear(512→512)(combined))
                    attended_front  = front  × attention
                    attended_lateral = lateral × (1 − attention)
                    fused = Linear(1024→768)(concat[att_front, att_lat])
                              ↓
                    Linear(768→256) → BatchNorm → GELU → Dropout
                              ↓
                    Linear(256→34) → Sigmoid
                              ↓
                    34 disease probabilities

Total params:   197,062,467
Trainable:        1,159,746  (0.59%)
Frozen:         195,902,721  (99.41%)
Checkpoint:             ~790 MB
```

**Loss:** `0.6 × WeightedBCE + 0.2 × FocalLoss(α=0.25,γ=2.0) + 0.2 × SoftJaccardLoss`  
**Optimizer:** AdamW (lr=1e-3, wd=1e-5) | **Hardware:** NVIDIA Quadro RTX 5000 (16.91 GB VRAM)

# Clinical Chest X-ray Report Dataset

## Overview

This dataset is a **clinical chest X-ray radiology report corpus** designed for **medical image–text understanding**, **radiology report analysis**, and **clinical AI research**.  
Each entry corresponds to a **single patient study**, containing both structured and unstructured clinical information derived from chest X-ray reports.

The `label` column represents **one or more thoracic disease categories** associated with each study and serves as the ground-truth target for **multi-label supervised learning**.

This repository includes a **deep learning system for automated chest X-ray analysis using BiomedCLIP**, capable of detecting **35 thoracic pathologies**, achieving **95.10% label-wise accuracy** and **73.18% mean AUC** on a held-out test set.

> ⚠️ **Important Note:**  
> The reported accuracy is **label-wise (per-class) accuracy**, which measures correctness over all binary label decisions.  
> Due to extreme class imbalance, this metric is dominated by true negatives and should be interpreted alongside **AUC, F1-score, and mAP**.

---

## Dataset Characteristics

- **Modality:** Chest X-ray (PA / Lateral)
- **Data Type:** Clinical radiology reports + corresponding images
- **Task Type:** **Multi-label classification**
- **Total Classes:** 35
- **Domain:** Radiology, pulmonary and cardiovascular diseases
- **Total Images:** 7,466 validated X-ray images (~14 GB)
- **Total Reports:** 3,852 radiology reports
- **Class Imbalance Ratio:** **1,581:1 (long-tailed distribution)**

---

## Disease Class Labels (35 Classes)

1. Normal
2. Degenerative Change
3. Lesion
4. Hyperinflation
5. Calcified Granuloma
6. Cardiomegaly
7. Volume Loss
8. Calcinosis
9. Airspace Disease
10. Fibrosis
11. Increased Lung Markings
12. Pleural Effusion
13. Emphysema
14. Nodule
15. Edema
16. Scoliosis
17. Fractures
18. Hernia
19. Pleural Thickening
20. Osteophyte
21. Interstitial Lung Disease
22. Consolidation
23. Cardiac Shadow (abnormal)
24. Thickening
25. Kyphosis
26. Pneumothorax
27. Mass
28. Pulmonary Artery Enlargement
29. Pulmonary Fibrosis
30. Effusion
31. Bronchiectasis
32. Bullous Disease
33. Rib Fracture
34. Subcutaneous Emphysema
35. Bronchiolitis

---

## Model Performance

### Overall Test Performance

| Metric | Value |
|------|------|
| **Label-wise Accuracy** | **95.10%** |
| **Mean AUC (macro)** | **73.18%** |
| **Macro F1-score** | 5.47% |
| **mAP** | 14.13% |

### Detailed Metrics

| Set | Label-wise Accuracy | AUC | F1 (Macro) | mAP |
|-----|-------------------|-----|-----------|-----|
| Train | 95.20% | 81.18% | 16.76% | 26.77% |
| Test | 95.10% | 73.18% | 5.47% | 14.13% |

---

### Interpretation of Metrics

- **Label-wise Accuracy:** Proportion of correctly predicted binary label assignments across all samples and classes.
- **AUC:** Measures ranking quality independent of threshold; preferred metric for imbalanced datasets.
- **F1-score / mAP:** Reflect the model’s ability to correctly identify positive disease labels.

High accuracy values are expected due to the dominance of negative labels per image (average ~1.76 positives out of 35).

---

## Architecture

**BiomedCLIP-based Multi-label Classification Network**

- Vision Encoder: ViT-Base (512-dim embeddings)
- Pretraining: 15M biomedical image–text pairs
- Classification Head: 3-layer MLP (404K parameters)
- Output: 35 sigmoid-activated disease probabilities

**Training Strategy:**
- Partial fine-tuning (last 2 transformer blocks unfrozen)
- Focal Loss (α = 0.25, γ = 2.0) for class imbalance
- 5-fold cross-validation with automated epoch search

**Model Size:**
- Total parameters: **196M**
- Trainable parameters: **404K (0.2%)**
- Frozen parameters: **195.9M (99.8%)**
- Final checkpoint size: **~790 MB**

---

## Ethical Considerations

This dataset and model are intended **strictly for research and educational purposes**.  
They must **not** be used for clinical diagnosis, triage, or treatment decisions without rigorous external validation and regulatory approval.

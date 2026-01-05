



# Chest X-Ray Multi-Label Classification

Deep learning system for automated chest X-ray analysis using BiomedCLIP. Detects 24 thoracic pathologies with 96.76% accuracy.


## Performance

| Metric | Value |
|--------|-------|
| **Test Accuracy** | 96.76% |
| **Test AUC** | 77.19% |
| **Best Model** | `epoch_search_e100_fold3.pth` |

**Top Conditions Detected:**
- Bronchiectasis (99.8% AUC)
- Bullous Disease (96.6% AUC)
- Edema (91.9% AUC)
- Pleural Effusion (88.2% AUC)
- Cardiomegaly (86.6% AUC)

## Architecture

```
BiomedCLIP (Frozen, 150M params)
    └── ViT-Base Vision Encoder (768-dim embeddings)
         └── Classification Head (3 layers, 580K params)
              └── 24 output classes
```

**Key Features:**
- Pre-trained on 15M biomedical image-text pairs
- Focal Loss (α=0.25, γ=2.0) handles class imbalance
- 5-fold cross-validation with automated epoch search
- 80/20 train-test split

## Dataset

**Indiana University Chest X-Ray Collection:**
- 7,466 valid images (14GB)
- 3,852 radiology reports
- 24 pathology labels (multi-label)
- Extreme class imbalance (476:1 ratio)

**Structure:**
```
archive/
├── indiana_reports_with_labels.csv
├── indiana_projections.csv
└── files/
    └── *.png (7,470 images)
```

## Training

### Two-Phase Pipeline

**Phase 1: Epoch Search**
- Tests [30, 50, 70, 90, 100] epochs
- 5-fold cross-validation per configuration
- Selects optimal: **100 epochs** (0.7676 AUC ± 0.017)

**Phase 2: Final Evaluation**
- Best checkpoint: Fold 3, 100 epochs
- Held-out test set: 1,494 images
- Comprehensive metrics + visualizations

### Training Options

```bash
python Xray.py \
    --mode train \              # train | evaluate | both
    --csv_dir archive \
    --image_dir archive/files \
    --batch_size 16 \           # Reduce to 8 if OOM
    --num_workers 8 \
    --lr 2e-5
```

**Hyperparameters:**
- Optimizer: AdamW (lr=2e-5, weight_decay=1e-4)
- Scheduler: ReduceLROnPlateau (factor=0.5, patience=3)
- Loss: Focal Loss (α=0.25, γ=2.0)
- Early stopping: patience=5
- Gradient clipping: max_norm=1.0



### Threshold Guidelines

| Threshold | Use Case |
|-----------|----------|
| 0.2 | High sensitivity screening |
| **0.3** | **Recommended default** |
| 0.5 | High confidence only |

## Results

### Overall Metrics

| Set | Accuracy | AUC | F1 | mAP |
|-----|----------|-----|-----|-----|
| Train | 96.83% | 85.62% | 16.14% | 28.72% |
| Test | 96.76% | 77.19% | 11.22% | 19.38% |

### Epoch Search Results

| Epochs | Mean AUC | Std |
|--------|----------|-----|
| 30 | 0.7295 | 0.027 |
| 50 | 0.7515 | 0.020 |
| 70 | 0.7593 | 0.012 |
| 90 | 0.7605 | 0.016 |
| **100** ✓ | **0.7676** | **0.017** |



**Low Performance on Some Classes:**
- Expected due to extreme imbalance
- Check `evaluation_results.csv` for per-class metrics
- Some classes have <10 test samples

## Technical Details

**Model Size:**
- Frozen backbone: 150M parameters
- Trainable head: 580K parameters
- Checkpoint: 752MB

**Training Time (on GPU):**
- Single epoch: ~40 seconds
- Full epoch search: ~8-10 hours
- Inference: ~50-100ms per image

**Data Augmentation:**
- Resize to 224×224
- Random horizontal flip (training)
- Normalization (ImageNet stats)

## Limitations

- Not approved for clinical use
-  Requires validation by medical professionals
- Low F1 scores due to extreme class imbalance
- Some classes have insufficient test samples
- Conservative predictions may miss rare conditions


## Dependencies

```
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
pandas>=2.0.0
Pillow>=9.5.0
scikit-learn>=1.3.0
tqdm>=4.65.0
transformers>=4.30.0
open-clip-torch>=2.20.0
matplotlib>=3.7.0
seaborn>=0.12.0
```

## References

- **BiomedCLIP:** [microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224)
- **Focal Loss:** [Lin et al., 2017](https://arxiv.org/abs/1708.02002)
- **Dataset:** Indiana University Chest X-Ray Collection


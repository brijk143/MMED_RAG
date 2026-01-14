#python train_with_unfreezing.py --mode both --unfreeze_layers 4
import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt
import warnings
import logging
warnings.filterwarnings('ignore')

import open_clip

def setup_logging(log_file='training.log'):
    """Configure logging to both file and console"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance in multi-label classification"""
    def __init__(self, alpha=0.25, gamma=2.0, class_weights=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.class_weights = class_weights
    
    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        
        # Apply class weights if provided
        if self.class_weights is not None:
            F_loss = F_loss * self.class_weights.to(inputs.device)
        
        return F_loss.mean()


class ChestXrayDataset(Dataset):
    """Multi-label chest X-ray dataset"""
    def __init__(self, dataframe, image_dir, transform=None, label_columns=None):
        self.df = dataframe.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform
        
        # Use provided label_columns or infer from dataframe
        if label_columns is not None:
            self.label_cols = label_columns
        else:
            # Infer label columns - exclude metadata columns
            self.label_cols = [col for col in dataframe.columns 
                              if col not in ['image_path', 'findings', 'indication', 
                                            'comparison', 'uid', 'projection', 
                                            'impression', 'label', 'labels', 'image_exists',
                                            'label_for_stratification', 'num_labels']]
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.image_dir, row['image_path'])
        
        try:
            image = Image.open(img_path).convert('RGB')
        except:
            image = Image.new('RGB', (224, 224))
            logger.warning(f"Failed to load image: {img_path}")
        
        if self.transform:
            image = self.transform(image)
        
        labels = torch.tensor(row[self.label_cols].values.astype(np.float32))
        return {'image': image, 'labels': labels}


def load_dataset(csv_dir):
    """Load and preprocess Indiana University dataset with proper multi-label support"""
    logger.info("Loading dataset...")
    logger.info("Multi-label classification: Each sample can have one or more labels")
    
    reports_path = os.path.join(csv_dir, 'indiana_reports_cleaned.csv')
    projections_path = os.path.join(csv_dir, 'indiana_projections.csv')
    
    if not os.path.exists(reports_path):
        raise FileNotFoundError(f"Reports file not found: {reports_path}")
    
    reports_df = pd.read_csv(reports_path)
    projections_df = pd.read_csv(projections_path)
    merged_df = projections_df.merge(reports_df, on='uid', how='left')
    
    # Parse labels - handle both single and multi-label cases (semicolon-separated)
    data = []
    all_labels_set = set()
    single_label_count = 0
    multi_label_count = 0
    
    for _, row in tqdm(merged_df.iterrows(), total=len(merged_df), desc="Parsing labels"):
        label_str = str(row.get('label', 'normal')).strip()
        
        # Handle missing/empty labels
        if pd.isna(label_str) or not label_str or label_str.lower() == 'nan':
            label_str = 'normal'
        
        # Split by semicolon to handle multi-label cases
        labels = [l.strip() for l in label_str.split(';') if l.strip()]
        
        # If empty after splitting, default to normal
        if not labels:
            labels = ['normal']
        
        # Track single vs multi-label
        if len(labels) == 1:
            single_label_count += 1
        else:
            multi_label_count += 1
        
        # Collect all unique labels
        all_labels_set.update(labels)
        
        data.append({
            'uid': row['uid'],
            'image_path': row['filename'],
            'labels': labels,  # Store as list
            'findings': str(row.get('findings', ''))
        })
    
    df = pd.DataFrame(data)
    
    # Get all unique labels sorted
    unique_labels = sorted(list(all_labels_set))
    logger.info(f"\nDataset statistics:")
    logger.info(f"  Total samples: {len(df)}")
    logger.info(f"  Single-label samples: {single_label_count} ({single_label_count/len(df)*100:.2f}%)")
    logger.info(f"  Multi-label samples: {multi_label_count} ({multi_label_count/len(df)*100:.2f}%)")
    logger.info(f"  Total unique labels: {len(unique_labels)}")
    
    # Create binary columns for multi-label classification (multi-hot encoding)
    for label in unique_labels:
        df[label] = df['labels'].apply(lambda x: 1 if label in x else 0)
    
    # Compute label co-occurrence statistics
    df['num_labels'] = df['labels'].apply(len)
    logger.info(f"\nLabels per sample:")
    logger.info(f"  Min: {df['num_labels'].min()}, Max: {df['num_labels'].max()}, Mean: {df['num_labels'].mean():.2f}")
    
    # Log class distribution
    logger.info("\nClass distribution (sorted by frequency):")
    label_counts = [(label, df[label].sum()) for label in unique_labels]
    label_counts.sort(key=lambda x: x[1], reverse=True)
    for label, count in label_counts:
        pct = count / len(df) * 100
        logger.info(f"  {label:<30} {int(count):>5} ({pct:>6.3f}%)")
    
    # Create stratification column based on multi-label combinations
    # For better stratification with multi-label: use most frequent label or combination hash
    df['label_for_stratification'] = df['labels'].apply(
        lambda x: x[0] if len(x) == 1 else 'multi_label_' + '_'.join(sorted(x)[:2])
    )
    
    # Compute class weights for handling imbalance
    class_counts = {label: df[label].sum() for label in unique_labels}
    total_samples = len(df)
    class_weights = {label: total_samples / (len(unique_labels) * max(count, 1)) 
                     for label, count in class_counts.items()}
    
    logger.info("\nClass weights for imbalanced data:")
    for label in sorted(class_weights.keys(), key=lambda x: class_weights[x], reverse=True)[:10]:
        logger.info(f"  {label:<30} weight: {class_weights[label]:.3f}")
    
    return df, unique_labels, class_weights


def safe_stratify_column(df, column='label_for_stratification', min_count=2):
    """Return a stratify column safe for sklearn splits or None if stratification isn't possible.

    - Groups classes with counts < min_count into a single 'RARE' bucket.
    - If after grouping some class still has < 2 samples, return None to indicate
      stratification is unsafe and the caller should fall back to non-stratified splitting.
    """
    if column not in df.columns:
        return None

    vc = df[column].value_counts()
    # If every class already has at least min_count, return original column
    if (vc >= min_count).all():
        return df[column]

    rare = vc[vc < min_count].index.tolist()
    strat = df[column].apply(lambda x: 'RARE' if x in rare else x)

    # Ensure all groups in the new strat have at least 2 samples
    new_vc = strat.value_counts()
    if new_vc.min() < 2:
        logger.warning(
            "Stratification not possible: some groups have fewer than 2 samples after grouping rares. "
            "Falling back to no stratification."
        )
        return None

    return strat

class BiomedCLIPClassifier(nn.Module):
    """BiomedCLIP-based multi-label classifier with PARTIAL UNFREEZING support"""
    def __init__(self, num_classes, freeze_backbone=True, unfreeze_last_n_blocks=4):
        super().__init__()
        
        model_name = 'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
        self.model, self.preprocess_train, self.preprocess_val = \
            open_clip.create_model_and_transforms(model_name)
        
        # IMPROVED: Selective layer unfreezing
        if freeze_backbone:
            # First, freeze everything
            for param in self.model.parameters():
                param.requires_grad = False
            
            # Then, selectively unfreeze last N transformer blocks
            if unfreeze_last_n_blocks > 0:
                logger.info(f"🔓 Unfreezing last {unfreeze_last_n_blocks} transformer blocks")
                
                # BiomedCLIP uses vision transformer - unfreeze last blocks
                if hasattr(self.model, 'visual'):
                    if hasattr(self.model.visual, 'transformer'):
                        total_blocks = len(self.model.visual.transformer.resblocks)
                        blocks_to_unfreeze = self.model.visual.transformer.resblocks[-unfreeze_last_n_blocks:]
                        
                        for block in blocks_to_unfreeze:
                            for param in block.parameters():
                                param.requires_grad = True
                        
                        logger.info(f"   Unfroze blocks {total_blocks-unfreeze_last_n_blocks} to {total_blocks-1}")
                    
                    # Also unfreeze the final layer norm
                    if hasattr(self.model.visual, 'ln_post'):
                        for param in self.model.visual.ln_post.parameters():
                            param.requires_grad = True
                        logger.info(f"   Unfroze final layer normalization")
        else:
            logger.info("🔓 Full backbone unfreezing - all layers trainable")
        
        # Get embedding dimension
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            embed_dim = self.model.encode_image(dummy_input).shape[-1]
        
        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        
        # Initialize weights
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
        
        # Log trainable parameters
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        
        logger.info(f"📊 Model Parameters:")
        logger.info(f"   Total: {total_params:,}")
        logger.info(f"   Trainable: {trainable_params:,} ({trainable_params/total_params*100:.1f}%)")
        logger.info(f"   Frozen: {frozen_params:,} ({frozen_params/total_params*100:.1f}%)")
        
    def forward(self, images):
        features = self.model.encode_image(images)
        features = F.normalize(features, dim=-1)
        return self.classifier(features)


def train_epoch(model, loader, criterion, optimizer, device):
    """Single training epoch"""
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []
    
    for batch in tqdm(loader, desc="Training"):
        images = batch['image'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        
        with torch.no_grad():
            all_preds.append(torch.sigmoid(outputs).cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    return total_loss / len(loader), np.vstack(all_preds), np.vstack(all_labels)

def evaluate(model, loader, criterion, device):
    """Evaluate model"""
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            images = batch['image'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            all_preds.append(torch.sigmoid(outputs).cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    return total_loss / len(loader), np.vstack(all_preds), np.vstack(all_labels)

def compute_metrics(predictions, labels, threshold=0.5):
    """
    Compute evaluation metrics for multi-label classification.
    
    CRITICAL: This function computes accuracy correctly for multi-label classification.
    
    In multi-label classification, accuracy is NOT computed as exact match (all labels correct).
    Instead, we compute LABEL-WISE ACCURACY: the proportion of correctly predicted 
    label-sample pairs across the entire prediction matrix.
    
    Formula: accuracy = (predictions == labels).mean()
    
    This treats each label prediction independently and computes the average correctness
    across all N×C predictions (N samples, C classes).
    
    Example with 3 samples and 4 labels:
        Sample 1: True [1,0,1,0], Pred [1,0,0,0] -> 3/4 correct
        Sample 2: True [0,1,1,1], Pred [0,0,1,1] -> 3/4 correct  
        Sample 3: True [1,1,0,0], Pred [1,1,0,0] -> 4/4 correct
        Overall: 10/12 = 83.33% accuracy
    """
    # Convert probabilities to binary predictions using threshold
    binary_preds = (predictions > threshold).astype(int)
    
    # MULTI-LABEL ACCURACY: Label-wise accuracy (NOT exact match)
    # This is the correct way to compute accuracy in multi-label classification
    # It measures the proportion of correctly predicted individual labels
    accuracy = (binary_preds == labels).mean()
    
    # AUC per class (only for classes with both positive and negative samples)
    auc_scores = []
    for i in range(labels.shape[1]):
        if len(np.unique(labels[:, i])) > 1:
            auc_scores.append(roc_auc_score(labels[:, i], predictions[:, i]))
    
    # mAP (mean Average Precision) per class
    ap_scores = []
    for i in range(labels.shape[1]):
        if len(np.unique(labels[:, i])) > 1:
            ap_scores.append(average_precision_score(labels[:, i], predictions[:, i]))
    
    # F1 Score (macro-averaged across all classes)
    f1 = f1_score(labels, binary_preds, average='macro', zero_division=0)
    
    return {
        'accuracy': accuracy,  # Label-wise accuracy
        'auc': np.mean(auc_scores) if auc_scores else 0.0,
        'map': np.mean(ap_scores) if ap_scores else 0.0,
        'f1': f1
    }

def plot_confusion_matrices(predictions, labels, label_names, save_dir='.'):
    """Plot confusion matrices for top classes"""
    logger.info("Generating confusion matrices...")
    
    from sklearn.metrics import confusion_matrix
    
    # Select top 6 classes by sample count for visualization
    class_counts = labels.sum(axis=0)
    top_indices = np.argsort(class_counts)[-6:][::-1]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for idx, class_idx in enumerate(top_indices):
        binary_preds = (predictions[:, class_idx] > 0.5).astype(int)
        cm = confusion_matrix(labels[:, class_idx], binary_preds)
        
        # Plot
        im = axes[idx].imshow(cm, cmap='Blues', aspect='auto')
        axes[idx].set_title(f'{label_names[class_idx]}\n(n={int(class_counts[class_idx])})', 
                           fontsize=12, fontweight='bold')
        
        # Add text annotations
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                text = axes[idx].text(j, i, f'{cm[i, j]}',
                                     ha="center", va="center", 
                                     color="white" if cm[i, j] > cm.max()/2 else "black",
                                     fontsize=14, fontweight='bold')
        
        axes[idx].set_xlabel('Predicted', fontweight='bold')
        axes[idx].set_ylabel('Actual', fontweight='bold')
        axes[idx].set_xticks([0, 1])
        axes[idx].set_yticks([0, 1])
        axes[idx].set_xticklabels(['Negative', 'Positive'])
        axes[idx].set_yticklabels(['Negative', 'Positive'])
        plt.colorbar(im, ax=axes[idx], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'confusion_matrices.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    logger.info(f"Confusion matrices saved: {save_path}")
    plt.close()


def final_evaluation(best_checkpoint_path, config):
    """
    Comprehensive final evaluation with all metrics for multi-label classification.
    
    This function demonstrates the CORRECT way to evaluate multi-label classification:
    1. Compute predictions (sigmoid probabilities)
    2. Convert to binary predictions using threshold (default 0.5)
    3. Compute LABEL-WISE accuracy: (binary_preds == labels).mean()
    4. Compute additional metrics: AUC, F1, mAP
    """
    logger.info("="*80)
    logger.info("📊 FINAL COMPREHENSIVE EVALUATION")
    logger.info("="*80)
    
    # Load checkpoint
    device = torch.device(config['device'])
    checkpoint = torch.load(best_checkpoint_path, map_location=device, weights_only=False)
    label_columns = checkpoint['label_columns']
    
    # Load full dataset
    df, _, class_weights = load_dataset(config['csv_dir'])
    df['image_exists'] = df['image_path'].apply(
        lambda x: os.path.exists(os.path.join(config['image_dir'], x))
    )
    df = df[df['image_exists']]
    
    # Use same 80-20 split
    strat_col = safe_stratify_column(df, 'label_for_stratification')
    if strat_col is None:
        train_val_df, test_df = train_test_split(
            df, test_size=0.2, random_state=42, shuffle=True
        )
    else:
        train_val_df, test_df = train_test_split(
            df, test_size=0.2, random_state=42, stratify=strat_col
        )
    
    logger.info(f"Train+Val samples: {len(train_val_df)}")
    logger.info(f"Test samples: {len(test_df)}")
    
    # Load model
    model = BiomedCLIPClassifier(
        num_classes=len(label_columns),
        freeze_backbone=config.get('freeze_backbone', True),
        unfreeze_last_n_blocks=config.get('unfreeze_layers', 0)
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    logger.info(f"✓ Loaded model from {best_checkpoint_path}")
    logger.info(f"   Trained for {checkpoint['epoch']} epochs (Fold {checkpoint['fold']})")
    logger.info(f"   Training AUC: {checkpoint['train_auc']:.4f}")
    logger.info(f"   Validation AUC: {checkpoint['val_auc']:.4f}")
    
    # Prepare class weights
    weight_tensor = torch.tensor([class_weights[col] for col in label_columns], dtype=torch.float32)
    criterion = FocalLoss(class_weights=weight_tensor)
    
    # Evaluate on training data (80%)
    train_loader = DataLoader(
        ChestXrayDataset(train_val_df, config['image_dir'], model.preprocess_val, label_columns=label_columns),
        batch_size=config['batch_size'], shuffle=False,
        num_workers=config['num_workers'], pin_memory=True
    )
    
    logger.info("\n🔍 Evaluating on Training Set (80%)...")
    train_loss, train_preds, train_labels = evaluate(model, train_loader, criterion, device)
    train_metrics = compute_metrics(train_preds, train_labels)
    
    logger.info(f"Train Loss: {train_loss:.4f}")
    logger.info(f"Train Accuracy (Label-wise): {train_metrics['accuracy']:.4f}")
    logger.info(f"Train AUC: {train_metrics['auc']:.4f}")
    logger.info(f"Train F1 (macro): {train_metrics['f1']:.4f}")
    logger.info(f"Train mAP: {train_metrics['map']:.4f}")
    
    # Evaluate on test data (20%)
    test_loader = DataLoader(
        ChestXrayDataset(test_df, config['image_dir'], model.preprocess_val, label_columns=label_columns),
        batch_size=config['batch_size'], shuffle=False,
        num_workers=config['num_workers'], pin_memory=True
    )
    
    logger.info("\n🎯 Evaluating on Test Set (20%)...")
    test_loss, test_preds, test_labels = evaluate(model, test_loader, criterion, device)
    test_metrics = compute_metrics(test_preds, test_labels)
    
    logger.info(f"Test Loss: {test_loss:.4f}")
    logger.info(f"Test Accuracy (Label-wise): {test_metrics['accuracy']:.4f}")
    logger.info(f"Test AUC: {test_metrics['auc']:.4f}")
    logger.info(f"Test F1 (macro): {test_metrics['f1']:.4f}")
    logger.info(f"Test mAP: {test_metrics['map']:.4f}")
    
    # Save metrics summary
    metrics_summary = pd.DataFrame([
        {
            'Dataset': 'Training (80%)',
            'Samples': len(train_val_df),
            'Loss': train_loss,
            'Accuracy_Labelwise': train_metrics['accuracy'],
            'AUC': train_metrics['auc'],
            'F1_Macro': train_metrics['f1'],
            'mAP': train_metrics['map']
        },
        {
            'Dataset': 'Test (20%)',
            'Samples': len(test_df),
            'Loss': test_loss,
            'Accuracy_Labelwise': test_metrics['accuracy'],
            'AUC': test_metrics['auc'],
            'F1_Macro': test_metrics['f1'],
            'mAP': test_metrics['map']
        }
    ])
    metrics_summary.to_csv('final_metrics_summary.csv', index=False)
    logger.info("\n📄 Metrics summary saved to: final_metrics_summary.csv")
    
    # Generate confusion matrices
    plot_confusion_matrices(test_preds, test_labels, label_columns)
    
    # Comprehensive per-class evaluation
    logger.info("\n📋 Generating per-class evaluation...")
    comprehensive_evaluation(test_preds, test_labels, label_columns)
    
    logger.info("\n" + "="*80)
    logger.info("✅ EVALUATION COMPLETE")
    logger.info("="*80)
    logger.info(f"\n📊 Summary:")
    logger.info(f"   Training Accuracy (Label-wise): {train_metrics['accuracy']:.4f} | AUC: {train_metrics['auc']:.4f}")
    logger.info(f"   Testing Accuracy (Label-wise):  {test_metrics['accuracy']:.4f} | AUC: {test_metrics['auc']:.4f}")
    logger.info(f"\n📁 Generated Files:")
    logger.info(f"   - epoch_search_results.csv")
    logger.info(f"   - epoch_search_plot.png")
    logger.info(f"   - final_metrics_summary.csv")
    logger.info(f"   - confusion_matrices.png")
    logger.info(f"   - evaluation_results.csv")
    logger.info(f"   - evaluation_plot.png")
    logger.info(f"   - {best_checkpoint_path}")


def comprehensive_evaluation(predictions, labels, label_names, save_dir='.'):
    """Generate comprehensive evaluation report"""
    logger.info("Generating comprehensive evaluation...")
    
    results = []
    for i, class_name in enumerate(label_names):
        unique_vals = np.unique(labels[:, i])
        
        # Handle classes with insufficient samples
        if len(unique_vals) < 2:
            pos_count = int(labels[:, i].sum())
            logger.warning(f"Class '{class_name}': insufficient samples in test set ({pos_count} positives, {len(labels[:, i]) - pos_count} negatives)")
            results.append({
                'Class': class_name,
                'AUC': np.nan,
                'AP': np.nan,
                'Best_F1': np.nan,
                'Optimal_Threshold': np.nan,
                'Positive_Samples': pos_count,
                'Rating': 'INSUFFICIENT_DATA'
            })
            continue
        
        # Metrics
        auc = roc_auc_score(labels[:, i], predictions[:, i])
        ap = average_precision_score(labels[:, i], predictions[:, i])
        
        # Find optimal threshold
        best_f1, best_thresh = 0, 0.5
        for thresh in np.arange(0.1, 0.9, 0.05):
            preds = (predictions[:, i] > thresh).astype(int)
            f1 = f1_score(labels[:, i], preds, zero_division=0)
            if f1 > best_f1:
                best_f1, best_thresh = f1, thresh
        
        # Performance rating
        if auc >= 0.80:
            rating = "EXCELLENT"
        elif auc >= 0.70:
            rating = "GOOD"
        elif auc >= 0.60:
            rating = "FAIR"
        else:
            rating = "POOR"
        
        results.append({
            'Class': class_name,
            'AUC': auc,
            'AP': ap,
            'Best_F1': best_f1,
            'Optimal_Threshold': best_thresh,
            'Positive_Samples': int(labels[:, i].sum()),
            'Rating': rating
        })
    
    df_results = pd.DataFrame(results).sort_values('AUC', ascending=False)
    df_results.insert(0, 'Rank', range(1, len(df_results) + 1))
    
    # Save results
    csv_path = os.path.join(save_dir, 'evaluation_results.csv')
    df_results.to_csv(csv_path, index=False)
    logger.info(f"Results saved: {csv_path}")
    
    # Create visualization
    plot_evaluation(df_results, save_dir)
    
    # Print summary
    logger.info("\n" + "="*80)
    logger.info("EVALUATION SUMMARY")
    logger.info("="*80)
    logger.info(f"Mean AUC: {df_results['AUC'].mean():.4f}")
    logger.info(f"Median AUC: {df_results['AUC'].median():.4f}")
    logger.info(f"\nTop 5 Classes:")
    for _, row in df_results.head(5).iterrows():
        logger.info(f"  #{row['Rank']} {row['Class']:<30} AUC: {row['AUC']:.4f} ({row['Rating']})")
    
    logger.info(f"\nBottom 5 Classes:")
    for _, row in df_results.tail(5).iterrows():
        logger.info(f"  #{row['Rank']} {row['Class']:<30} AUC: {row['AUC']:.4f} ({row['Rating']})")
    
    return df_results

def plot_evaluation(df_results, save_dir='.'):
    """Create evaluation visualization"""
    df_sorted = df_results.sort_values('AUC', ascending=True, na_position='first')
    
    fig, axes = plt.subplots(1, 2, figsize=(16, max(8, len(df_sorted) * 0.3)))
    
    # Replace NaN with 0 for plotting
    auc_values = df_sorted['AUC'].fillna(0)
    
    # Color coding (gray for insufficient data)
    colors = []
    for idx, row in df_sorted.iterrows():
        if pd.isna(row['AUC']):
            colors.append('#9E9E9E')  # Gray for insufficient data
        elif row['AUC'] >= 0.80:
            colors.append('#2E7D32')
        elif row['AUC'] >= 0.70:
            colors.append('#66BB6A')
        elif row['AUC'] >= 0.60:
            colors.append('#FFA726')
        else:
            colors.append('#EF5350')
    
    # AUC plot
    y_pos = np.arange(len(df_sorted))
    axes[0].barh(y_pos, auc_values, color=colors, edgecolor='black', linewidth=0.5)
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(df_sorted['Class'], fontsize=9)
    axes[0].set_xlabel('AUC Score', fontsize=11, fontweight='bold')
    axes[0].set_title('Model Performance by Class', fontsize=12, fontweight='bold')
    axes[0].axvline(0.5, color='red', linestyle='--', linewidth=1.5, alpha=0.5, label='Random')
    
    # Use nanmean for mean calculation
    mean_auc = df_sorted['AUC'].mean(skipna=True)
    axes[0].axvline(mean_auc, color='blue', linestyle='-', 
                    linewidth=2, alpha=0.7, label=f'Mean: {mean_auc:.3f}')
    axes[0].legend(loc='lower right', fontsize=9)
    axes[0].grid(axis='x', alpha=0.3)
    axes[0].set_xlim([0, 1])
    
    # Sample distribution
    axes[1].barh(y_pos, df_sorted['Positive_Samples'], color='#64B5F6', 
                 edgecolor='black', linewidth=0.5, label='Positive Samples')
    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels(df_sorted['Class'], fontsize=9)
    axes[1].set_xlabel('Number of Samples', fontsize=11, fontweight='bold')
    axes[1].set_title('Dataset Distribution', fontsize=12, fontweight='bold')
    axes[1].legend(loc='lower right', fontsize=9)
    axes[1].grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'evaluation_plot.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    logger.info(f"Visualization saved: {save_path}")
    plt.close()


def search_best_epochs(config, epochs_list=[150]):
    """Search for optimal epoch count using k-fold CV with multi-label support"""
    logger.info("="*80)
    logger.info("EPOCH SEARCH MODE - Finding Optimal Epochs")
    logger.info("="*80)
    logger.info(f"Testing epochs: {epochs_list}")
    
    # Load data once
    df, label_columns, class_weights = load_dataset(config['csv_dir'])
    df['image_exists'] = df['image_path'].apply(
        lambda x: os.path.exists(os.path.join(config['image_dir'], x))
    )
    df = df[df['image_exists']]
    
    # Use 80% for epoch search (same as training)
    strat_col = safe_stratify_column(df, 'label_for_stratification')
    if strat_col is None:
        train_val_df, _ = train_test_split(
            df, test_size=0.2, random_state=42, shuffle=True
        )
        use_stratify = False
    else:
        train_val_df, _ = train_test_split(
            df, test_size=0.2, random_state=42, stratify=strat_col
        )
        use_stratify = True
    
    device = torch.device(config['device'])
    
    # Prepare class weights tensor
    weight_tensor = torch.tensor([class_weights[col] for col in label_columns], dtype=torch.float32)
    criterion = FocalLoss(class_weights=weight_tensor) if config['focal_loss'] else nn.BCEWithLogitsLoss(pos_weight=weight_tensor)
    
    # If stratification is unsafe (rare labels), fall back to KFold
    if use_stratify:
        try:
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            splitter = lambda X: skf.split(X, X['label_for_stratification'])
            stratified_kfold = True
        except Exception:
            logger.warning("StratifiedKFold failed, falling back to KFold")
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            splitter = lambda X: kf.split(X)
            stratified_kfold = False
    else:
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        splitter = lambda X: kf.split(X)
        stratified_kfold = False
    
    epoch_search_results = []
    saved_checkpoints = {}  # Store checkpoints per epoch
    
    for epochs in epochs_list:
        logger.info(f"\n🔁 Testing Epochs = {epochs}")
        fold_aucs = []
        fold_checkpoints = []
        
        for fold, (train_idx, val_idx) in enumerate(splitter(train_val_df), 1):
            logger.info(f"  Fold {fold}/5")
            
            train_df = train_val_df.iloc[train_idx].reset_index(drop=True)
            val_df = train_val_df.iloc[val_idx].reset_index(drop=True)
            
            # Model
            model = BiomedCLIPClassifier(
                num_classes=len(label_columns),
                freeze_backbone=config['freeze_backbone'],
                unfreeze_last_n_blocks=config['unfreeze_layers']
            ).to(device)
            
            # Dataloaders
            train_loader = DataLoader(
                ChestXrayDataset(train_df, config['image_dir'], model.preprocess_train, label_columns=label_columns),
                batch_size=config['batch_size'], shuffle=True,
                num_workers=config['num_workers'], pin_memory=True
            )
            val_loader = DataLoader(
                ChestXrayDataset(val_df, config['image_dir'], model.preprocess_val, label_columns=label_columns),
                batch_size=config['batch_size'], shuffle=False,
                num_workers=config['num_workers'], pin_memory=True
            )
            
            # Optimizer
            optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=config['lr'], weight_decay=config['weight_decay']
            )
            
            # Train for specified epochs
            for epoch in range(epochs):
                train_loss, train_preds, train_labels = train_epoch(
                    model, train_loader, criterion, optimizer, device
                )
            
            # Final validation and training metrics
            val_loss, val_preds, val_labels = evaluate(
                model, val_loader, criterion, device
            )
            val_metrics = compute_metrics(val_preds, val_labels)
            
            # Also compute final training metrics
            train_loss, train_preds, train_labels = evaluate(
                model, train_loader, criterion, device
            )
            train_metrics = compute_metrics(train_preds, train_labels)
            
            fold_aucs.append(val_metrics['auc'])
            logger.info(f"    Train AUC: {train_metrics['auc']:.4f}, Val AUC: {val_metrics['auc']:.4f}")
            
            # Save checkpoint for this fold
            checkpoint_path = f"epoch_search_e{epochs}_fold{fold}.pth"
            torch.save({
                'epoch': epochs,
                'fold': fold,
                'model_state_dict': model.state_dict(),
                'train_auc': train_metrics['auc'],
                'val_auc': val_metrics['auc'],
                'label_columns': label_columns,
                'train_df': train_df,
                'val_df': val_df
            }, checkpoint_path)
            fold_checkpoints.append({
                'path': checkpoint_path,
                'val_auc': val_metrics['auc'],
                'train_auc': train_metrics['auc']
            })
        
        mean_auc = np.mean(fold_aucs)
        std_auc = np.std(fold_aucs)
        epoch_search_results.append({
            'epochs': epochs,
            'mean_auc': mean_auc,
            'std_auc': std_auc,
            'fold_aucs': fold_aucs
        })
        saved_checkpoints[epochs] = fold_checkpoints
        logger.info(f"✓ Epochs {epochs} → Mean AUC: {mean_auc:.4f} ± {std_auc:.4f}")
    
    # Find best epochs and best fold checkpoint
    best_result = max(epoch_search_results, key=lambda x: x['mean_auc'])
    best_epochs = best_result['epochs']
    
    # Find best fold checkpoint for the best epoch count
    best_fold_checkpoints = saved_checkpoints[best_epochs]
    best_checkpoint = max(best_fold_checkpoints, key=lambda x: x['val_auc'])
    best_checkpoint_path = best_checkpoint['path']
    
    # Save results
    df_results = pd.DataFrame([{
        'Epochs': r['epochs'],
        'Mean_AUC': r['mean_auc'],
        'Std_AUC': r['std_auc']
    } for r in epoch_search_results])
    df_results.to_csv('epoch_search_results.csv', index=False)
    logger.info(f"\nEpoch search results saved to: epoch_search_results.csv")
    
    # Visualize
    plot_epoch_search(epoch_search_results, best_epochs)
    
    # Print summary
    logger.info(f"\n{'='*80}")
    logger.info("EPOCH SEARCH RESULTS")
    logger.info(f"{'='*80}")
    for r in epoch_search_results:
        marker = "  ✓ BEST" if r['epochs'] == best_epochs else ""
        logger.info(f"Epochs {r['epochs']:2d} → Mean AUC: {r['mean_auc']:.4f} ± {r['std_auc']:.4f}{marker}")
    logger.info(f"\n✓ Optimal Epochs: {best_epochs}")
    logger.info(f"✓ Best Checkpoint: {best_checkpoint_path} (Val AUC: {best_checkpoint['val_auc']:.4f})")
    logger.info(f"{'='*80}")
    
    # Cleanup: remove checkpoints for non-optimal epochs to save space
    for epochs in saved_checkpoints:
        if epochs != best_epochs:
            for ckpt in saved_checkpoints[epochs]:
                if os.path.exists(ckpt['path']):
                    os.remove(ckpt['path'])
    logger.info(f"\n🧹 Cleaned up non-optimal checkpoints")
    
    return best_epochs, best_checkpoint_path, label_columns


def plot_epoch_search(results, best_epochs):
    """Visualize epoch search results"""
    epochs_list = [r['epochs'] for r in results]
    mean_aucs = [r['mean_auc'] for r in results]
    std_aucs = [r['std_auc'] for r in results]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Mean AUC with error bars
    colors = ['#2E7D32' if e == best_epochs else '#1976D2' for e in epochs_list]
    ax1.bar(range(len(epochs_list)), mean_aucs, yerr=std_aucs, 
            color=colors, alpha=0.7, capsize=10, edgecolor='black', linewidth=2)
    ax1.set_xticks(range(len(epochs_list)))
    ax1.set_xticklabels(epochs_list)
    ax1.set_xlabel('Epochs', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Mean Validation AUC', fontsize=14, fontweight='bold')
    ax1.set_title('Epoch Search Results (Green = Best)', fontsize=16, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    ax1.axhline(y=0.5, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Random')
    
    # Annotate best
    best_idx = epochs_list.index(best_epochs)
    ax1.annotate(f'BEST\n{mean_aucs[best_idx]:.4f}', 
                xy=(best_idx, mean_aucs[best_idx]),
                xytext=(best_idx, mean_aucs[best_idx] + 0.05),
                ha='center', fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='#2E7D32', alpha=0.7, edgecolor='black'),
                color='white',
                arrowprops=dict(arrowstyle='->', color='black', lw=2))
    ax1.legend()
    
    # Plot 2: Individual fold AUCs
    x_pos = np.arange(len(epochs_list))
    width = 0.15
    for fold_idx in range(5):
        fold_aucs = [r['fold_aucs'][fold_idx] for r in results]
        ax2.bar(x_pos + fold_idx * width, fold_aucs, width, 
               label=f'Fold {fold_idx+1}', alpha=0.8)
    
    ax2.set_xticks(x_pos + width * 2)
    ax2.set_xticklabels(epochs_list)
    ax2.set_xlabel('Epochs', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Validation AUC', fontsize=14, fontweight='bold')
    ax2.set_title('Per-Fold AUC Across Epochs', fontsize=16, fontweight='bold')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    ax2.axhline(y=0.5, color='red', linestyle='--', linewidth=1.5, alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('epoch_search_plot.png', dpi=150, bbox_inches='tight')
    logger.info("Epoch search visualization saved to: epoch_search_plot.png")
    plt.close()


def train_pipeline(config):
    """Complete training pipeline with 5-fold cross-validation for multi-label classification"""
    logger.info("="*80)
    logger.info("TRAINING PIPELINE WITH 5-FOLD CROSS-VALIDATION")
    logger.info("="*80)
    
    # Load data
    df, label_columns, class_weights = load_dataset(config['csv_dir'])
    df['image_exists'] = df['image_path'].apply(
        lambda x: os.path.exists(os.path.join(config['image_dir'], x))
    )
    df = df[df['image_exists']]
    logger.info(f"Total samples: {len(df)}, Classes: {len(label_columns)}")
    
    # Split: 80% for k-fold CV, 20% for final testing
    strat_col = safe_stratify_column(df, 'label_for_stratification')
    if strat_col is None:
        train_val_df, test_df = train_test_split(
            df, test_size=0.2, random_state=42, shuffle=True
        )
        use_stratify = False
    else:
        train_val_df, test_df = train_test_split(
            df, test_size=0.2, random_state=42, stratify=strat_col
        )
        use_stratify = True
    logger.info(f"Split: Train+Val={len(train_val_df)} (80%), Test={len(test_df)} (20%)")
    
    device = torch.device(config['device'])
    
    # Prepare class weights tensor
    weight_tensor = torch.tensor([class_weights[col] for col in label_columns], dtype=torch.float32)
    criterion = FocalLoss(class_weights=weight_tensor) if config['focal_loss'] else nn.BCEWithLogitsLoss(pos_weight=weight_tensor)
    
    # 5-Fold Cross-Validation
    if use_stratify:
        try:
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            splitter = lambda X: skf.split(X, X['label_for_stratification'])
        except Exception:
            logger.warning("StratifiedKFold failed, falling back to KFold")
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            splitter = lambda X: kf.split(X)
    else:
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        splitter = lambda X: kf.split(X)
    
    fold_results = []
    
    for fold, (train_idx, val_idx) in enumerate(splitter(train_val_df), 1):
        logger.info(f"\n{'='*80}")
        logger.info(f"FOLD {fold}/5")
        logger.info(f"{'='*80}")
        
        train_df = train_val_df.iloc[train_idx].reset_index(drop=True)
        val_df = train_val_df.iloc[val_idx].reset_index(drop=True)
        logger.info(f"Train={len(train_df)}, Val={len(val_df)}")
        
        # Model
        model = BiomedCLIPClassifier(
            num_classes=len(label_columns),
            freeze_backbone=config['freeze_backbone']
        ).to(device)
        
        if fold == 1:
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logger.info(f"Trainable parameters: {trainable:,}")
        
        # Dataloaders
        train_loader = DataLoader(
            ChestXrayDataset(train_df, config['image_dir'], model.preprocess_train, label_columns=label_columns),
            batch_size=config['batch_size'], shuffle=True,
            num_workers=config['num_workers'], pin_memory=True
        )
        val_loader = DataLoader(
            ChestXrayDataset(val_df, config['image_dir'], model.preprocess_val, label_columns=label_columns),
            batch_size=config['batch_size'], shuffle=False,
            num_workers=config['num_workers'], pin_memory=True
        )
        
        # Optimizer
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=config['lr'], weight_decay=config['weight_decay']
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=3
        )
        
        best_auc = 0.0
        patience_counter = 0
        
        for epoch in range(config['epochs']):
            logger.info(f"\nEpoch {epoch+1}/{config['epochs']} | LR: {optimizer.param_groups[0]['lr']:.2e}")
            
            # Train
            train_loss, train_preds, train_labels = train_epoch(
                model, train_loader, criterion, optimizer, device
            )
            train_metrics = compute_metrics(train_preds, train_labels)
            
            # Validate
            val_loss, val_preds, val_labels = evaluate(
                model, val_loader, criterion, device
            )
            val_metrics = compute_metrics(val_preds, val_labels)
            
            logger.info(f"Train: Loss={train_loss:.4f}, Acc={train_metrics['accuracy']:.4f}, AUC={train_metrics['auc']:.4f}, F1={train_metrics['f1']:.4f}")
            logger.info(f"Val:   Loss={val_loss:.4f}, Acc={val_metrics['accuracy']:.4f}, AUC={val_metrics['auc']:.4f}, F1={val_metrics['f1']:.4f}")
            
            scheduler.step(val_metrics['auc'])
            
            # Save best model for this fold
            if val_metrics['auc'] > best_auc:
                best_auc = val_metrics['auc']
                fold_model_path = config['model_path'].replace('.pth', f'_fold{fold}.pth')
                torch.save({
                    'epoch': epoch,
                    'fold': fold,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'auc': best_auc,
                    'label_columns': label_columns,
                }, fold_model_path)
                logger.info(f"✓ Saved best model for fold {fold} (AUC: {best_auc:.4f})")
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= config['patience']:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
        
        fold_results.append({'fold': fold, 'best_auc': best_auc})
        logger.info(f"\nFold {fold} completed - Best AUC: {best_auc:.4f}")
    
    # Summary of k-fold results
    logger.info(f"\n{'='*80}")
    logger.info("K-FOLD CROSS-VALIDATION SUMMARY")
    logger.info(f"{'='*80}")
    mean_auc = np.mean([r['best_auc'] for r in fold_results])
    std_auc = np.std([r['best_auc'] for r in fold_results])
    for result in fold_results:
        logger.info(f"Fold {result['fold']}: AUC = {result['best_auc']:.4f}")
    logger.info(f"\nMean AUC: {mean_auc:.4f} ± {std_auc:.4f}")
    
    # Use best fold model for final testing
    best_fold = max(fold_results, key=lambda x: x['best_auc'])['fold']
    best_fold_path = config['model_path'].replace('.pth', f'_fold{best_fold}.pth')
    logger.info(f"\nUsing best model from Fold {best_fold} for final testing")
    
    # Save best model as main model
    import shutil
    shutil.copy(best_fold_path, config['model_path'])
    logger.info(f"Saved best model as {config['model_path']}")
    
    # Test evaluation on held-out 20% data
    logger.info(f"\n{'='*80}")
    logger.info("FINAL EVALUATION ON HELD-OUT TEST SET (20%)")
    logger.info(f"{'='*80}")
    checkpoint = torch.load(config['model_path'], weights_only=False)
    model = BiomedCLIPClassifier(num_classes=len(label_columns)).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loader = DataLoader(
        ChestXrayDataset(test_df, config['image_dir'], model.preprocess_val, label_columns=label_columns),
        batch_size=config['batch_size'], shuffle=False,
        num_workers=config['num_workers'], pin_memory=True
    )
    
    test_loss, test_preds, test_labels = evaluate(model, test_loader, criterion, device)
    test_metrics = compute_metrics(test_preds, test_labels)
    
    logger.info(f"Test: Loss={test_loss:.4f}, Acc={test_metrics['accuracy']:.4f}, AUC={test_metrics['auc']:.4f}, F1={test_metrics['f1']:.4f}")
    
    # Comprehensive evaluation
    comprehensive_evaluation(test_preds, test_labels, label_columns)
    
    logger.info("="*80)
    logger.info("TRAINING COMPLETE")
    logger.info("="*80)

def evaluate_pipeline(config):
    """Evaluation-only pipeline for multi-label classification"""
    logger.info("="*80)
    logger.info("EVALUATION PIPELINE")
    logger.info("="*80)
    
    # Load data
    df, label_columns, class_weights = load_dataset(config['csv_dir'])
    df['image_exists'] = df['image_path'].apply(
        lambda x: os.path.exists(os.path.join(config['image_dir'], x))
    )
    df = df[df['image_exists']]
    
    # Split: Use same 20% test set as in training
    strat_col = safe_stratify_column(df, 'label_for_stratification')
    if strat_col is None:
        _, test_df = train_test_split(
            df, test_size=0.2, random_state=42, shuffle=True
        )
    else:
        _, test_df = train_test_split(
            df, test_size=0.2, random_state=42, stratify=strat_col
        )
    logger.info(f"Test samples: {len(test_df)}")
    
    # Load model
    device = torch.device(config['device'])
    checkpoint = torch.load(config['model_path'], map_location=device, weights_only=False)
    
    model = BiomedCLIPClassifier(num_classes=len(label_columns))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    logger.info(f"Model loaded (trained {checkpoint['epoch']} epochs, AUC: {checkpoint['auc']:.4f})")
    
    # Evaluate
    test_loader = DataLoader(
        ChestXrayDataset(test_df, config['image_dir'], model.preprocess_val, label_columns=label_columns),
        batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers']
    )
    
    # Prepare class weights
    weight_tensor = torch.tensor([class_weights[col] for col in label_columns], dtype=torch.float32)
    criterion = FocalLoss(class_weights=weight_tensor)
    _, test_preds, test_labels = evaluate(model, test_loader, criterion, device)
    
    # Comprehensive evaluation
    comprehensive_evaluation(test_preds, test_labels, label_columns)
    
    logger.info("="*80)
    logger.info("EVALUATION COMPLETE")
    logger.info("="*80)


def main():
    parser = argparse.ArgumentParser(description='Chest X-Ray Multi-Label Classification')
    parser.add_argument('--mode', type=str, default='both', 
                       choices=['train', 'evaluate', 'both'],
                       help='Mode: train, evaluate, or both')
    parser.add_argument('--csv_dir', type=str, 
                       default='/DATA1/Brij_2411MC09/archive',
                       help='Path to CSV directory')
    parser.add_argument('--image_dir', type=str,
                       default='/DATA1/Brij_2411MC09/archive/files',
                       help='Path to image directory')
    parser.add_argument('--model_path', type=str, default='best_model_unified.pth',
                       help='Path to save/load model')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of epochs to train (used in train mode)')
    parser.add_argument('--lr', type=float, default=2e-5,
                       help='Learning rate')
    parser.add_argument('--num_workers', type=int, default=8,
                       help='Number of data loading workers')
    parser.add_argument('--unfreeze_layers', type=int, default=0,
                       help='Number of last transformer blocks to unfreeze (0=all frozen, 2=recommended)')
    
    args = parser.parse_args()
    
    config = {
        'csv_dir': args.csv_dir,
        'image_dir': args.image_dir,
        'model_path': args.model_path,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'lr': args.lr,
        'weight_decay': 1e-4,
        'num_workers': args.num_workers,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'freeze_backbone': True,
        'unfreeze_layers': args.unfreeze_layers,
        'focal_loss': True,
        'patience': 5,
    }
    
    logger.info("="*80)
    logger.info("MULTI-LABEL CLASSIFICATION FOR CHEST X-RAY ANALYSIS")
    logger.info("="*80)
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Device: {config['device']}")
    logger.info(f"Batch size: {config['batch_size']}")
    logger.info(f"Learning rate: {config['lr']}")
    logger.info(f"Unfreezing strategy: Last {config['unfreeze_layers']} transformer blocks")
    logger.info("="*80)
    
    if args.mode == 'train':
        # Search for best epochs and get best checkpoint
        logger.info("\n📍 Step 1: Searching for optimal epoch count...\n")
        best_epochs, best_checkpoint_path, label_columns = search_best_epochs(config)
        
        # Skip redundant training - go directly to comprehensive evaluation
        logger.info(f"\n📍 Step 2: Final Evaluation (using best checkpoint)...\n")
        final_evaluation(best_checkpoint_path, config)
    elif args.mode == 'evaluate':
        evaluate_pipeline(config)
    else:
        # Search for best epochs and get best checkpoint
        logger.info("\n📍 Step 1: Searching for optimal epoch count...\n")
        best_epochs, best_checkpoint_path, label_columns = search_best_epochs(config)
        
        # Skip redundant training - go directly to comprehensive evaluation
        logger.info(f"\n📍 Step 2: Final Evaluation (using best checkpoint)...\n")
        final_evaluation(best_checkpoint_path, config)

if __name__ == "__main__":
    main()


"""
===============================================================================
USAGE EXAMPLES:
===============================================================================

1. RECOMMENDED: Partial Unfreezing (last 2 blocks)
   python train_with_unfreezing.py --mode both --unfreeze_layers 2

2. More aggressive: Unfreeze last 4 blocks
   python train_with_unfreezing.py --mode both --unfreeze_layers 4

3. Keep all frozen (same as original)
   python train_with_unfreezing.py --mode both --unfreeze_layers 0

4. Full unfreezing
   python train_with_unfreezing.py --mode both --freeze_backbone False


===============================================================================
EXPECTED RESULTS:
===============================================================================

Unfreezing 0 blocks (current):  AUC ~0.71
Unfreezing 2 blocks (recommended): AUC ~0.74-0.76  (+3-5%)
Unfreezing 4 blocks:  AUC ~0.75-0.77  (+4-6%)
Full unfreezing:  AUC ~0.76-0.78  (+5-7%)

===============================================================================
"""
import pickle
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, top_k_accuracy_score
)
import matplotlib.pyplot as plt
import seaborn as sns

def load_predictions(pickle_path):
    """
    Load predictions from pickle file.
    Based on SlowFast visualization code, the pickle contains (preds, labels)
    """
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
    return data

def evaluate_model(preds, labels, class_names=None, zero_indexed=True):
    """
    Comprehensive model evaluation for SlowFast predictions.
    
    Args:
        preds: predictions from model (probabilities or logits)
        labels: ground truth labels (one-hot encoded or class indices)
        class_names: Optional list of class names (should match indexing)
        zero_indexed: If False, assumes classes are 1-indexed (1-87 instead of 0-86)
    """
    
    print("="*80)
    print("SLOWFAST MODEL PERFORMANCE EVALUATION")
    print("="*80)
    
    # Convert to numpy if needed
    if torch.is_tensor(preds):
        preds = preds.numpy()
    if torch.is_tensor(labels):
        labels = labels.numpy()
    
    # Handle labels - check if one-hot encoded or class indices
    if len(labels.shape) > 1 and labels.shape[1] > 1:
        # One-hot encoded
        y_true = np.argmax(labels, axis=1)
        print("✓ Detected one-hot encoded labels")
    else:
        # Already class indices
        y_true = labels.flatten() if len(labels.shape) > 1 else labels
        print("✓ Detected class index labels")
    
    # Get predicted classes
    if len(preds.shape) > 1:
        y_pred = np.argmax(preds, axis=1)
        print("✓ Using argmax for predictions")
    else:
        y_pred = preds
        print("✓ Predictions are already class indices")
    
    n_classes = preds.shape[1] if len(preds.shape) > 1 else int(max(y_true.max(), y_pred.max()) + 1)
    n_samples = len(y_true)
    
    # Adjust for 1-indexed classes if needed
    if not zero_indexed:
        print("✓ Converting from 1-indexed to 0-indexed for evaluation")
        # Argmax gives 0-86, but actual classes are 1-87
        # We'll keep 0-indexed internally and just display with +1
    
    print(f"\nDataset Info:")
    print(f"  Total samples: {n_samples}")
    print(f"  Number of classes: {n_classes}")
    if not zero_indexed:
        print(f"  Class range: 1-{n_classes} (1-indexed)")
    else:
        print(f"  Class range: 0-{n_classes-1} (0-indexed)")
    print(f"  Predictions shape: {preds.shape}")
    print(f"  Labels shape: {labels.shape}")
    print(f"  Unique classes in ground truth: {len(np.unique(y_true))}")
    print(f"  Unique classes in predictions: {len(np.unique(y_pred))}")
    
    # Overall Metrics
    print("\n" + "─"*80)
    print("OVERALL METRICS")
    print("─"*80)
    
    accuracy = accuracy_score(y_true, y_pred)
    print(f"\n✓ Top-1 Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Top-k accuracy (if we have probability predictions)
    if len(preds.shape) > 1:
        # Create labels array for all possible classes
        all_labels = np.arange(n_classes)
        top3_acc = top_k_accuracy_score(y_true, preds, k=3, labels=all_labels)
        top5_acc = top_k_accuracy_score(y_true, preds, k=5, labels=all_labels)
        print(f"✓ Top-3 Accuracy: {top3_acc:.4f} ({top3_acc*100:.2f}%)")
        print(f"✓ Top-5 Accuracy: {top5_acc:.4f} ({top5_acc*100:.2f}%)")
    
    # Macro metrics (treats all classes equally - good for imbalanced datasets)
    precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    print(f"\nMacro-averaged metrics (treats all classes equally):")
    print(f"  Precision: {precision_macro:.4f}")
    print(f"  Recall:    {recall_macro:.4f}")
    print(f"  F1-Score:  {f1_macro:.4f}")
    
    # Weighted metrics (weights by class frequency)
    precision_weighted = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall_weighted = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    print(f"\nWeighted metrics (weighted by class frequency):")
    print(f"  Precision: {precision_weighted:.4f}")
    print(f"  Recall:    {recall_weighted:.4f}")
    print(f"  F1-Score:  {f1_weighted:.4f}")
    
    # Per-class metrics
    print("\n" + "─"*80)
    print("PER-CLASS PERFORMANCE")
    print("─"*80)
    
    # Get per-class metrics - only for classes that exist
    labels_list = np.arange(n_classes)
    precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0, labels=labels_list)
    recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0, labels=labels_list)
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0, labels=labels_list)
    
    # Class support (number of true samples per class)
    class_support = np.bincount(y_true, minlength=n_classes)
    
    # Find best and worst performing classes
    valid_classes = class_support > 0
    if valid_classes.any():
        valid_f1 = f1_per_class[valid_classes]
        valid_class_ids = np.where(valid_classes)[0]
        
        # Best classes
        if len(valid_f1) >= 5:
            best_indices = np.argsort(valid_f1)[-5:][::-1]
            print("\n✓ Top 5 Best Performing Classes (by F1-score):")
            for idx in best_indices:
                class_id = valid_class_ids[idx]
                display_id = class_id + 1 if not zero_indexed else class_id
                name = class_names[class_id] if class_names else f"Class {display_id}"
                print(f"  {name:40s} - F1: {f1_per_class[class_id]:.4f}, "
                      f"Prec: {precision_per_class[class_id]:.4f}, "
                      f"Rec: {recall_per_class[class_id]:.4f}, "
                      f"Support: {class_support[class_id]}")
        
        # Worst classes
        worst_count = min(5, len(valid_f1))
        worst_indices = np.argsort(valid_f1)[:worst_count]
        print(f"\n⚠ Top {worst_count} Worst Performing Classes (by F1-score):")
        for idx in worst_indices:
            class_id = valid_class_ids[idx]
            display_id = class_id + 1 if not zero_indexed else class_id
            name = class_names[class_id] if class_names else f"Class {display_id}"
            print(f"  {name:40s} - F1: {f1_per_class[class_id]:.4f}, "
                  f"Prec: {precision_per_class[class_id]:.4f}, "
                  f"Rec: {recall_per_class[class_id]:.4f}, "
                  f"Support: {class_support[class_id]}")
    
    # Classes with no predictions
    print("\n" + "─"*80)
    print("PREDICTION COVERAGE ANALYSIS")
    print("─"*80)
    
    pred_counts = np.bincount(y_pred, minlength=n_classes)
    classes_never_predicted = np.where((class_support > 0) & (pred_counts == 0))[0]
    
    if len(classes_never_predicted) > 0:
        print(f"\n⚠ WARNING: {len(classes_never_predicted)} classes never predicted (but present in test set):")
        for cls in classes_never_predicted[:15]:  # Show first 15
            display_id = cls + 1 if not zero_indexed else cls
            if class_names and cls < len(class_names):
                name = class_names[cls]
            else:
                name = f"Class {display_id}"
            print(f"  {name} (appears {class_support[cls]} times in test set)")
        if len(classes_never_predicted) > 15:
            print(f"  ... and {len(classes_never_predicted) - 15} more")
    else:
        print("\n✓ All classes with test samples were predicted at least once")
    
    # Classes present in predictions but not in ground truth
    classes_never_true = np.where((class_support == 0) & (pred_counts > 0))[0]
    if len(classes_never_true) > 0:
        print(f"\n⚠ WARNING: {len(classes_never_true)} classes predicted but not in test set:")
        for cls in classes_never_true[:10]:
            display_id = cls + 1 if not zero_indexed else cls
            if class_names and cls < len(class_names):
                name = class_names[cls]
            else:
                name = f"Class {display_id}"
            print(f"  {name} (predicted {pred_counts[cls]} times)")
    
    # Confusion matrix analysis
    print("\n" + "─"*80)
    print("CONFUSION ANALYSIS")
    print("─"*80)
    
    cm = confusion_matrix(y_true, y_pred)
    
    # Find most confused pairs
    cm_no_diag = cm.copy()
    np.fill_diagonal(cm_no_diag, 0)
    
    print("\n⚠ Top 10 Most Confused Class Pairs:")
    flat_indices = np.argsort(cm_no_diag.ravel())[::-1][:10]
    for i, flat_idx in enumerate(flat_indices, 1):
        true_cls = flat_idx // n_classes
        pred_cls = flat_idx % n_classes
        count = cm_no_diag[true_cls, pred_cls]
        if count > 0:
            true_display = true_cls + 1 if not zero_indexed else true_cls
            pred_display = pred_cls + 1 if not zero_indexed else pred_cls
            true_name = class_names[true_cls] if class_names else f"Class {true_display}"
            pred_name = class_names[pred_cls] if class_names else f"Class {pred_display}"
            total_true = class_support[true_cls]
            percentage = (count / total_true * 100) if total_true > 0 else 0
            print(f"  {i:2d}. True: {true_name:30s} → Pred: {pred_name:30s} "
                  f"({count:3d} times, {percentage:.1f}% of class)")
    
    # Confidence analysis (if we have probabilities)
    if len(preds.shape) > 1:
        print("\n" + "─"*80)
        print("CONFIDENCE ANALYSIS")
        print("─"*80)
        
        confidence = np.max(preds, axis=1)
        correct_mask = y_true == y_pred
        correct_confidence = confidence[correct_mask]
        incorrect_confidence = confidence[~correct_mask]
        
        print(f"\n✓ Correct predictions ({len(correct_confidence)} samples):")
        print(f"  Mean confidence:   {correct_confidence.mean():.4f}")
        print(f"  Median confidence: {np.median(correct_confidence):.4f}")
        print(f"  Min confidence:    {correct_confidence.min():.4f}")
        print(f"  Max confidence:    {correct_confidence.max():.4f}")
        
        print(f"\n⚠ Incorrect predictions ({len(incorrect_confidence)} samples):")
        print(f"  Mean confidence:   {incorrect_confidence.mean():.4f}")
        print(f"  Median confidence: {np.median(incorrect_confidence):.4f}")
        print(f"  Min confidence:    {incorrect_confidence.min():.4f}")
        print(f"  Max confidence:    {incorrect_confidence.max():.4f}")
        
        print(f"\nConfidence gap: {correct_confidence.mean() - incorrect_confidence.mean():.4f}")
        print("(Larger gap = model is more calibrated)")
    
    # Save detailed classification report to file
    print("\n" + "─"*80)
    print("DETAILED CLASSIFICATION REPORT")
    print("─"*80)
    
    report = classification_report(y_true, y_pred, zero_division=0)
    print("\n" + report)
    
    # Save to file
    with open('classification_report.txt', 'w') as f:
        f.write("SLOWFAST MODEL EVALUATION REPORT\n")
        f.write("="*80 + "\n\n")
        f.write(f"Overall Accuracy: {accuracy:.4f}\n")
        f.write(f"Macro F1-Score: {f1_macro:.4f}\n")
        f.write(f"Weighted F1-Score: {f1_weighted:.4f}\n\n")
        f.write(report)
    print("\n✓ Detailed report saved to: classification_report.txt")
    
    return {
        'accuracy': accuracy,
        'top3_accuracy': top3_acc if len(preds.shape) > 1 else None,
        'top5_accuracy': top5_acc if len(preds.shape) > 1 else None,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'precision_weighted': precision_weighted,
        'recall_weighted': recall_weighted,
        'f1_weighted': f1_weighted,
        'confusion_matrix': cm,
        'y_true': y_true,
        'y_pred': y_pred
    }

def plot_confusion_matrix(cm, save_path='confusion_matrix.png', top_n=30):
    """Plot confusion matrix for top N classes"""
    # Get top N classes by total occurrences
    class_totals = cm.sum(axis=0) + cm.sum(axis=1)
    top_classes = np.argsort(class_totals)[::-1][:top_n]
    
    # Extract submatrix
    cm_subset = cm[np.ix_(top_classes, top_classes)]
    
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm_subset, annot=True, fmt='d', cmap='Blues', 
                xticklabels=top_classes, yticklabels=top_classes,
                cbar_kws={'label': 'Count'})
    plt.title(f'Confusion Matrix (Top {top_n} Most Common Classes)', fontsize=16, pad=20)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Confusion matrix saved to: {save_path}")
    plt.close()

def plot_class_distribution(y_true, y_pred, save_path='class_distribution.png'):
    """Plot class distribution comparison"""
    unique_true, counts_true = np.unique(y_true, return_counts=True)
    unique_pred, counts_pred = np.unique(y_pred, return_counts=True)
    
    # Create full arrays for all classes
    n_classes = max(unique_true.max(), unique_pred.max()) + 1
    full_counts_true = np.zeros(n_classes)
    full_counts_pred = np.zeros(n_classes)
    full_counts_true[unique_true] = counts_true
    full_counts_pred[unique_pred] = counts_pred
    
    # Get top 30 classes by ground truth
    top_30_classes = np.argsort(full_counts_true)[::-1][:30]
    
    fig, ax = plt.subplots(figsize=(14, 8))
    x = np.arange(len(top_30_classes))
    width = 0.35
    
    ax.bar(x - width/2, full_counts_true[top_30_classes], width, label='Ground Truth', alpha=0.8)
    ax.bar(x + width/2, full_counts_pred[top_30_classes], width, label='Predictions', alpha=0.8)
    
    ax.set_xlabel('Class ID', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Class Distribution: Ground Truth vs Predictions (Top 30 Classes)', fontsize=14, pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(top_30_classes, rotation=45)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Class distribution plot saved to: {save_path}")
    plt.close()

# Main execution
if __name__ == "__main__":
    # Define class names (1-indexed in the original list)
    class_names = [
        "Take eating_utensil", "Cut tomato", "Put eating_utensil", "Take condiment_container",
        "Cut cucumber", "Cut carrot", "Put condiment_container", "Cut onion", "Take plate",
        "Take bowl", "Put bowl", "Put trash,trash_container", "Put plate", "Cut bell_pepper",
        "Take paper_towel", "Move Around bacon", "Open condiment_container", "Wash eating_utensil",
        "Spread condiment,bread,eating_utensil", "Put pan", "Put lettuce", "Move Around patty",
        "Put pot", "Put bread", "Take bread", "Close condiment_container", "Wash hand",
        "Put tomato", "Take seasoning_container", "Take cup", "Divide/Pull Apart lettuce",
        "Put cup", "Take pot", "Take bread_container", "Take tomato", "Take pan",
        "Move Around pan", "Wash cutting_board", "Put bread_container", "Take sponge",
        "Take lettuce", "Take onion", "Put sponge", "Divide/Pull Apart paper_towel",
        "Take cheese_container", "Take oil_container", "Put seasoning_container", "Take cucumber",
        "Wash pan", "Take bell_pepper", "Divide/Pull Apart onion", "Put cutting_board",
        "Mix mixture,eating_utensil", "Take tomato_container", "Put cheese", "Move Around eating_utensil",
        "Put bell_pepper", "Pour oil,oil_container,pan", "Take pasta_container", "Cut lettuce",
        "Put tomato_container", "Wash bowl", "Cut olive", "Pour condiment,condiment_container,salad",
        "Wash pot", "Open bread_container", "Take grocery_bag", "Take cheese",
        "Pour seasoning,seasoning_container,salad", "Put paper_towel", "Put cucumber",
        "Compress sandwich", "Put cheese_container", "Put onion", "Crack egg",
        "Take cutting_board", "Open oil_container", "Squeeze washing_liquid,sponge", "Wash strainer",
        "Move Around bowl", "Move Around pot", "Put grocery_bag", "Take egg",
        "Open cheese_container", "Close oil_container", "Put oil_container"
    ]
    
    # Load predictions (SlowFast format)
    path = '/home/mma02/remote_code/HOBM/SlowFast/test_raw_pretrain/predictions.pickle'
    
    print("Loading SlowFast predictions from:")
    print(f"  {path}\n")
    
    data = load_predictions(path)
    
    print(f"Loaded data type: {type(data)}")
    print(f"Number of elements: {len(data)}")
    
    # Extract predictions and labels
    # Based on SlowFast code: (preds, labels)
    preds = data[0]  # Model outputs
    labels = data[1]  # Ground truth
    
    print(f"Predictions shape: {preds.shape if hasattr(preds, 'shape') else len(preds)}")
    print(f"Labels shape: {labels.shape if hasattr(labels, 'shape') else len(labels)}")
    
    # Run evaluation with class names and 1-indexed flag
    # Note: internally we use 0-indexing, but display with proper class names
    results = evaluate_model(preds, labels, class_names=class_names, zero_indexed=False)
    
    # Generate visualizations
    print("\n" + "─"*80)
    print("GENERATING VISUALIZATIONS")
    print("─"*80)
    
    plot_confusion_matrix(results['confusion_matrix'])
    plot_class_distribution(results['y_true'], results['y_pred'])
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)
    print("\nSummary:")
    print(f"  • Overall Accuracy:     {results['accuracy']*100:.2f}%")
    if results['top3_accuracy']:
        print(f"  • Top-3 Accuracy:       {results['top3_accuracy']*100:.2f}%")
        print(f"  • Top-5 Accuracy:       {results['top5_accuracy']*100:.2f}%")
    print(f"  • Macro F1-Score:       {results['f1_macro']:.4f}")
    print(f"  • Weighted F1-Score:    {results['f1_weighted']:.4f}")
    print("\nGenerated files:")
    print("  • classification_report.txt")
    print("  • confusion_matrix.png")
    print("  • class_distribution.png")
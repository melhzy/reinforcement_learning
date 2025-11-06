"""
Utility functions for reinforcement learning tutorials
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Dict
import json


def set_random_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def plot_training_curve(
    rewards: List[float],
    title: str = "Training Progress",
    window: int = 100,
    save_path: str = None
):
    """
    Plot training rewards with moving average.
    
    Args:
        rewards: List of episode rewards
        title: Plot title
        window: Window size for moving average
        save_path: Path to save figure (optional)
    """
    plt.figure(figsize=(12, 5))
    
    # Raw rewards
    plt.subplot(1, 2, 1)
    plt.plot(rewards, alpha=0.3, label='Raw Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title(f'{title} - Raw Rewards')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Moving average
    plt.subplot(1, 2, 2)
    if len(rewards) >= window:
        moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
        plt.plot(moving_avg, label=f'MA ({window} episodes)')
    else:
        plt.plot(rewards, label='Rewards')
    
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.title(f'{title} - Moving Average')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_multi_metric(
    metrics: Dict[str, List[float]],
    title: str = "Training Metrics",
    save_path: str = None
):
    """
    Plot multiple training metrics.
    
    Args:
        metrics: Dictionary of metric name to values
        title: Plot title
        save_path: Path to save figure (optional)
    """
    num_metrics = len(metrics)
    fig, axes = plt.subplots(1, num_metrics, figsize=(6 * num_metrics, 5))
    
    if num_metrics == 1:
        axes = [axes]
    
    for ax, (metric_name, values) in zip(axes, metrics.items()):
        ax.plot(values)
        ax.set_xlabel('Episode')
        ax.set_ylabel(metric_name)
        ax.set_title(metric_name)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_confusion_matrix(
    y_true: List[int],
    y_pred: List[int],
    class_names: List[str],
    title: str = "Confusion Matrix",
    save_path: str = None
):
    """
    Plot confusion matrix for classification results.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Names of classes
        title: Plot title
        save_path: Path to save figure (optional)
    """
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def normalize_features(features: np.ndarray) -> np.ndarray:
    """
    Normalize features to zero mean and unit variance.
    
    Args:
        features: Feature array (n_samples, n_features)
        
    Returns:
        Normalized features
    """
    mean = np.mean(features, axis=0)
    std = np.std(features, axis=0)
    std[std == 0] = 1.0  # Avoid division by zero
    
    return (features - mean) / std


def clr_transform(compositional_data: np.ndarray) -> np.ndarray:
    """
    Centered log-ratio (CLR) transformation for compositional data.
    Commonly used for microbiome data.
    
    Args:
        compositional_data: Compositional data (must be positive)
        
    Returns:
        CLR-transformed data
    """
    # Add small constant to avoid log(0)
    data = compositional_data + 1e-6
    
    # Compute geometric mean
    geometric_mean = np.exp(np.mean(np.log(data), axis=-1, keepdims=True))
    
    # CLR transform
    clr_data = np.log(data / geometric_mean)
    
    return clr_data


def compute_classification_metrics(
    y_true: List[int],
    y_pred: List[int],
    class_names: List[str] = None
) -> Dict:
    """
    Compute comprehensive classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Names of classes (optional)
        
    Returns:
        Dictionary of metrics
    """
    from sklearn.metrics import (
        accuracy_score,
        precision_recall_fscore_support,
        confusion_matrix
    )
    
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None
    )
    
    metrics = {
        'accuracy': accuracy,
        'per_class': {}
    }
    
    num_classes = len(np.unique(y_true))
    for i in range(num_classes):
        class_name = class_names[i] if class_names else f"Class {i}"
        metrics['per_class'][class_name] = {
            'precision': precision[i],
            'recall': recall[i],
            'f1_score': f1[i],
            'support': support[i]
        }
    
    # Overall metrics
    metrics['macro_avg'] = {
        'precision': np.mean(precision),
        'recall': np.mean(recall),
        'f1_score': np.mean(f1)
    }
    
    return metrics


def save_results(results: Dict, filepath: str):
    """
    Save results to JSON file.
    
    Args:
        results: Results dictionary
        filepath: Path to save file
    """
    # Convert numpy types to native Python types
    def convert_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_types(item) for item in obj]
        return obj
    
    results_converted = convert_types(results)
    
    with open(filepath, 'w') as f:
        json.dump(results_converted, f, indent=2)
    
    print(f"Results saved to {filepath}")


def load_results(filepath: str) -> Dict:
    """
    Load results from JSON file.
    
    Args:
        filepath: Path to load file
        
    Returns:
        Results dictionary
    """
    with open(filepath, 'r') as f:
        results = json.load(f)
    
    return results


def print_metrics_table(metrics: Dict):
    """
    Print classification metrics in a formatted table.
    
    Args:
        metrics: Metrics dictionary from compute_classification_metrics
    """
    print("\n" + "=" * 70)
    print("Classification Metrics")
    print("=" * 70)
    
    print(f"\nOverall Accuracy: {metrics['accuracy']:.4f}")
    
    print("\nPer-Class Metrics:")
    print(f"{'Class':<20} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
    print("-" * 70)
    
    for class_name, class_metrics in metrics['per_class'].items():
        print(f"{class_name:<20} "
              f"{class_metrics['precision']:<12.4f} "
              f"{class_metrics['recall']:<12.4f} "
              f"{class_metrics['f1_score']:<12.4f} "
              f"{class_metrics['support']:<10.0f}")
    
    print("\nMacro Average:")
    print(f"{'Average':<20} "
          f"{metrics['macro_avg']['precision']:<12.4f} "
          f"{metrics['macro_avg']['recall']:<12.4f} "
          f"{metrics['macro_avg']['f1_score']:<12.4f}")
    
    print("=" * 70 + "\n")


def create_synthetic_alzheimer_features(
    num_samples: int,
    class_label: int
) -> np.ndarray:
    """
    Create synthetic Alzheimer's disease features.
    
    Args:
        num_samples: Number of samples to generate
        class_label: 0=CN, 1=MCI, 2=AD
        
    Returns:
        Feature array (num_samples, num_features)
    """
    features = []
    
    for _ in range(num_samples):
        # Clinical features
        age = 65 + class_label * 5 + np.random.randn() * 3
        mmse = 28 - class_label * 5 + np.random.randn() * 2
        cdr = class_label * 0.5 + np.random.randn() * 0.1
        education = 14 + np.random.randn() * 2
        
        # Biomarkers
        abeta42 = 600 - class_label * 150 + np.random.randn() * 50
        abeta40 = 6000 + np.random.randn() * 500
        tau = 300 + class_label * 100 + np.random.randn() * 50
        ptau181 = 35 + class_label * 20 + np.random.randn() * 10
        
        # Microbiome (simplified)
        p_gingivalis = 1.0 + class_label * 0.8 + np.random.randn() * 0.3
        fb_ratio = 2.0 + (class_label - 1) * 0.5 + np.random.randn() * 0.3
        
        sample_features = [
            age, mmse, cdr, education,
            abeta42, abeta40, tau, ptau181,
            p_gingivalis, fb_ratio
        ]
        
        features.append(sample_features)
    
    return np.array(features)


if __name__ == "__main__":
    # Example usage
    print("Testing utility functions...\n")
    
    # Set random seed
    set_random_seed(42)
    
    # Generate synthetic training rewards
    rewards = [np.random.randn() * 10 + i * 0.1 for i in range(1000)]
    
    # Plot training curve
    plot_training_curve(rewards, title="Example Training", window=50)
    
    # Create synthetic classification data
    features = []
    labels = []
    for label in range(3):
        features.append(create_synthetic_alzheimer_features(100, label))
        labels.extend([label] * 100)
    
    features = np.vstack(features)
    
    # Normalize features
    features_normalized = normalize_features(features)
    
    print(f"Generated {len(labels)} synthetic samples")
    print(f"Feature shape: {features.shape}")
    print(f"Normalized feature mean: {features_normalized.mean():.4f}")
    print(f"Normalized feature std: {features_normalized.std():.4f}")
    
    print("\nUtility functions test complete!")

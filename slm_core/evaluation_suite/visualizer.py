"""
Visualization utilities for MYTHRI SLM evaluation.
Generates plots for training curves, calibration, confusion matrices, and per-class metrics.
"""
import os
import json
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for server/script runs
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

def plot_training_curves(log_dir: Path):
    """Plot training and validation loss/learning curves."""
    log_path = log_dir / "exp_0_baseline_training_log.json"
    if not log_path.exists():
        return
        
    with open(log_path, "r") as f:
        history = json.load(f)
        
    epochs = [e["epoch"] for e in history]
    train_loss = [e["train_loss"] for e in history]
    val_loss = [e["val_loss"] for e in history]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].plot(epochs, train_loss, 'b-o', label='Train Loss')
    axes[0].plot(epochs, val_loss, 'r-o', label='Val Loss')
    axes[0].set_title('Loss Curves')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    if "emotion_micro_f1" in history[0]:
        emo_f1 = [e.get("emotion_micro_f1", 0) for e in history]
        axes[1].plot(epochs, emo_f1, 'g-o', label='Emotion Micro F1')
        
    if "risk_f1" in history[0]:
        rsk_f1 = [e.get("risk_f1", 0) for e in history]
        axes[1].plot(epochs, rsk_f1, 'm-o', label='Risk F1')
        
    axes[1].set_title('Validation Metrics')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Score')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plot_path = log_dir / "training_curves.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"  Saved training curves to {plot_path}")

def plot_reliability_diagrams(calibration_data: dict, log_dir: Path):
    """Plot reliability diagrams for emotion and risk heads."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Emotion
    emo_bins = calibration_data.get("emotion_reliability", {}).get("bins", [])
    if emo_bins:
        conf = [b["confidence"] for b in emo_bins]
        acc = [b["accuracy"] for b in emo_bins]
        axes[0].bar(conf, acc, width=0.05, alpha=0.7, label='Accuracy')
        axes[0].plot([0, 1], [0, 1], 'r--', label='Perfect Calibration')
        axes[0].set_title(f'Emotion Reliability (ECE: {calibration_data.get("emotion_ece", 0):.4f})')
        axes[0].set_xlabel('Confidence')
        axes[0].set_ylabel('Accuracy')
        axes[0].set_xlim(0, 1)
        axes[0].set_ylim(0, 1)
        axes[0].legend()
    
    # Risk
    rsk_bins = calibration_data.get("risk_reliability", {}).get("bins", [])
    if rsk_bins:
        conf = [b["confidence"] for b in rsk_bins]
        acc = [b["accuracy"] for b in rsk_bins]
        axes[1].bar(conf, acc, width=0.05, alpha=0.7, label='Accuracy')
        axes[1].plot([0, 1], [0, 1], 'r--', label='Perfect Calibration')
        axes[1].set_title(f'Risk Reliability (ECE: {calibration_data.get("risk_ece", 0):.4f})')
        axes[1].set_xlabel('Confidence')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_xlim(0, 1)
        axes[1].set_ylim(0, 1)
        axes[1].legend()
        
    plt.tight_layout()
    plot_path = log_dir / "reliability_diagrams.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"  Saved reliability diagrams to {plot_path}")

def plot_risk_confusion_matrix(cm: list, log_dir: Path):
    """Plot heatmap for risk confusion matrix."""
    cm_array = np.array(cm)
    labels = ['Normal', 'Distressed', 'Risk']
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_array, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title('Risk Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    
    plt.tight_layout()
    plot_path = log_dir / "risk_confusion_matrix.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"  Saved risk confusion matrix to {plot_path}")

def plot_per_class_emotion(per_class: dict, emotion_labels: list, log_dir: Path):
    """Plot bar chart of F1 scores for each emotion."""
    classes = list(per_class.keys())
    f1_scores = [per_class[c]["f1"] for c in classes]
    
    # Sort by F1 score descending
    sorted_indices = np.argsort(f1_scores)[::-1]
    classes = [classes[i] for i in sorted_indices]
    f1_scores = [f1_scores[i] for i in sorted_indices]
    
    # Replace class_X with actual labels if available
    display_labels = []
    for c in classes:
        if c.startswith("class_") and len(c.split("_")) > 1 and c.split("_")[1].isdigit():
            idx = int(c.split("_")[1])
            display_labels.append(emotion_labels[idx] if idx < len(emotion_labels) else c)
        else:
            display_labels.append(c)
    
    plt.figure(figsize=(15, 6))
    bars = plt.bar(range(len(f1_scores)), f1_scores, color='teal')
    plt.xticks(range(len(f1_scores)), display_labels, rotation=90)
    plt.title('Per-Class Emotion F1 Scores')
    plt.xlabel('Emotion')
    plt.ylabel('F1 Score')
    plt.axhline(y=0.5, color='r', linestyle='--', label='F1 = 0.5')
    plt.legend()
    
    plt.tight_layout()
    plot_path = log_dir / "per_class_emotion_f1.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"  Saved per-class emotion F1 chart to {plot_path}")
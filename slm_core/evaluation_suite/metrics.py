"""
Evaluation metrics for the cognitive SLM.
Uses sklearn for robust multi-label and multi-class metrics.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from sklearn.metrics import (
    precision_recall_fscore_support, 
    confusion_matrix,
    accuracy_score
)

from .calibration import (
    expected_calibration_error,
    binary_expected_calibration_error,
    multilabel_expected_calibration_error,
    compute_brier_score,
)

def compute_emotion_metrics(
    predictions: torch.Tensor,   # (N, num_emotions) — sigmoid probabilities
    labels: torch.Tensor,         # (N, num_emotions) — multi-hot (0/1)
    threshold: float = 0.5,
    emotion_labels: Optional[List[str]] = None,
) -> Dict:
    """Compute multi-label emotion classification metrics with calibration across all emotion classes."""
    if isinstance(predictions, torch.Tensor):
        preds_prob = predictions.detach().cpu().numpy()
        preds = (predictions > threshold).float().detach().cpu().numpy()
    else:
        preds_prob = np.array(predictions)
        preds = (preds_prob > threshold).astype(float)
        
    if isinstance(labels, torch.Tensor):
        true = labels.detach().cpu().numpy()
    else:
        true = np.array(labels)
    
    # sklearn expects 2D arrays for multi-label
    p_micro, r_micro, f1_micro, _ = precision_recall_fscore_support(true, preds, average='micro', zero_division=0)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(true, preds, average='macro', zero_division=0)
    p_weight, r_weight, f1_weight, _ = precision_recall_fscore_support(true, preds, average='weighted', zero_division=0)
    
    # Per-class breakdown
    p_class, r_class, f1_class, support = precision_recall_fscore_support(true, preds, average=None, zero_division=0)
    
    num_classes = preds.shape[1] if len(preds.shape) > 1 else 0
    per_class = {}
    per_emotion = {}
    class_eces = []
    class_supports = []
    for i in range(num_classes):
        name = emotion_labels[i] if (emotion_labels and i < len(emotion_labels)) else f"class_{i}"
        ece_i = binary_expected_calibration_error(preds_prob[:, i], true[:, i]) if len(true) > 0 else 0.0
        brier_i = float(np.mean((preds_prob[:, i] - true[:, i]) ** 2)) if len(true) > 0 else 0.0
        class_eces.append(ece_i)
        class_supports.append(float(support[i]))
        item_stats = {
            "name": name,
            "precision": float(p_class[i]),
            "recall": float(r_class[i]),
            "f1": float(f1_class[i]),
            "ece": float(ece_i),
            "brier": float(brier_i),
            "support": int(support[i]),
        }
        per_class[f"class_{i}"] = item_stats
        per_emotion[name] = item_stats
    
    # Calibration overall evaluating all emotion classes
    macro_ece = float(np.mean(class_eces)) if class_eces else 0.0
    micro_ece = multilabel_expected_calibration_error(preds_prob, true, num_bins=15, mode="micro") if len(true) > 0 else 0.0
    total_supp = sum(class_supports)
    weighted_ece = float(np.average(class_eces, weights=class_supports)) if total_supp > 0 else macro_ece
    overall_brier = compute_brier_score(preds_prob, true, is_multilabel=True) if len(true) > 0 else 0.0
    
    return {
        "micro_precision": float(p_micro),
        "micro_recall": float(r_micro),
        "micro_f1": float(f1_micro),
        "macro_precision": float(p_macro),
        "macro_recall": float(r_macro),
        "macro_f1": float(f1_macro),
        "weighted_precision": float(p_weight),
        "weighted_recall": float(r_weight),
        "weighted_f1": float(f1_weight),
        "ece": float(macro_ece),
        "macro_ece": float(macro_ece),
        "micro_ece": float(micro_ece),
        "weighted_ece": float(weighted_ece),
        "brier": float(overall_brier),
        "per_class": per_class,
        "per_emotion": per_emotion,
    }

def compute_risk_metrics(
    predictions: torch.Tensor,   # (N,) — predicted class
    labels: torch.Tensor,         # (N,) — true class
    probabilities: torch.Tensor,  # (N, num_classes)
    num_classes: int = 3,
) -> Dict:
    """Compute risk classification metrics with calibration."""
    if isinstance(predictions, torch.Tensor):
        preds = predictions.detach().cpu().numpy()
    else:
        preds = np.array(predictions)
        
    if isinstance(labels, torch.Tensor):
        true = labels.detach().cpu().numpy()
    else:
        true = np.array(labels)
        
    if isinstance(probabilities, torch.Tensor):
        probs = probabilities.detach().cpu().numpy()
    else:
        probs = np.array(probabilities)
    
    p_micro, r_micro, f1_micro, _ = precision_recall_fscore_support(true, preds, average='micro', zero_division=0)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(true, preds, average='macro', zero_division=0)
    p_weight, r_weight, f1_weight, _ = precision_recall_fscore_support(true, preds, average='weighted', zero_division=0)
    
    # Per-class risk metrics (normal, distressed, risk)
    p_risk_classes, r_risk_classes, f1_risk_classes, risk_support = precision_recall_fscore_support(
        true, preds, labels=list(range(num_classes)), average=None, zero_division=0
    )
    risk_class_names = ["normal", "distressed", "risk"]
    per_risk_class = {}
    for i in range(num_classes):
        cname = risk_class_names[i] if i < len(risk_class_names) else f"class_{i}"
        true_binary = (true == i).astype(float)
        ece_r = binary_expected_calibration_error(probs[:, i], true_binary) if len(true) > 0 else 0.0
        brier_r = float(np.mean((probs[:, i] - true_binary) ** 2)) if len(true) > 0 else 0.0
        per_risk_class[cname] = {
            "name": cname,
            "precision": float(p_risk_classes[i]),
            "recall": float(r_risk_classes[i]),
            "f1": float(f1_risk_classes[i]),
            "ece": float(ece_r),
            "brier": float(brier_r),
            "support": int(risk_support[i]),
        }

    # Risk-specific (Class 2 = risk)
    risk_preds = (preds == 2).astype(int)
    risk_labels = (true == 2).astype(int)
    
    tp = int(np.sum((risk_preds == 1) & (risk_labels == 1)))
    fp = int(np.sum((risk_preds == 1) & (risk_labels == 0)))
    fn = int(np.sum((risk_preds == 0) & (risk_labels == 1)))
    tn = int(np.sum((risk_preds == 0) & (risk_labels == 0)))
    
    risk_precision = tp / (tp + fp + 1e-8)
    risk_recall = tp / (tp + fn + 1e-8)
    risk_f1 = 2 * risk_precision * risk_recall / (risk_precision + risk_recall + 1e-8)
    
    overall_ece = expected_calibration_error(probs, true, is_multilabel=False) if len(true) > 0 else 0.0
    overall_brier = compute_brier_score(probs, true, is_multilabel=False, num_classes=num_classes) if len(true) > 0 else 0.0
    
    return {
        "accuracy": float(accuracy_score(true, preds)),
        "micro_precision": float(p_micro),
        "micro_recall": float(r_micro),
        "micro_f1": float(f1_micro),
        "macro_precision": float(p_macro),
        "macro_recall": float(r_macro),
        "macro_f1": float(f1_macro),
        "weighted_precision": float(p_weight),
        "weighted_recall": float(r_weight),
        "weighted_f1": float(f1_weight),
        "risk_tp": tp, "risk_fp": fp, "risk_fn": fn, "risk_tn": tn,
        "risk_precision": float(risk_precision),
        "risk_recall": float(risk_recall),
        "risk_f1": float(risk_f1),
        "risk_false_negative_rate": float(fn / (fn + tp + 1e-8)),
        "risk_false_positive_rate": float(fp / (fp + tn + 1e-8)),
        "ece": float(overall_ece),
        "brier": float(overall_brier),
        "per_risk_class": per_risk_class,
        "confusion_matrix": confusion_matrix(true, preds, labels=list(range(num_classes))).tolist()
    }

def compute_risk_by_type(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    risk_types: List[str],
) -> Dict[str, Dict]:
    """Break down risk metrics by risk type for failure analysis."""
    type_metrics = defaultdict(lambda: {"correct": 0, "total": 0})
    for pred, label, rtype in zip(predictions.numpy(), labels.numpy(), risk_types):
        type_metrics[rtype]["total"] += 1
        if pred == label:
            type_metrics[rtype]["correct"] += 1
    return {rtype: {"accuracy": m["correct"]/max(m["total"],1), "support": m["total"]} for rtype, m in type_metrics.items()}

def compute_latency(model, tokenizer, sample_texts, device, num_warmup=5, num_runs=20):
    import time
    model.eval()
    for text in sample_texts[:num_warmup]:
        encoded = tokenizer.encode(text)
        input_ids = encoded["input_ids"].unsqueeze(0).to(device)
        attention_mask = encoded["attention_mask"].unsqueeze(0).to(device)
        with torch.no_grad(): _ = model(input_ids, attention_mask)
    
    latencies = []
    for text in sample_texts[:num_runs]:
        encoded = tokenizer.encode(text)
        input_ids = encoded["input_ids"].unsqueeze(0).to(device)
        attention_mask = encoded["attention_mask"].unsqueeze(0).to(device)
        start = time.perf_counter()
        with torch.no_grad(): _ = model(input_ids, attention_mask)
        end = time.perf_counter()
        latencies.append((end - start) * 1000)
        
    return {
        "mean_ms": float(np.mean(latencies)),
        "p50_ms": float(np.percentile(latencies, 50)),
        "p95_ms": float(np.percentile(latencies, 95)),
        "p99_ms": float(np.percentile(latencies, 99)),
    }
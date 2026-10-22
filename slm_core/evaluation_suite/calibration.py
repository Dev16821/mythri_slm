"""
Calibration evaluation for the cognitive SLM.

A well-calibrated model should:
- When it says "80% confident", it should be right ~80% of the time.
- Not be overconfident (saying 99% when it's wrong).
- Not be underconfident (saying 40% when it's always right).

This is especially critical for the risk head — an overconfident
risk model might miss crises, while an underconfident one triggers
too many false alarms.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple


def binary_expected_calibration_error(
    probabilities: torch.Tensor,
    labels: torch.Tensor,
    num_bins: int = 15,
) -> float:
    """
    Compute binary Expected Calibration Error (ECE) for an individual label or class.
    
    Groups probabilities into bins and measures the gap between the mean predicted
    probability and the empirical positive class fraction in each bin.
    """
    if isinstance(probabilities, torch.Tensor):
        probs = probabilities.detach().cpu().numpy()
    else:
        probs = np.asarray(probabilities)
        
    if isinstance(labels, torch.Tensor):
        true = labels.detach().cpu().numpy()
    else:
        true = np.asarray(labels)
        
    probs = np.clip(probs.flatten(), 0.0, 1.0)
    true = (true.flatten() > 0.5).astype(np.float32)
    
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    ece = 0.0
    n = len(probs)
    if n == 0:
        return 0.0
        
    for i in range(num_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        mask = (probs > lo) & (probs <= hi)
        if i == 0:
            mask = mask | (probs == lo)
            
        bin_size = mask.sum()
        if bin_size > 0:
            bin_confidence = probs[mask].mean()
            bin_accuracy = true[mask].mean()
            ece += (bin_size / n) * abs(bin_confidence - bin_accuracy)
            
    return float(ece)


def multilabel_expected_calibration_error(
    probabilities: torch.Tensor,
    labels: torch.Tensor,
    num_bins: int = 15,
    mode: str = "macro",
) -> float:
    """
    Compute multi-label Expected Calibration Error (ECE) across all classes.
    
    In multi-label classification (e.g. 35 emotion classes), each class is an
    independent binary prediction. This function evaluates all emotion classes
    rather than discarding them in favor of the highest-scoring class.
    
    Modes:
    - 'macro' (default): Unweighted average of binary ECE across all classes.
      Treats each emotion class equally.
    - 'micro' (or 'global'): Global binary ECE pooled across all (sample, class) predictions.
    - 'weighted': Support-weighted average of per-class binary ECEs.
    
    Args:
        probabilities: (N, num_classes) sigmoid probabilities
        labels: (N, num_classes) binary ground-truth labels
        num_bins: Number of confidence bins
        mode: Aggregation mode ('macro', 'micro', or 'weighted')
    
    Returns:
        float: Expected calibration error (lower is better, 0.0 = perfect calibration)
    """
    if isinstance(probabilities, torch.Tensor):
        probs = probabilities.detach().cpu().numpy()
    else:
        probs = np.asarray(probabilities)
        
    if isinstance(labels, torch.Tensor):
        true = labels.detach().cpu().numpy()
    else:
        true = np.asarray(labels)
        
    if probs.size == 0 or true.size == 0:
        return 0.0
        
    if probs.ndim == 1:
        probs = probs[:, np.newaxis]
        true = true[:, np.newaxis]
        
    num_samples, num_classes = probs.shape
    if num_samples == 0 or num_classes == 0:
        return 0.0
        
    probs = np.clip(probs, 0.0, 1.0)
    true = (true > 0.5).astype(np.float32)
    
    mode_normalized = mode.lower().strip()
    
    if mode_normalized in ("micro", "global"):
        return binary_expected_calibration_error(
            probs.flatten(), true.flatten(), num_bins=num_bins
        )
        
    per_class_eces = [
        binary_expected_calibration_error(probs[:, c], true[:, c], num_bins=num_bins)
        for c in range(num_classes)
    ]
    
    if mode_normalized == "weighted":
        supports = true.sum(axis=0)
        total_support = supports.sum()
        if total_support > 0:
            return float(np.average(per_class_eces, weights=supports))
        return float(np.mean(per_class_eces))
        
    # Default: macro
    return float(np.mean(per_class_eces))


def expected_calibration_error(
    probabilities: torch.Tensor,   # (N, num_classes) or (N, num_labels)
    labels: torch.Tensor,           # (N,) class indices or (N, num_labels) multi-hot
    num_bins: int = 15,
    is_multilabel: bool = False,
    multilabel_mode: str = "macro",
) -> float:
    """
    Expected Calibration Error (ECE).
    
    Groups predictions into confidence bins and measures
    the difference between confidence and accuracy in each bin.
    
    For multi-label tasks (is_multilabel=True), evaluates all classes
    using multilabel_expected_calibration_error across all classes.
    
    For multi-class tasks (is_multilabel=False), standard top-1 ECE is computed.
    
    Lower is better. 0 = perfectly calibrated.
    """
    if is_multilabel:
        return multilabel_expected_calibration_error(
            probabilities, labels, num_bins=num_bins, mode=multilabel_mode
        )

    if not isinstance(probabilities, torch.Tensor):
        probabilities = torch.tensor(probabilities, dtype=torch.float32)
    else:
        probabilities = probabilities.detach().cpu().float()
        
    if not isinstance(labels, torch.Tensor):
        labels = torch.tensor(labels)
    else:
        labels = labels.detach().cpu()
        
    if labels.ndim > 1 and labels.shape[-1] == 1:
        labels = labels.squeeze(-1)
    elif labels.ndim > 1 and labels.shape[-1] > 1:
        labels = labels.argmax(dim=-1)
        
    # For multi-class: standard top-1 calibration
    confidences, predictions = probabilities.max(dim=-1)
    correct = (predictions == labels).float()
    
    confidences = confidences.numpy()
    correct = correct.numpy()
    
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    ece = 0.0
    n = len(confidences)
    if n == 0:
        return 0.0
    
    for i in range(num_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        mask = (confidences > lo) & (confidences <= hi)
        if i == 0:
            mask = mask | (confidences == lo)
        
        bin_size = mask.sum()
        if bin_size > 0:
            bin_confidence = confidences[mask].mean()
            bin_accuracy = correct[mask].mean()
            ece += (bin_size / n) * abs(bin_confidence - bin_accuracy)
    
    return float(ece)


def compute_brier_score(
    probabilities: torch.Tensor,
    labels: torch.Tensor,
    is_multilabel: bool = False,
    num_classes: int = 3,
) -> float:
    """
    Compute Brier score (mean squared calibration error, lower is better, 0 = perfect).
    """
    if isinstance(probabilities, torch.Tensor):
        probs = probabilities.detach().cpu().float()
    else:
        probs = torch.tensor(probabilities, dtype=torch.float32)
        
    if isinstance(labels, torch.Tensor):
        true = labels.detach().cpu().float()
    else:
        true = torch.tensor(labels, dtype=torch.float32)
        
    if len(probs) == 0:
        return 0.0
        
    if is_multilabel:
        return float(((probs - true) ** 2).mean().item())
    else:
        if true.ndim > 1 and true.shape[-1] == 1:
            true = true.squeeze(-1)
        elif true.ndim > 1 and true.shape[-1] > 1:
            true = true.argmax(dim=-1)
        one_hot = torch.zeros(len(true), num_classes)
        one_hot.scatter_(1, true.long().unsqueeze(1), 1.0)
        return float(((probs - one_hot) ** 2).sum(dim=-1).mean().item())


def reliability_diagram_data(
    probabilities: torch.Tensor,
    labels: torch.Tensor,
    num_bins: int = 15,
    is_multilabel: bool = False,
) -> Dict:
    """
    Compute data for a reliability diagram.
    
    For multi-label evaluation (is_multilabel=True), pools all (sample, class)
    binary predictions to evaluate calibration across the full output space.
    For multi-class evaluation (is_multilabel=False), uses standard top-1 predictions.
    
    Returns bin-level confidence and accuracy for plotting.
    """
    if is_multilabel:
        if isinstance(probabilities, torch.Tensor):
            probs = probabilities.detach().cpu().float().numpy()
        else:
            probs = np.asarray(probabilities, dtype=np.float32)
            
        if isinstance(labels, torch.Tensor):
            true = labels.detach().cpu().float().numpy()
        else:
            true = np.asarray(labels, dtype=np.float32)
            
        probs = np.clip(probs.flatten(), 0.0, 1.0)
        true = (true.flatten() > 0.5).astype(np.float32)
        
        confidences = probs
        correct = true
    else:
        if not isinstance(probabilities, torch.Tensor):
            probabilities = torch.tensor(probabilities, dtype=torch.float32)
        else:
            probabilities = probabilities.detach().cpu().float()
            
        if not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels)
        else:
            labels = labels.detach().cpu()
            
        if labels.ndim > 1 and labels.shape[-1] == 1:
            labels = labels.squeeze(-1)
        elif labels.ndim > 1 and labels.shape[-1] > 1:
            labels = labels.argmax(dim=-1)
            
        confidences, predictions = probabilities.max(dim=-1)
        correct = (predictions == labels).float()
        
        confidences = confidences.numpy()
        correct = correct.numpy()
        
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    bins = []
    n = len(confidences)
    if n == 0:
        return {"bins": []}
        
    for i in range(num_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        mask = (confidences > lo) & (confidences <= hi)
        if i == 0:
            mask = mask | (confidences == lo)
        
        bin_size = mask.sum()
        if bin_size > 0:
            bins.append({
                "bin_lo": float(lo),
                "bin_hi": float(hi),
                "confidence": float(confidences[mask].mean()),
                "accuracy": float(correct[mask].mean()),
                "count": int(bin_size),
            })
    
    return {"bins": bins}


def compute_calibration_metrics(
    emotion_probs: torch.Tensor,
    emotion_labels: torch.Tensor,
    risk_probs: torch.Tensor,
    risk_labels: torch.Tensor,
    num_bins: int = 15,
) -> Dict:
    """Compute calibration metrics for both emotion and risk heads."""
    num_risk_classes = risk_probs.size(1) if (isinstance(risk_probs, torch.Tensor) and len(risk_probs.shape) > 1) else 3
    
    emotion_macro_ece = multilabel_expected_calibration_error(
        emotion_probs, emotion_labels, num_bins=num_bins, mode="macro"
    )
    emotion_micro_ece = multilabel_expected_calibration_error(
        emotion_probs, emotion_labels, num_bins=num_bins, mode="micro"
    )
    emotion_weighted_ece = multilabel_expected_calibration_error(
        emotion_probs, emotion_labels, num_bins=num_bins, mode="weighted"
    )
    risk_ece = expected_calibration_error(
        risk_probs, risk_labels, num_bins=num_bins, is_multilabel=False
    )
    
    return {
        "emotion_ece": float(emotion_macro_ece),
        "emotion_macro_ece": float(emotion_macro_ece),
        "emotion_micro_ece": float(emotion_micro_ece),
        "emotion_weighted_ece": float(emotion_weighted_ece),
        "risk_ece": float(risk_ece),
        "emotion_brier": float(compute_brier_score(
            emotion_probs, emotion_labels, is_multilabel=True
        )),
        "risk_brier": float(compute_brier_score(
            risk_probs, risk_labels, is_multilabel=False, num_classes=num_risk_classes
        )),
        "emotion_reliability": reliability_diagram_data(
            emotion_probs, emotion_labels, num_bins=num_bins, is_multilabel=True
        ),
        "risk_reliability": reliability_diagram_data(
            risk_probs, risk_labels, num_bins=num_bins, is_multilabel=False
        ),
    }
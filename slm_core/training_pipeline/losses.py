"""
Loss functions for multi-task training of the cognitive SLM.

Each loss is designed with a specific purpose:

- MultiLabelFocalLoss: Class-balanced focal loss for multi-label emotion
  classification. Resolves extreme class disparity across 35 emotions (joy=5142 vs grief=4)
  by combining per-class positive vs. negative balancing factors (alpha_c),
  per-class inverse-frequency weights (w_c), and positive-class weights (pos_weight_c)
  so predicting 0 on rare emotions is heavily penalized.

- MaskedBCELoss: Multi-label BCE for emotion classification with task masking,
  label smoothing, and positive class weighting.

- FocalLoss: Class-balanced focal loss for multi-class risk classification.
  Accepts per-class alpha vectors to prioritize minority crisis and distressed classes
  over the majority normal conversational class.

- MultiTaskLoss: Combines emotion and risk losses with balancing weights.
"""

import json
from pathlib import Path
from collections import Counter
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_risk_class_weights(
    filepath: Optional[str] = "datasets/raw/risk_train.jsonl",
    num_classes: int = 3,
) -> torch.Tensor:
    """
    Computes class balancing weights for risk classes based on inverse class frequency.
    Default classes: 0 = normal, 1 = distressed, 2 = crisis risk.
    """
    p = Path(filepath) if filepath else None
    if p and p.exists():
        counts = Counter()
        with open(p, "r") as f:
            for line in f:
                counts[json.loads(line)["label"]] += 1
        total = sum(counts.values())
        raw_weights = [total / (num_classes * max(counts[i], 1)) for i in range(num_classes)]
        w = torch.tensor(raw_weights, dtype=torch.float32)
        return w / w.mean()
    # Default balanced weights: normal: 0.60, distressed: 1.50, risk: 1.20
    return torch.tensor([0.60, 1.50, 1.20], dtype=torch.float32)


def compute_emotion_class_weights(
    filepath: Optional[str] = "datasets/raw/emotion_train.jsonl",
    num_classes: int = 35,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes class-balanced alpha, frequency weights, and positive-class weights
    for multi-label emotion classification to prevent predicting 0 for rare emotions.
    
    Returns:
        alpha_c: per-class positive vs. negative balancing factor in [0.55, 0.90]
        w_c: per-class frequency weighting factor across the 35 emotion classes (mean = 1.0)
        pos_weight: per-class positive weight multiplier in [1.0, 25.0] to heavily penalize false negatives
    """
    p = Path(filepath) if filepath else None
    if p and p.exists():
        counts = Counter()
        total_samples = 0
        with open(p, "r") as f:
            for line in f:
                total_samples += 1
                for lbl in json.loads(line).get("labels", []):
                    counts[lbl] += 1
        c_tensor = torch.tensor([counts[i] for i in range(num_classes)], dtype=torch.float32)
        min_c, max_c = c_tensor.min(), c_tensor.max()
        norm_c = (c_tensor - min_c) / (max_c - min_c).clamp(min=1.0)
        # Rare classes get higher positive weighting alpha_c in [0.55, 0.90]
        alpha_c = 0.55 + 0.35 * (1.0 - norm_c)
        # Smoothed inverse frequency weight w_c across 35 emotions
        inv_freq = 1.0 / (c_tensor + 5.0) ** 0.35
        w_c = inv_freq / inv_freq.mean()
        # Positive class weight multiplier pos_weight (smoothed inverse positive frequency)
        pos_weight = torch.sqrt((total_samples - c_tensor) / c_tensor.clamp(min=1.0)).clamp(min=1.0, max=25.0)
        return alpha_c, w_c, pos_weight
    return (
        torch.full((num_classes,), 0.75),
        torch.ones(num_classes),
        torch.full((num_classes,), 5.0),
    )


class MaskedBCELoss(nn.Module):
    """
    Binary cross-entropy loss for multi-label emotion classification with label smoothing
    and positive class weighting.
    
    Only computes loss on examples where emotion labels are available (emotion_mask == 1).
    Genuine 'no emotion detected' examples (where all labels are 0) with emotion_mask == 1
    are preserved and penalized for false positives.
    """

    def __init__(
        self,
        label_smoothing: float = 0.05,
        pos_weight: Optional[Union[float, List[float], torch.Tensor]] = None,
        class_weights: Optional[Union[List[float], torch.Tensor]] = None,
        num_classes: int = 35,
    ):
        super().__init__()
        self.label_smoothing = label_smoothing
        self.num_classes = num_classes

        if pos_weight is not None:
            if isinstance(pos_weight, (int, float)):
                pw = torch.full((num_classes,), float(pos_weight), dtype=torch.float32)
            elif isinstance(pos_weight, (list, tuple)):
                pw = torch.tensor(pos_weight, dtype=torch.float32)
            elif isinstance(pos_weight, torch.Tensor):
                pw = pos_weight.clone().detach().float()
            else:
                raise ValueError(f"Unsupported pos_weight type: {type(pos_weight)}")
            self.register_buffer("pos_weight", pw)
        else:
            _, _, default_pw = compute_emotion_class_weights(num_classes=num_classes)
            self.register_buffer("pos_weight", default_pw)

        if class_weights is not None:
            if isinstance(class_weights, (int, float)):
                cw = torch.full((num_classes,), float(class_weights), dtype=torch.float32)
            elif isinstance(class_weights, (list, tuple)):
                cw = torch.tensor(class_weights, dtype=torch.float32)
            elif isinstance(class_weights, torch.Tensor):
                cw = class_weights.clone().detach().float()
            else:
                raise ValueError(f"Unsupported class_weights type: {type(class_weights)}")
            self.register_buffer("class_weights", cw)
        else:
            _, default_cw, _ = compute_emotion_class_weights(num_classes=num_classes)
            self.register_buffer("class_weights", default_cw)

    def forward(
        self,
        logits: torch.Tensor,       # (B, num_emotions)
        labels: torch.Tensor,        # (B, num_emotions) — -1 for unlabeled
        mask: Optional[torch.Tensor] = None, # (B,) — 1 if example has emotion labels
    ) -> torch.Tensor:
        if mask is None:
            mask = torch.ones(logits.size(0), device=logits.device)
        valid = mask.bool()
        if valid.sum() == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        # Replace -1 labels with 0 (unlabeled examples are excluded via mask)
        labels_clean = labels.clone()
        labels_clean[labels_clean == -1] = 0.0
        
        # Multi-label smoothing: y_smooth = y * (1 - 2*eps) + eps
        if self.label_smoothing > 0.0:
            labels_clean = labels_clean * (1.0 - 2.0 * self.label_smoothing) + self.label_smoothing
        
        pos_weight = self.pos_weight.to(logits.device) if self.pos_weight is not None else None
        loss = F.binary_cross_entropy_with_logits(logits, labels_clean, pos_weight=pos_weight, reduction="none")
        
        if self.class_weights is not None:
            loss = loss * self.class_weights.to(logits.device)

        loss = loss.mean(dim=-1)  # (B,) — average across emotions
        
        # Apply mask
        masked_loss = loss * mask
        num_valid = mask.sum().clamp(min=1.0)
        return masked_loss.sum() / num_valid


class MultiLabelFocalLoss(nn.Module):
    """
    Class-balanced focal loss for multi-label emotion classification with label smoothing
    and positive class weighting.
    
    Resolves the extreme class imbalance across the 35 emotions (e.g., joy=5142 vs grief=4)
    and positive-vs-negative imbalance within each class:
    
    FL(p_c, y_c) = w_c * [ - pos_weight_c * alpha_c * y_c * (1 - p_c)^gamma * log(p_c)
                          - (1 - alpha_c) * (1 - y_c) * p_c^gamma * log(1 - p_c) ]
                          
    - alpha_c in (0, 1): per-class positive vs. negative balancing factor
    - w_c: per-class frequency balancing weight across the 35 emotions
    - pos_weight_c: positive weight multiplier to severely penalize predicting 0 for rare emotions
    - mask: filters out unlabeled examples while keeping genuine all-zero 'no emotion detected' examples
    """

    def __init__(
        self,
        alpha: Optional[Union[float, List[float], torch.Tensor]] = None,
        gamma: float = 2.0,
        num_classes: int = 35,
        class_weights: Optional[Union[List[float], torch.Tensor]] = None,
        pos_weight: Optional[Union[float, List[float], torch.Tensor]] = None,
        label_smoothing: float = 0.05,
    ):
        super().__init__()
        self.gamma = gamma
        self.num_classes = num_classes
        self.label_smoothing = label_smoothing

        if alpha is None:
            default_alpha, _, _ = compute_emotion_class_weights(num_classes=num_classes)
            alpha_tensor = default_alpha
        elif isinstance(alpha, (int, float)):
            alpha_tensor = torch.full((num_classes,), float(alpha), dtype=torch.float32)
        elif isinstance(alpha, (list, tuple)):
            alpha_tensor = torch.tensor(alpha, dtype=torch.float32)
        elif isinstance(alpha, torch.Tensor):
            alpha_tensor = alpha.clone().detach().float()
        else:
            raise ValueError(f"Unsupported alpha type: {type(alpha)}")
        self.register_buffer("alpha", alpha_tensor)

        if class_weights is not None:
            if isinstance(class_weights, (int, float)):
                w_tensor = torch.full((num_classes,), float(class_weights), dtype=torch.float32)
            elif isinstance(class_weights, (list, tuple)):
                w_tensor = torch.tensor(class_weights, dtype=torch.float32)
            elif isinstance(class_weights, torch.Tensor):
                w_tensor = class_weights.clone().detach().float()
            else:
                raise ValueError(f"Unsupported class_weights type: {type(class_weights)}")
            self.register_buffer("class_weights", w_tensor)
        else:
            _, default_cw, _ = compute_emotion_class_weights(num_classes=num_classes)
            self.register_buffer("class_weights", default_cw)

        if pos_weight is not None:
            if isinstance(pos_weight, (int, float)):
                pw_tensor = torch.full((num_classes,), float(pos_weight), dtype=torch.float32)
            elif isinstance(pos_weight, (list, tuple)):
                pw_tensor = torch.tensor(pos_weight, dtype=torch.float32)
            elif isinstance(pos_weight, torch.Tensor):
                pw_tensor = pos_weight.clone().detach().float()
            else:
                raise ValueError(f"Unsupported pos_weight type: {type(pos_weight)}")
            self.register_buffer("pos_weight", pw_tensor)
        else:
            _, _, default_pw = compute_emotion_class_weights(num_classes=num_classes)
            self.register_buffer("pos_weight", default_pw)

    def forward(
        self,
        logits: torch.Tensor,       # (B, num_classes)
        labels: torch.Tensor,       # (B, num_classes) — -1 for unlabeled
        mask: Optional[torch.Tensor] = None, # (B,) — 1 if example has emotion labels
    ) -> torch.Tensor:
        if mask is None:
            mask = torch.ones(logits.size(0), device=logits.device)
        valid = mask.bool()
        if valid.sum() == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        labels_clean = labels.clone()
        labels_clean[labels_clean == -1] = 0.0
        
        # Multi-label smoothing
        if self.label_smoothing > 0.0:
            labels_clean = labels_clean * (1.0 - 2.0 * self.label_smoothing) + self.label_smoothing

        probs = torch.sigmoid(logits)
        p_t = labels_clean * probs + (1.0 - labels_clean) * (1.0 - probs)

        alpha_tensor = self.alpha.to(logits.device)
        alpha_t = labels_clean * alpha_tensor + (1.0 - labels_clean) * (1.0 - alpha_tensor)

        focal_weight = (1.0 - p_t).clamp(min=0.0) ** self.gamma
        pw = self.pos_weight.to(logits.device)
        bce = F.binary_cross_entropy_with_logits(logits, labels_clean, pos_weight=pw, reduction="none")

        cw = self.class_weights.to(logits.device)
        per_class_loss = cw * alpha_t * focal_weight * bce  # (B, num_classes)
        loss = per_class_loss.mean(dim=-1)                  # (B,)

        masked_loss = loss * mask
        num_valid = mask.sum().clamp(min=1.0)
        return masked_loss.sum() / num_valid


class FocalLoss(nn.Module):
    """
    Class-balanced Focal loss for multi-class risk classification with label smoothing.
    
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    
    Supports per-class alpha (vector/list) to prioritize minority crisis and distressed classes,
    preventing majority (normal) classes from dominating training.
    """

    def __init__(
        self,
        alpha: Optional[Union[float, List[float], torch.Tensor]] = None,
        gamma: float = 2.0,
        num_classes: int = 3,
        label_smoothing: float = 0.05,
        class_weights: Optional[Union[List[float], torch.Tensor]] = None,
    ):
        super().__init__()
        self.gamma = gamma
        self.num_classes = num_classes
        self.label_smoothing = label_smoothing

        if alpha is None:
            # Default balanced weights: [normal: 0.60, distressed: 1.50, risk: 1.20]
            alpha_tensor = torch.tensor([0.60, 1.50, 1.20], dtype=torch.float32)
        elif isinstance(alpha, (int, float)):
            if class_weights is not None:
                cw = torch.tensor(class_weights, dtype=torch.float32) if isinstance(class_weights, (list, tuple)) else class_weights.clone().detach().float()
                alpha_tensor = float(alpha) * cw
            elif num_classes == 3 and float(alpha) == 0.25:
                # Upgrade legacy scalar 0.25 to class-balanced alpha preserving relative scale
                alpha_tensor = torch.tensor([0.15, 0.375, 0.30], dtype=torch.float32)  # [normal: 0.15, distressed: 0.375, risk: 0.30]
            else:
                alpha_tensor = torch.full((num_classes,), float(alpha), dtype=torch.float32)
        elif isinstance(alpha, (list, tuple)):
            alpha_tensor = torch.tensor(alpha, dtype=torch.float32)
        elif isinstance(alpha, torch.Tensor):
            alpha_tensor = alpha.clone().detach().float()
        else:
            raise ValueError(f"Unsupported alpha type: {type(alpha)}")

        if class_weights is not None and not isinstance(alpha, (int, float)):
            cw = torch.tensor(class_weights, dtype=torch.float32) if isinstance(class_weights, (list, tuple)) else class_weights.clone().detach().float()
            alpha_tensor = alpha_tensor * cw

        self.register_buffer("alpha", alpha_tensor)

    def forward(
        self,
        logits: torch.Tensor,       # (B, num_classes)
        labels: torch.Tensor,       # (B,) — class index, -1 for unlabeled
        mask: Optional[torch.Tensor] = None, # (B,) — 1 if example has risk label
    ) -> torch.Tensor:
        if mask is None:
            mask = torch.ones(logits.size(0), device=logits.device)
        # Filter to valid examples
        valid = mask.bool()
        if valid.sum() == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)
        
        logits_valid = logits[valid]          # (N, num_classes)
        labels_valid = labels[valid]           # (N,)
        
        # Compute log softmax and probabilities
        log_probs = F.log_softmax(logits_valid, dim=-1)
        probs = torch.exp(log_probs)
        
        # One-hot target with label smoothing
        one_hot = F.one_hot(labels_valid, self.num_classes).float()
        if self.label_smoothing > 0.0:
            one_hot = one_hot * (1.0 - self.label_smoothing) + (self.label_smoothing / self.num_classes)
        
        # p_t = probability of true class
        p_t = (probs * one_hot).sum(dim=-1)  # (N,)
        
        # Focal modulation
        focal_weight = (1.0 - p_t).clamp(min=0.0) ** self.gamma
        
        # Class-specific alpha_t: (N,)
        alpha_tensor = self.alpha.to(logits.device)
        alpha_t = (alpha_tensor * one_hot).sum(dim=-1)
        
        loss = -alpha_t * focal_weight * (one_hot * log_probs).sum(dim=-1)
        return loss.mean()


class MultiTaskLoss(nn.Module):
    """
    Combined loss for multi-task training with regularized smoothed losses
    and true class-balanced focal weighting.
    
    total_loss = w_emotion * emotion_loss + w_risk * risk_loss
    """

    def __init__(
        self,
        emotion_weight: float = 1.0,
        risk_weight: float = 0.8,
        focal_alpha: Optional[Union[float, List[float], torch.Tensor]] = None,
        focal_gamma: float = 2.0,
        num_risk_classes: int = 3,
        num_emotions: int = 35,
        emotion_loss_type: str = "focal",
        emotion_focal_alpha: Optional[Union[float, List[float], torch.Tensor]] = None,
        emotion_focal_gamma: float = 2.0,
        emotion_class_weights: Optional[Union[List[float], torch.Tensor]] = None,
        emotion_pos_weight: Optional[Union[float, List[float], torch.Tensor]] = None,
        risk_focal_alpha: Optional[Union[float, List[float], torch.Tensor]] = None,
        risk_focal_gamma: float = 2.0,
        emotion_label_smoothing: float = 0.05,
        risk_label_smoothing: float = 0.05,
    ):
        super().__init__()
        self.emotion_weight = emotion_weight
        self.risk_weight = risk_weight
        self.emotion_loss_type = emotion_loss_type.lower()

        # Class balancing for emotion head
        default_alpha, default_weights, default_pos_weights = compute_emotion_class_weights(num_classes=num_emotions)
        emo_alpha = emotion_focal_alpha if emotion_focal_alpha is not None else default_alpha
        emo_weights = emotion_class_weights if emotion_class_weights is not None else default_weights
        emo_pos = emotion_pos_weight if emotion_pos_weight is not None else default_pos_weights

        if self.emotion_loss_type == "focal":
            self.emotion_loss = MultiLabelFocalLoss(
                alpha=emo_alpha,
                gamma=emotion_focal_gamma,
                num_classes=num_emotions,
                class_weights=emo_weights,
                pos_weight=emo_pos,
                label_smoothing=emotion_label_smoothing,
            )
        else:
            self.emotion_loss = MaskedBCELoss(
                label_smoothing=emotion_label_smoothing,
                pos_weight=emo_pos,
                class_weights=emo_weights,
                num_classes=num_emotions,
            )

        # Class balancing for risk head
        risk_alpha = risk_focal_alpha if risk_focal_alpha is not None else focal_alpha
        self.risk_loss = FocalLoss(
            alpha=risk_alpha,
            gamma=risk_focal_gamma if risk_focal_gamma is not None else focal_gamma,
            num_classes=num_risk_classes,
            label_smoothing=risk_label_smoothing,
        )

    def forward(
        self,
        emotion_logits: torch.Tensor,
        emotion_labels: torch.Tensor,
        emotion_mask: torch.Tensor,
        risk_logits: torch.Tensor,
        risk_labels: torch.Tensor,
        risk_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        emo_loss = self.emotion_loss(emotion_logits, emotion_labels, emotion_mask)
        rsk_loss = self.risk_loss(risk_logits, risk_labels, risk_mask)
        
        total = self.emotion_weight * emo_loss + self.risk_weight * rsk_loss
        
        return {
            "total": total,
            "emotion_loss": emo_loss.detach(),
            "risk_loss": rsk_loss.detach(),
        }
"""
Training loop for the cognitive SLM.

Features:
- Gradient accumulation (effective batch size > physical batch size)
- Gradient clipping
- Linear warmup + cosine decay learning rate schedule
- Periodic evaluation on validation set
- Checkpointing
- Training logging (stdout + file)
- Resource monitoring (CPU/memory)
"""

import os
import time
import json
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import AdamW
from typing import Dict, Optional, Callable, List
from pathlib import Path
import sys
import math

# Ensure root is in path
sys.path.insert(0, str(Path(__file__).parent.parent))

from .losses import MultiTaskLoss
from .patience import TrainingPatienceController
from evaluation.metrics import compute_emotion_metrics, compute_risk_metrics
from data_utils.synthetic_data import get_emotion_label_list


class CognitiveTrainer:
    """
    Trains the CognitiveSLM model with multi-task loss.
    
    Designed for constrained hardware:
    - Small batch size (32) with gradient accumulation (2x = 64 effective)
    - CPU-compatible (falls back from GPU gracefully)
    - Periodic checkpointing
    - Early stopping on validation metric
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict,
        device: Optional[torch.device] = None,
        checkpoint_dir: str = "checkpoints",
        log_dir: str = "logs",
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device or torch.device("cpu")
        
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_dir = Path(log_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Move model to device
        self.model.to(self.device)
        
        # Optimizer: filter out untrained heads so weight decay doesn't degrade them
        # Support differential learning rates when fine-tuning pretrained backbone
        base_lr = config["training"]["learning_rate"]
        weight_decay = config["training"]["weight_decay"]
        has_pretrained = getattr(model, "has_pretrained_backbone", False)
        backbone_lr = config["training"].get("backbone_learning_rate", None)

        no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight", "norm.weight", "layernorm.weight", "LayerNorm.bias", "layer_norm.bias", "norm.bias"]

        if has_pretrained or backbone_lr is not None:
            # Pretrained fine-tuning: lower LR for backbone (e.g. 0.1x base_lr or explicitly configured)
            actual_backbone_lr = backbone_lr if backbone_lr is not None else (base_lr * 0.1)
            backbone_scale = actual_backbone_lr / base_lr

            backbone_decay, backbone_no_decay = [], []
            head_decay, head_no_decay = [], []
            for n, p in model.named_parameters():
                if not p.requires_grad or any(head in n for head in ["boundary_head", "turn_function_head"]):
                    continue
                is_no_decay = any(nd in n for nd in no_decay)
                if "encoder" in n:
                    (backbone_no_decay if is_no_decay else backbone_decay).append(p)
                else:
                    (head_no_decay if is_no_decay else head_decay).append(p)

            param_groups = []
            if backbone_decay:
                param_groups.append({"params": backbone_decay, "lr": actual_backbone_lr, "weight_decay": weight_decay, "lr_scale": backbone_scale})
            if backbone_no_decay:
                param_groups.append({"params": backbone_no_decay, "lr": actual_backbone_lr, "weight_decay": 0.0, "lr_scale": backbone_scale})
            if head_decay:
                param_groups.append({"params": head_decay, "lr": base_lr, "weight_decay": weight_decay, "lr_scale": 1.0})
            if head_no_decay:
                param_groups.append({"params": head_no_decay, "lr": base_lr, "weight_decay": 0.0, "lr_scale": 1.0})
            self.optimizer = AdamW(param_groups, foreach=False)
        else:
            decay_params, no_decay_params = [], []
            for n, p in model.named_parameters():
                if not p.requires_grad or any(head in n for head in ["boundary_head", "turn_function_head"]):
                    continue
                if any(nd in n for nd in no_decay):
                    no_decay_params.append(p)
                else:
                    decay_params.append(p)

            param_groups = [
                {"params": decay_params, "lr": base_lr, "weight_decay": weight_decay, "lr_scale": 1.0},
                {"params": no_decay_params, "lr": base_lr, "weight_decay": 0.0, "lr_scale": 1.0},
            ]
            self.optimizer = AdamW(param_groups, foreach=False)

        # Loss with anti-overfitting label smoothing and class balancing
        label_smoothing = config.get("training", {}).get("label_smoothing", 0.05)
        emotion_loss_type = config.get("training", {}).get("emotion_loss_type", "focal")
        risk_alpha = config.get("training", {}).get("risk_focal_alpha", config.get("training", {}).get("focal_alpha"))
        emotion_alpha = config.get("training", {}).get("emotion_focal_alpha")
        emotion_weights = config.get("training", {}).get("emotion_class_weights")
        emotion_pos_weight = config.get("training", {}).get("emotion_pos_weight")

        self.criterion = MultiTaskLoss(
            emotion_weight=config["training"].get("emotion_loss_weight", 1.0),
            risk_weight=config["training"].get("risk_loss_weight", 0.8),
            focal_alpha=risk_alpha,
            focal_gamma=config["training"].get("focal_gamma", 2.0),
            num_risk_classes=config["model"]["num_risk_classes"],
            num_emotions=config["model"].get("num_emotions", 35),
            emotion_loss_type=emotion_loss_type,
            emotion_focal_alpha=emotion_alpha,
            emotion_focal_gamma=config["training"].get("emotion_focal_gamma", 2.0),
            emotion_class_weights=emotion_weights,
            emotion_pos_weight=emotion_pos_weight,
            risk_focal_alpha=risk_alpha,
            risk_focal_gamma=config["training"].get("risk_focal_gamma", config["training"].get("focal_gamma", 2.0)),
            emotion_label_smoothing=label_smoothing,
            risk_label_smoothing=label_smoothing,
        )
        
        # Training schedule
        self.epochs = config["training"]["epochs"]
        self.grad_accumulation = max(1, config["training"].get("grad_accumulation", 1))
        self.max_grad_norm = config["training"]["max_grad_norm"]
        self.warmup_steps = config["training"]["warmup_steps"]
        
        steps_per_epoch = math.ceil(len(train_loader) / self.grad_accumulation)
        total_steps = steps_per_epoch * self.epochs
        self.total_steps = max(total_steps, 1)
        
        # Logging
        self.log_entries: List[Dict] = []
        self.best_val_metric = float("inf")
        
        # Resource tracking
        self.train_times: List[float] = []
        self.eval_times: List[float] = []

        # Patience & Early Stopping Controller (overfitting & underfitting safeguards)
        early_stop_cfg = config.get("training", {}).get("early_stopping", {})
        patience = early_stop_cfg.get("patience", config.get("training", {}).get("patience", 4))
        min_delta = early_stop_cfg.get("min_delta", config.get("training", {}).get("min_delta", 1e-4))
        underfit_patience = early_stop_cfg.get("underfit_patience", config.get("training", {}).get("underfit_patience", 5))
        divergence_threshold = early_stop_cfg.get("divergence_threshold", config.get("training", {}).get("divergence_threshold", 3.0))
        restore_best = early_stop_cfg.get("restore_best_weights", config.get("training", {}).get("restore_best_weights", True))

        self.patience_controller = TrainingPatienceController(
            patience=patience,
            min_delta=min_delta,
            underfit_patience=underfit_patience,
            divergence_threshold=divergence_threshold,
            mode="min",
            restore_best_weights=restore_best,
            verbose=True,
        )

        # Emotion labels
        if "data" in config and "emotion_labels" in config["data"] and config["data"]["emotion_labels"]:
            self.emotion_labels = config["data"]["emotion_labels"]
        else:
            try:
                self.emotion_labels = get_emotion_label_list()
            except Exception:
                num_emo = config.get("model", {}).get("num_emotions", 35)
                self.emotion_labels = [f"emotion_{i}" for i in range(num_emo)]

    def _get_lr(self, step: int) -> float:
        """Linear warmup followed by cosine decay."""
        base_lr = self.config["training"]["learning_rate"]
        if step < self.warmup_steps:
            return base_lr * (step + 1) / self.warmup_steps
        else:
            progress = min(1.0, max(0.0, (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)))
            return 0.5 * base_lr * (1 + math.cos(math.pi * progress))

    def _evaluate(self) -> Dict:
        """Run evaluation on validation set."""
        self.model.eval()
        all_emotion_probs = []
        all_emotion_labels = []
        all_risk_probs = []
        all_risk_preds = []
        all_risk_labels = []
        total_loss = 0.0
        total_emo_loss = 0.0
        total_risk_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                emotion_labels = batch["emotion_labels"].to(self.device)
                emotion_mask = batch["emotion_mask"].to(self.device)
                risk_label = batch["risk_label"].to(self.device)
                risk_mask = batch["risk_mask"].to(self.device)
                
                output = self.model(input_ids, attention_mask)
                
                loss_dict = self.criterion(
                    emotion_logits=output.emotion_logits,
                    emotion_labels=emotion_labels,
                    emotion_mask=emotion_mask,
                    risk_logits=output.risk_logits,
                    risk_labels=risk_label,
                    risk_mask=risk_mask,
                )
                total_loss += loss_dict["total"].item()
                total_emo_loss += loss_dict["emotion_loss"].item()
                total_risk_loss += loss_dict["risk_loss"].item()
                num_batches += 1
                
                # Collect predictions
                emo_mask = emotion_mask.bool()
                if emo_mask.any():
                    all_emotion_probs.append(output.emotion_probs[emo_mask].cpu())
                    all_emotion_labels.append(emotion_labels[emo_mask].cpu())
                
                rsk_mask = risk_mask.bool()
                if rsk_mask.any():
                    all_risk_probs.append(output.risk_probs[rsk_mask].cpu())
                    all_risk_preds.append(output.risk_prediction[rsk_mask].cpu())
                    all_risk_labels.append(risk_label[rsk_mask].cpu())
        
        # Compute metrics
        metrics = {
            "val_loss": float(total_loss / max(num_batches, 1)),
            "val_emotion_loss": float(total_emo_loss / max(num_batches, 1)),
            "val_risk_loss": float(total_risk_loss / max(num_batches, 1)),
        }
        
        if all_emotion_probs:
            probs = torch.cat(all_emotion_probs)
            labels = torch.cat(all_emotion_labels)
            labels_clean = labels.clone()
            labels_clean[labels_clean == -1] = 0.0
            labels_clean = (labels_clean > 0).float()
            
            emo_metrics = compute_emotion_metrics(
                probs, labels_clean, threshold=0.5, emotion_labels=self.emotion_labels
            )
            
            # micro
            metrics["emotion_micro_f1"] = emo_metrics["micro_f1"]
            metrics["emotion_micro_precision"] = emo_metrics["micro_precision"]
            metrics["emotion_micro_recall"] = emo_metrics["micro_recall"]
            # macro
            metrics["emotion_macro_f1"] = emo_metrics["macro_f1"]
            metrics["emotion_macro_precision"] = emo_metrics["macro_precision"]
            metrics["emotion_macro_recall"] = emo_metrics["macro_recall"]
            # weighted
            metrics["emotion_weighted_f1"] = emo_metrics["weighted_f1"]
            metrics["emotion_weighted_precision"] = emo_metrics["weighted_precision"]
            metrics["emotion_weighted_recall"] = emo_metrics["weighted_recall"]
            # calibrated
            metrics["emotion_ece"] = emo_metrics.get("ece", 0.0)
            metrics["emotion_macro_ece"] = emo_metrics.get("macro_ece", emo_metrics.get("ece", 0.0))
            metrics["emotion_micro_ece"] = emo_metrics.get("micro_ece", 0.0)
            metrics["emotion_weighted_ece"] = emo_metrics.get("weighted_ece", 0.0)
            metrics["emotion_brier"] = emo_metrics.get("brier", 0.0)
            metrics["emotion_per_class"] = emo_metrics["per_class"]
            metrics["emotion_per_emotion"] = emo_metrics["per_emotion"]
        
        if all_risk_probs:
            probs = torch.cat(all_risk_probs)
            preds = torch.cat(all_risk_preds)
            labels = torch.cat(all_risk_labels)
            
            rsk_metrics = compute_risk_metrics(
                preds, labels, probs, num_classes=self.config["model"]["num_risk_classes"]
            )
            
            metrics["risk_accuracy"] = rsk_metrics["accuracy"]
            metrics["risk_precision"] = rsk_metrics["risk_precision"]
            metrics["risk_recall"] = rsk_metrics["risk_recall"]
            metrics["risk_f1"] = rsk_metrics["risk_f1"]
            # micro
            metrics["risk_micro_f1"] = rsk_metrics["micro_f1"]
            metrics["risk_micro_precision"] = rsk_metrics["micro_precision"]
            metrics["risk_micro_recall"] = rsk_metrics["micro_recall"]
            # macro
            metrics["risk_macro_f1"] = rsk_metrics["macro_f1"]
            metrics["risk_macro_precision"] = rsk_metrics["macro_precision"]
            metrics["risk_macro_recall"] = rsk_metrics["macro_recall"]
            # weighted
            metrics["risk_weighted_f1"] = rsk_metrics["weighted_f1"]
            metrics["risk_weighted_precision"] = rsk_metrics["weighted_precision"]
            metrics["risk_weighted_recall"] = rsk_metrics["weighted_recall"]
            # calibrated
            metrics["risk_ece"] = rsk_metrics.get("ece", 0.0)
            metrics["risk_brier"] = rsk_metrics.get("brier", 0.0)
            metrics["risk_false_negative_rate"] = rsk_metrics["risk_false_negative_rate"]
            metrics["risk_false_positive_rate"] = rsk_metrics["risk_false_positive_rate"]
            if "per_risk_class" in rsk_metrics:
                metrics["risk_per_class"] = rsk_metrics["per_risk_class"]
        
        return metrics

    def _print_epoch_summary(
        self,
        epoch: int,
        train_loss: float,
        train_emo_loss: float,
        train_risk_loss: float,
        val_metrics: Dict,
        epoch_time: float,
        eval_time: float,
        lr: float,
    ):
        """Prints a comprehensive, readable summary of all metrics and metrics per emotion for the epoch."""
        sep_double = "=" * 88
        sep_single = "-" * 88
        
        print(f"\n{sep_double}")
        print(f" EPOCH {epoch}/{self.epochs} EVALUATION & METRICS SUMMARY")
        print(f"{sep_double}")
        print(f"Timing: Train = {epoch_time:.1f}s | Eval = {eval_time:.1f}s | Learning Rate: {lr:.2e}")
        
        print(f"\n--- Losses ---")
        print(
            f"  Train: Total = {train_loss:.4f} | "
            f"Emotion BCE = {train_emo_loss:.4f} | "
            f"Risk Focal = {train_risk_loss:.4f}"
        )
        print(
            f"  Val:   Total = {val_metrics.get('val_loss', 0.0):.4f} | "
            f"Emotion BCE = {val_metrics.get('val_emotion_loss', 0.0):.4f} | "
            f"Risk Focal = {val_metrics.get('val_risk_loss', 0.0):.4f}"
        )
        
        print(f"\n--- Overall Emotion Metrics (Micro, Macro, Weighted, Calibrated) ---")
        print(
            f"  Micro:       F1 = {val_metrics.get('emotion_micro_f1', 0.0):.4f} | "
            f"Precision = {val_metrics.get('emotion_micro_precision', 0.0):.4f} | "
            f"Recall = {val_metrics.get('emotion_micro_recall', 0.0):.4f}"
        )
        print(
            f"  Macro:       F1 = {val_metrics.get('emotion_macro_f1', 0.0):.4f} | "
            f"Precision = {val_metrics.get('emotion_macro_precision', 0.0):.4f} | "
            f"Recall = {val_metrics.get('emotion_macro_recall', 0.0):.4f}"
        )
        print(
            f"  Weighted:    F1 = {val_metrics.get('emotion_weighted_f1', 0.0):.4f} | "
            f"Precision = {val_metrics.get('emotion_weighted_precision', 0.0):.4f} | "
            f"Recall = {val_metrics.get('emotion_weighted_recall', 0.0):.4f}"
        )
        print(
            f"  Calibration: Macro ECE = {val_metrics.get('emotion_macro_ece', val_metrics.get('emotion_ece', 0.0)):.4f} | "
            f"Micro ECE = {val_metrics.get('emotion_micro_ece', 0.0):.4f} | "
            f"Brier Score = {val_metrics.get('emotion_brier', 0.0):.4f}"
        )
        
        print(f"\n--- Overall Risk Metrics (Micro, Macro, Weighted, Calibrated) ---")
        print(
            f"  Micro:       F1 = {val_metrics.get('risk_micro_f1', 0.0):.4f} | "
            f"Precision = {val_metrics.get('risk_micro_precision', 0.0):.4f} | "
            f"Recall = {val_metrics.get('risk_micro_recall', 0.0):.4f}"
        )
        print(
            f"  Macro:       F1 = {val_metrics.get('risk_macro_f1', 0.0):.4f} | "
            f"Precision = {val_metrics.get('risk_macro_precision', 0.0):.4f} | "
            f"Recall = {val_metrics.get('risk_macro_recall', 0.0):.4f}"
        )
        print(
            f"  Weighted:    F1 = {val_metrics.get('risk_weighted_f1', 0.0):.4f} | "
            f"Precision = {val_metrics.get('risk_weighted_precision', 0.0):.4f} | "
            f"Recall = {val_metrics.get('risk_weighted_recall', 0.0):.4f}"
        )
        print(
            f"  Risk (Class 2 / Crisis): F1 = {val_metrics.get('risk_f1', 0.0):.4f} | "
            f"Precision = {val_metrics.get('risk_precision', 0.0):.4f} | "
            f"Recall = {val_metrics.get('risk_recall', 0.0):.4f}"
        )
        print(
            f"  Accuracy = {val_metrics.get('risk_accuracy', 0.0):.4f} | "
            f"FNR = {val_metrics.get('risk_false_negative_rate', 0.0):.2%} | "
            f"FPR = {val_metrics.get('risk_false_positive_rate', 0.0):.2%}"
        )
        print(
            f"  Calibration: ECE = {val_metrics.get('risk_ece', 0.0):.4f} | "
            f"Brier Score = {val_metrics.get('risk_brier', 0.0):.4f}"
        )
        
        per_risk = val_metrics.get("risk_per_class", {})
        if per_risk:
            print(f"\n--- Metrics Per Risk Class (with Calibration) ---")
            print(f"{'Class':<16} {'Precision':>11} {'Recall':>11} {'F1-Score':>11} {'ECE':>11} {'Brier':>11} {'Support':>11}")
            print(f"{sep_single}")
            for rname, rstats in per_risk.items():
                print(
                    f"{rname:<16} "
                    f"{rstats.get('precision', 0.0):>11.4f} "
                    f"{rstats.get('recall', 0.0):>11.4f} "
                    f"{rstats.get('f1', 0.0):>11.4f} "
                    f"{rstats.get('ece', 0.0):>11.4f} "
                    f"{rstats.get('brier', 0.0):>11.4f} "
                    f"{rstats.get('support', 0):>11d}"
                )
            print(f"{sep_single}")
        
        per_emotion = val_metrics.get("emotion_per_emotion", {})
        if per_emotion:
            print(f"\n--- Metrics Per Emotion (Epoch {epoch}) (with Calibration) ---")
            print(f"{'Emotion':<22} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'ECE':>10} {'Brier':>10} {'Support':>10}")
            print(f"{sep_single}")
            total_supp = 0
            for ename, estats in per_emotion.items():
                supp = estats.get("support", 0)
                total_supp += supp
                print(
                    f"{ename:<22} "
                    f"{estats.get('precision', 0.0):>10.4f} "
                    f"{estats.get('recall', 0.0):>10.4f} "
                    f"{estats.get('f1', 0.0):>10.4f} "
                    f"{estats.get('ece', 0.0):>10.4f} "
                    f"{estats.get('brier', 0.0):>10.4f} "
                    f"{supp:>10d}"
                )
            print(f"{sep_single}")
            macro_ece = float(np.mean([estats.get('ece', 0.0) for estats in per_emotion.values()]))
            macro_brier = float(np.mean([estats.get('brier', 0.0) for estats in per_emotion.values()]))
            print(
                f"{'Macro Avg':<22} "
                f"{val_metrics.get('emotion_macro_precision', 0.0):>10.4f} "
                f"{val_metrics.get('emotion_macro_recall', 0.0):>10.4f} "
                f"{val_metrics.get('emotion_macro_f1', 0.0):>10.4f} "
                f"{macro_ece:>10.4f} "
                f"{macro_brier:>10.4f} "
                f"{total_supp:>10d}"
            )
            print(
                f"{'Micro Avg':<22} "
                f"{val_metrics.get('emotion_micro_precision', 0.0):>10.4f} "
                f"{val_metrics.get('emotion_micro_recall', 0.0):>10.4f} "
                f"{val_metrics.get('emotion_micro_f1', 0.0):>10.4f} "
                f"{val_metrics.get('emotion_micro_ece', val_metrics.get('emotion_ece', 0.0)):>10.4f} "
                f"{val_metrics.get('emotion_brier', 0.0):>10.4f} "
                f"{total_supp:>10d}"
            )
            weights = [estats.get('support', 0) for estats in per_emotion.values()]
            denom = sum(weights) if sum(weights) > 0 else 1
            weighted_ece = sum(w * e.get('ece', 0.0) for w, e in zip(weights, per_emotion.values())) / denom
            weighted_brier = sum(w * e.get('brier', 0.0) for w, e in zip(weights, per_emotion.values())) / denom
            print(
                f"{'Weighted Avg':<22} "
                f"{val_metrics.get('emotion_weighted_precision', 0.0):>10.4f} "
                f"{val_metrics.get('emotion_weighted_recall', 0.0):>10.4f} "
                f"{val_metrics.get('emotion_weighted_f1', 0.0):>10.4f} "
                f"{weighted_ece:>10.4f} "
                f"{weighted_brier:>10.4f} "
                f"{total_supp:>10d}"
            )
            print(f"{sep_single}")
        
        print(f"{sep_double}\n")

    def train(self, eval_fn: Optional[Callable] = None) -> Dict:
        """
        Main training loop.
        
        Returns final metrics dict.
        """
        global_step = 0
        best_metric = float("inf")
        self.optimizer.zero_grad()
        
        print(f"\n{'='*60}")
        print(f"Starting training: Experiment {self.config['experiment']['id']}")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Validation samples: {len(self.val_loader.dataset)}")
        print(f"Epochs: {self.epochs}")
        print(f"Batch size: {self.config['training']['batch_size']} (accumulation: {self.grad_accumulation})")
        print(f"Effective batch size: {self.config['training']['batch_size'] * self.grad_accumulation}")
        print(f"{'='*60}\n")
        
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            
        for epoch in range(self.epochs):
            self.model.train()
            epoch_start = time.time()
            
            accumulated_loss = 0.0
            batch_losses = []
            batch_emo_losses = []
            batch_risk_losses = []
            
            for batch_idx, batch in enumerate(self.train_loader):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                emotion_labels = batch["emotion_labels"].to(self.device)
                emotion_mask = batch["emotion_mask"].to(self.device)
                risk_label = batch["risk_label"].to(self.device)
                risk_mask = batch["risk_mask"].to(self.device)
                
                # Forward pass
                output = self.model(input_ids, attention_mask)
                
                # Compute loss
                loss_dict = self.criterion(
                    emotion_logits=output.emotion_logits,
                    emotion_labels=emotion_labels,
                    emotion_mask=emotion_mask,
                    risk_logits=output.risk_logits,
                    risk_labels=risk_label,
                    risk_mask=risk_mask,
                )
                
                # Dynamic accumulation steps: normalize loss by the actual number of
                # batches in the current accumulation window (handles remainder batches at epoch end).
                batches_left = len(self.train_loader) - (batch_idx // self.grad_accumulation) * self.grad_accumulation
                current_accum_steps = min(self.grad_accumulation, batches_left)
                
                loss = loss_dict["total"] / current_accum_steps
                loss.backward()
                
                accumulated_loss += loss_dict["total"].item()
                batch_losses.append(loss_dict["total"].item())
                batch_emo_losses.append(loss_dict["emotion_loss"].item())
                batch_risk_losses.append(loss_dict["risk_loss"].item())
                
                # Step optimizer on accumulation boundary or on the last batch of the epoch
                is_accum_boundary = (batch_idx + 1) % self.grad_accumulation == 0
                is_last_batch = (batch_idx + 1) == len(self.train_loader)
                if is_accum_boundary or is_last_batch:
                    # Clip gradients
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.max_grad_norm,
                    )
                    
                    # Update learning rate
                    lr = self._get_lr(global_step)
                    for param_group in self.optimizer.param_groups:
                        lr_scale = param_group.get("lr_scale", 1.0)
                        param_group["lr"] = lr * lr_scale

                    
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    global_step += 1
                    
                    if global_step % 50 == 0:
                        avg_loss = accumulated_loss / (batch_idx + 1)
                        elapsed = time.time() - epoch_start
                        print(
                            f"Epoch {epoch+1}/{self.epochs} | "
                            f"Step {global_step} | "
                            f"Loss: {avg_loss:.4f} | "
                            f"Emo: {loss_dict['emotion_loss']:.4f} | "
                            f"Risk: {loss_dict['risk_loss']:.4f} | "
                            f"LR: {lr:.2e} | "
                            f"Time: {elapsed:.1f}s"
                        )
            
            # Ensure no residual gradients carry over into evaluation or the next epoch
            self.optimizer.zero_grad()
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            
            # End of epoch evaluation
            epoch_time = time.time() - epoch_start
            self.train_times.append(epoch_time)
            
            eval_start = time.time()
            val_metrics = self._evaluate()
            eval_time = time.time() - eval_start
            self.eval_times.append(eval_time)
            
            train_loss = sum(batch_losses) / max(len(batch_losses), 1)
            train_emo_loss = sum(batch_emo_losses) / max(len(batch_emo_losses), 1)
            train_risk_loss = sum(batch_risk_losses) / max(len(batch_risk_losses), 1)
            current_lr = self.optimizer.param_groups[0]["lr"]
            
            # Display all metrics and metrics per emotion
            self._print_epoch_summary(
                epoch=epoch + 1,
                train_loss=train_loss,
                train_emo_loss=train_emo_loss,
                train_risk_loss=train_risk_loss,
                val_metrics=val_metrics,
                epoch_time=epoch_time,
                eval_time=eval_time,
                lr=current_lr,
            )
            
            # Check patience controller (underfitting & overfitting detection)
            patience_res = self.patience_controller.step(
                epoch=epoch + 1,
                val_metric=val_metrics["val_loss"],
                train_loss=train_loss,
                model=self.model,
                metrics=val_metrics,
            )

            # Log
            log_entry = {
                "epoch": epoch + 1,
                "global_step": global_step,
                "train_loss": train_loss,
                "train_emotion_loss": train_emo_loss,
                "train_risk_loss": train_risk_loss,
                **val_metrics,
                "epoch_time_s": epoch_time,
                "eval_time_s": eval_time,
                "patience_overfit_counter": self.patience_controller.overfit_counter,
                "patience_underfit_counter": self.patience_controller.underfit_counter,
            }
            self.log_entries.append(log_entry)
            
            # Save best checkpoint
            if patience_res.is_best:
                best_metric = val_metrics["val_loss"]
                checkpoint_path = self.checkpoint_dir / "best_model.pt"
                torch.save({
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "epoch": epoch + 1,
                    "metrics": val_metrics,
                    "config": self.config,
                }, checkpoint_path)
                try:
                    from safetensors.torch import save_file
                    save_file(self.model.state_dict(), self.checkpoint_dir / "best_model.safetensors")
                except Exception:
                    pass
                print(f"  [Checkpoint] Saved new best model to {checkpoint_path} and .safetensors (Val Loss: {best_metric:.4f})\n")
            
            # Save latest
            checkpoint_path = self.checkpoint_dir / "latest_model.pt"
            torch.save({
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "epoch": epoch + 1,
                "metrics": val_metrics,
                "config": self.config,
            }, checkpoint_path)
            try:
                from safetensors.torch import save_file
                save_file(self.model.state_dict(), self.checkpoint_dir / "latest_model.safetensors")
            except Exception:
                pass

            # Early stopping check
            if patience_res.should_stop:
                print(f"\n{'!'*80}")
                print(f" [EARLY STOPPING TRIGGERED at Epoch {epoch + 1}]")
                print(f" Type   : {patience_res.stop_type.upper() if patience_res.stop_type else 'EARLY_STOP'}")
                print(f" Reason : {patience_res.reason}")
                print(f"{'!'*80}\n")
                break
        
        # Restore optimal learned model parameters
        if self.patience_controller.restore_best_weights:
            restored = self.patience_controller.restore_model(self.model)
            if restored and self.patience_controller.best_metrics:
                val_metrics = self.patience_controller.best_metrics

        # Save training log
        log_path = self.log_dir / f"{self.config['experiment']['id']}_training_log.json"
        with open(log_path, "w") as f:
            json.dump(self.log_entries, f, indent=2, default=str)
        
        # Final evaluation with custom eval_fn if provided
        final_metrics = val_metrics
        if eval_fn:
            print("\n--- Final Evaluation (custom) ---")
            final_metrics = eval_fn(self.model, self.device)
        
        return final_metrics
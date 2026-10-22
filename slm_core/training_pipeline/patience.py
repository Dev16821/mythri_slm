"""
Patience and Early Stopping Controller for MYTHRI SLM training.

Monitors training and validation dynamics after each epoch to detect:
1. Overfitting:
   - Validation loss ceases to improve for `patience` consecutive epochs.
   - Validation loss severely diverges from training loss (val_loss / train_loss > divergence_threshold).
2. Underfitting / Stagnation:
   - Training loss fails to decrease over `underfit_patience` epochs (optimization plateau or underfitting).
3. Parameter Preservation:
   - Caches the best model parameters (state_dict) in CPU memory.
   - Restores the optimal learned parameters back into the model when training stops or terminates,
     ensuring downstream inference and evaluation always use the best checkpoint.
"""

from copy import deepcopy
from typing import Dict, Optional, Any
import torch
import torch.nn as nn


class EarlyStoppingResult:
    """Outcome of a patience check after an epoch."""

    def __init__(
        self,
        should_stop: bool = False,
        reason: Optional[str] = None,
        is_best: bool = False,
        stop_type: Optional[str] = None,  # 'overfitting' | 'underfitting' | None
    ):
        self.should_stop = should_stop
        self.reason = reason
        self.is_best = is_best
        self.stop_type = stop_type

    def __repr__(self) -> str:
        return (
            f"EarlyStoppingResult(should_stop={self.should_stop}, "
            f"is_best={self.is_best}, stop_type={self.stop_type}, reason={self.reason})"
        )


class TrainingPatienceController:
    """
    Dual-condition Early Stopping controller with automatic parameter restoration.

    Args:
        patience: Consecutive epochs with no validation improvement before declaring overfitting.
        min_delta: Minimum delta to qualify as an improvement.
        underfit_patience: Consecutive epochs with no train loss reduction before declaring underfitting.
        divergence_threshold: Ratio (val_loss / train_loss) above which severe overfitting is flagged.
        mode: 'min' for loss metrics (default), 'max' for score metrics (e.g. F1).
        restore_best_weights: If True, restores model parameters to the best observed checkpoint.
        verbose: If True, logs patience diagnostics to stdout.
    """

    def __init__(
        self,
        patience: int = 4,
        min_delta: float = 1e-4,
        underfit_patience: int = 5,
        divergence_threshold: float = 3.0,
        mode: str = "min",
        restore_best_weights: bool = True,
        verbose: bool = True,
    ):
        self.patience = max(1, patience)
        self.min_delta = min_delta
        self.underfit_patience = max(1, underfit_patience)
        self.divergence_threshold = divergence_threshold
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        self.verbose = verbose

        self.overfit_counter = 0
        self.underfit_counter = 0

        self.best_score: Optional[float] = None
        self.best_train_loss: Optional[float] = None
        self.best_epoch: int = 0
        self.best_state_dict: Optional[Dict[str, torch.Tensor]] = None
        self.best_metrics: Dict[str, Any] = {}

        self.stopped_epoch: int = 0
        self.stop_reason: Optional[str] = None
        self.stop_type: Optional[str] = None

    def step(
        self,
        epoch: int,
        val_metric: float,
        train_loss: float,
        model: nn.Module,
        metrics: Optional[Dict[str, Any]] = None,
    ) -> EarlyStoppingResult:
        """
        Evaluate epoch results against overfitting and underfitting criteria.

        Args:
            epoch: Current epoch index (1-indexed).
            val_metric: Validation metric value (e.g. val_loss).
            train_loss: Current epoch training loss.
            model: Active PyTorch model.
            metrics: Optional full validation metrics dictionary.

        Returns:
            EarlyStoppingResult indicating whether training should halt and if this is a new best epoch.
        """
        score = -val_metric if self.mode == "min" else val_metric
        is_best = False

        # 1. First epoch: initialize baselines and store initial best weights
        if self.best_score is None:
            self.best_score = score
            self.best_train_loss = train_loss
            self.best_epoch = epoch
            self.best_metrics = deepcopy(metrics) if metrics else {}
            if self.restore_best_weights:
                self.best_state_dict = {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                }
            is_best = True
            if self.verbose:
                print(f"  [Patience] Initial baseline set at Epoch {epoch}: Val Metric = {val_metric:.4f}")
            return EarlyStoppingResult(should_stop=False, is_best=True)

        # 2. Check for Validation Improvement (Overfitting tracker)
        improved = (score - self.best_score) > self.min_delta

        if improved:
            improvement = abs(score - self.best_score)
            self.best_score = score
            self.best_epoch = epoch
            self.overfit_counter = 0
            self.best_metrics = deepcopy(metrics) if metrics else {}
            if self.restore_best_weights:
                self.best_state_dict = {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                }
            is_best = True
            if self.verbose:
                best_val = -self.best_score if self.mode == "min" else self.best_score
                print(
                    f"  [Patience] Val metric improved by {improvement:.4f} "
                    f"(Best: {best_val:.4f}) at Epoch {epoch}. Overfit counter reset to 0/{self.patience}."
                )
        else:
            self.overfit_counter += 1
            if self.verbose:
                best_val = -self.best_score if self.mode == "min" else self.best_score
                print(
                    f"  [Patience] No validation improvement for {self.overfit_counter}/{self.patience} epochs "
                    f"(Best: {best_val:.4f} at Epoch {self.best_epoch})."
                )

        # 3. Check for Training Loss Stagnation (Underfitting tracker)
        if self.best_train_loss is not None:
            train_improved = (self.best_train_loss - train_loss) > self.min_delta
            if train_improved:
                self.best_train_loss = train_loss
                self.underfit_counter = 0
            else:
                self.underfit_counter += 1
                if self.verbose and self.underfit_counter >= 2:
                    print(
                        f"  [Patience] Training loss stagnation warning: "
                        f"No significant decrease for {self.underfit_counter}/{self.underfit_patience} epochs "
                        f"(Train Loss: {train_loss:.4f})."
                    )

        # 4. Check Overfitting Early Stopping Condition
        # A) Standard patience threshold reached
        if self.overfit_counter >= self.patience:
            self.stopped_epoch = epoch
            self.stop_type = "overfitting"
            best_val = -self.best_score if self.mode == "min" else self.best_score
            self.stop_reason = (
                f"OVERFITTING DETECTED: Validation loss failed to improve for {self.patience} consecutive epochs. "
                f"(Optimal checkpoint was Epoch {self.best_epoch} with val_loss={best_val:.4f})."
            )
            return EarlyStoppingResult(
                should_stop=True,
                reason=self.stop_reason,
                is_best=is_best,
                stop_type="overfitting",
            )

        # B) Divergence trigger: val_loss worsening while train_loss is dropping steeply
        if (
            train_loss > 0
            and (val_metric / train_loss) >= self.divergence_threshold
            and self.overfit_counter >= 2
        ):
            self.stopped_epoch = epoch
            self.stop_type = "overfitting"
            self.stop_reason = (
                f"OVERFITTING DIVERGENCE: Validation loss ({val_metric:.4f}) is {val_metric / train_loss:.2f}x "
                f"higher than training loss ({train_loss:.4f}) with deteriorating generalization."
            )
            return EarlyStoppingResult(
                should_stop=True,
                reason=self.stop_reason,
                is_best=is_best,
                stop_type="overfitting",
            )

        # 5. Check Underfitting / Stagnation Early Stopping Condition
        if self.underfit_counter >= self.underfit_patience:
            self.stopped_epoch = epoch
            self.stop_type = "underfitting"
            self.stop_reason = (
                f"UNDERFITTING / STAGNATION DETECTED: Training loss failed to decrease for "
                f"{self.underfit_patience} consecutive epochs (stuck around train_loss={train_loss:.4f})."
            )
            return EarlyStoppingResult(
                should_stop=True,
                reason=self.stop_reason,
                is_best=is_best,
                stop_type="underfitting",
            )

        return EarlyStoppingResult(should_stop=False, is_best=is_best)

    def restore_model(self, model: nn.Module) -> bool:
        """
        Restores the cached optimal parameters into the model.

        Args:
            model: PyTorch model to update with the best learned parameters.

        Returns:
            True if weights were successfully restored, False otherwise.
        """
        if self.best_state_dict is not None:
            # Place restored tensors onto the model's current device
            device = next(model.parameters()).device
            restored_state = {k: v.to(device) for k, v in self.best_state_dict.items()}
            model.load_state_dict(restored_state)
            if self.verbose:
                best_val = -self.best_score if self.mode == "min" else self.best_score
                print(
                    f"\n  [Patience] Updated and restored model parameters to best checkpoint "
                    f"from Epoch {self.best_epoch} (Val Metric: {best_val:.4f})."
                )
            return True
        return False

    def get_summary(self) -> Dict[str, Any]:
        """Returns diagnostic metrics about the patience / early stopping state."""
        best_val = -self.best_score if (self.best_score is not None and self.mode == "min") else self.best_score
        return {
            "stopped_early": self.stopped_epoch > 0,
            "stopped_epoch": self.stopped_epoch,
            "stop_type": self.stop_type,
            "stop_reason": self.stop_reason,
            "best_epoch": self.best_epoch,
            "best_metric": best_val,
            "overfit_counter": self.overfit_counter,
            "underfit_counter": self.underfit_counter,
            "best_metrics": self.best_metrics,
        }


# Convenience alias
EarlyStopping = TrainingPatienceController
Patience = TrainingPatienceController

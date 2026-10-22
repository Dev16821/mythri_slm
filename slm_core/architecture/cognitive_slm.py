"""
CognitiveSLM-v0: The unified cognitive layer for MYTHRI.

Combines a shared transformer encoder with multi-task heads
for emotion and risk prediction. Also provides:

- MC Dropout-based uncertainty estimation
- Latent representation output (for future LLM control experiments)
- Abstention mechanism (refuse prediction when too uncertain)
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, List, Any, Union
from dataclasses import dataclass

from .transformer_encoder import TransformerEncoder, PretrainedDistilBertEncoder
from .multi_task_heads import EmotionHead, RiskHead, TopicBoundaryHead, TurnFunctionHead


@dataclass
class CognitiveOutput:
    """Structured output from the cognitive SLM."""
    emotion_logits: torch.Tensor          # (B, num_emotions)
    emotion_probs: torch.Tensor           # (B, num_emotions)
    emotion_intensity: torch.Tensor       # (B, 1)
    risk_logits: torch.Tensor             # (B, num_risk_classes)
    risk_probs: torch.Tensor               # (B, num_risk_classes)
    risk_prediction: torch.Tensor         # (B,)
    risk_probability: torch.Tensor         # (B, 1) — P(risk)
    latent_representation: torch.Tensor    # (B, hidden_dim) — pooled encoder output
    uncertainty: Optional[torch.Tensor] = None  # (B,) — MC dropout variance
    abstain: Optional[torch.Tensor] = None      # (B,) — boolean
    topic_decision: Optional[Dict[str, Any]] = None  # Topic continuity / boundary decision
    turn_function_info: Optional[Dict[str, Any]] = None  # Turn speech-act & question intent



class CognitiveSLM(nn.Module):
    """
    The unified cognitive SLM.
    
    Architecture:
    
        Input IDs
            ↓
        TransformerEncoder (shared backbone)
            ↓
        Pooled Representation (CLS + Mean)
            ↓
        ┌───────────┬───────────────┐
        ↓                           ↓
    EmotionHead                  RiskHead
    (multi-label)              (multi-class)
    
    The pooled representation is also exposed as a "latent cognitive state"
    that can be used for LLM conditioning in future experiments.
    """

    def __init__(
        self,
        vocab_size: int = 30522,
        hidden_dim: Optional[int] = None,
        num_layers: int = 6,
        num_heads: int = 8,
        ffn_dim: int = 2048,
        max_seq_len: int = 128,
        dropout: float = 0.1,
        num_emotions: int = 35,
        num_risk_classes: int = 3,
        emotion_head_hidden: int = 512,
        risk_head_hidden: int = 256,
        initializer_range: float = 0.02,
        pretrained_backbone: Optional[str] = None,
        freeze_backbone: bool = False,
        pretrained_embeddings: Optional[str] = None,
        boundary_head_trained: bool = False,
        turn_function_head_trained: bool = False,
    ):
        super().__init__()
        self.initializer_range = initializer_range
        self.pretrained_backbone = pretrained_backbone
        self.has_pretrained_backbone = pretrained_backbone is not None

        if self.has_pretrained_backbone:
            # Pretrained backbone (e.g., distilbert-base-uncased)
            target_proj_dim = hidden_dim if (hidden_dim is not None and hidden_dim != 768) else None
            self.encoder = PretrainedDistilBertEncoder(
                model_name=pretrained_backbone,
                dropout=dropout,
                freeze=freeze_backbone,
                target_dim=target_proj_dim,
            )
            self.hidden_dim = self.encoder.hidden_dim
            pooled_dim = self.encoder.pooled_dim
        else:
            # From-scratch compact transformer encoder
            self.hidden_dim = hidden_dim if hidden_dim is not None else 512
            self.encoder = TransformerEncoder(
                vocab_size=vocab_size,
                hidden_dim=self.hidden_dim,
                num_layers=num_layers,
                num_heads=num_heads,
                ffn_dim=ffn_dim,
                max_seq_len=max_seq_len,
                dropout=dropout,
                initializer_range=initializer_range,
                pretrained_embeddings=pretrained_embeddings,
            )
            pooled_dim = 2 * self.hidden_dim

        # Task heads
        self.emotion_head = EmotionHead(
            input_dim=pooled_dim,
            hidden_dim=emotion_head_hidden,
            num_emotions=num_emotions,
            dropout=dropout,
            initializer_range=initializer_range,
        )

        self.risk_head = RiskHead(
            input_dim=pooled_dim,
            hidden_dim=risk_head_hidden,
            num_classes=num_risk_classes,
            dropout=dropout,
            initializer_range=initializer_range,
            num_emotions=num_emotions,
        )

        self.boundary_head = TopicBoundaryHead(
            latent_dim=pooled_dim,
            dropout=dropout,
            initializer_range=initializer_range,
            is_trained=boundary_head_trained,
        )

        self.turn_function_head = TurnFunctionHead(
            input_dim=pooled_dim,
            dropout=dropout,
            initializer_range=initializer_range,
            is_trained=turn_function_head_trained,
        )

        # Store latent dim for external use
        self.latent_dim = pooled_dim

        # Initialization strategy:
        if not self.has_pretrained_backbone:
            self.apply(self._init_weights)
            if hasattr(self.encoder, "_init_residual_projections"):
                self.encoder._init_residual_projections()
            if pretrained_embeddings is not None and hasattr(self.encoder, "load_pretrained_embeddings"):
                self.encoder.load_pretrained_embeddings(pretrained_embeddings)
        else:
            # For pretrained backbone: ONLY initialize task heads (never overwrite pretrained weights!)
            self.emotion_head.apply(self._init_weights)
            self.risk_head.apply(self._init_weights)
            self.boundary_head.apply(self._init_weights)
            self.turn_function_head.apply(self._init_weights)
            if hasattr(self.encoder, "proj") and self.encoder.proj is not None:
                self.encoder.proj.apply(self._init_weights)

        # Affective projection starts at zero weight for backward compatibility with legacy checkpoints
        if hasattr(self.risk_head, "affective_proj") and self.risk_head.affective_proj is not None:
            nn.init.zeros_(self.risk_head.affective_proj.weight)

    def _init_weights(self, module: nn.Module):
        """Unified initialization strategy across all modules and heads."""
        if getattr(module, "_is_zero_init", False):
            if hasattr(module, "weight") and module.weight is not None:
                nn.init.zeros_(module.weight)
            return
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            if module.weight is not None:
                nn.init.ones_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def evaluate_boundary(
        self,
        current_latent: torch.Tensor,
        previous_latent: torch.Tensor,
        threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Evaluate topic boundary transition between two consecutive latent vectors."""
        return self.boundary_head(current_latent, previous_latent, threshold=threshold)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        previous_latent: Optional[torch.Tensor] = None,
        text: Optional[Union[str, List[str]]] = None,
    ) -> CognitiveOutput:
        """
        input_ids: (B, S)
        attention_mask: (B, S) — 1 for valid tokens, 0 for padding
        previous_latent: (B, latent_dim) or (latent_dim,) optional prior turn latent state
        text: Optional raw text or list of texts for syntax-verified turn progression
        
        Returns CognitiveOutput with all task predictions.
        """
        # Shared encoder forward pass
        latent = self.encoder(input_ids, attention_mask)  # (B, 2*hidden_dim)

        # Task-specific heads
        emotion_out = self.emotion_head(latent)
        risk_out = self.risk_head(
            latent,
            emotion_probs=emotion_out["probabilities"],
            emotion_intensity=emotion_out["intensity"],
        )
        turn_fn_out = self.turn_function_head(latent, text=text)

        # Optional topic boundary evaluation
        topic_decision = None
        if previous_latent is not None:
            topic_decision = self.boundary_head(latent, previous_latent)

        return CognitiveOutput(
            emotion_logits=emotion_out["logits"],
            emotion_probs=emotion_out["probabilities"],
            emotion_intensity=emotion_out["intensity"],
            risk_logits=risk_out["logits"],
            risk_probs=risk_out["probabilities"],
            risk_prediction=risk_out["prediction"],
            risk_probability=risk_out["risk_probability"],
            latent_representation=latent,
            topic_decision=topic_decision,
            turn_function_info=turn_fn_out,
        )



    @torch.no_grad()
    def predict_with_uncertainty(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        previous_latent: Optional[torch.Tensor] = None,
        num_samples: int = 10,
        uncertainty_threshold: float = 0.15,
        head_only: bool = True,
        text: Optional[Union[str, List[str]]] = None,
    ) -> CognitiveOutput:
        """
        Monte Carlo Dropout inference.
        
        Performs vectorized parallel MC-dropout passes without sequential iteration.
        When head_only=True (default for real-time inference), evaluates the encoder once
        and executes parallel MC-dropout passes through the task heads, eliminating redundant
        re-encoding passes while computing epistemic uncertainty.
        
        If uncertainty exceeds threshold, sets abstain=True for that sample.
        """
        batch_size = input_ids.shape[0]

        if head_only:
            # Deterministic eval pass through encoder & dialogue heads
            self.eval()
            latent = self.encoder(input_ids, attention_mask)
            turn_fn_out = self.turn_function_head(latent, text=text)
            topic_decision = None
            if previous_latent is not None:
                topic_decision = self.boundary_head(latent, previous_latent)

            # Parallel MC-dropout across multi-task heads
            self.emotion_head.train()
            self.risk_head.train()
            rep_latent = latent.repeat_interleave(num_samples, dim=0)
            emo_out = self.emotion_head(rep_latent)
            risk_out = self.risk_head(
                rep_latent,
                emotion_probs=emo_out["probabilities"],
                emotion_intensity=emo_out["intensity"],
            )
            self.emotion_head.eval()
            self.risk_head.eval()

            # Reshape: (num_samples, B, num_classes)
            emotion_probs_stack = emo_out["probabilities"].view(batch_size, num_samples, -1).transpose(0, 1)
            risk_probs_stack = risk_out["probabilities"].view(batch_size, num_samples, -1).transpose(0, 1)
        else:
            # Batched full-model MC-dropout (runs all samples in parallel instead of sequentially)
            self.train()
            rep_input_ids = input_ids.repeat_interleave(num_samples, dim=0)
            rep_attention_mask = (
                attention_mask.repeat_interleave(num_samples, dim=0)
                if attention_mask is not None else None
            )
            latent_samples = self.encoder(rep_input_ids, rep_attention_mask)
            emo_out = self.emotion_head(latent_samples)
            risk_out = self.risk_head(
                latent_samples,
                emotion_probs=emo_out["probabilities"],
                emotion_intensity=emo_out["intensity"],
            )

            emotion_probs_stack = emo_out["probabilities"].view(batch_size, num_samples, -1).transpose(0, 1)
            risk_probs_stack = risk_out["probabilities"].view(batch_size, num_samples, -1).transpose(0, 1)

            self.eval()
            latent = self.encoder(input_ids, attention_mask)
            turn_fn_out = self.turn_function_head(latent, text=text)
            topic_decision = None
            if previous_latent is not None:
                topic_decision = self.boundary_head(latent, previous_latent)

        mean_emotion_probs = emotion_probs_stack.mean(dim=0)
        mean_risk_probs = risk_probs_stack.mean(dim=0)


        # Uncertainty: assess candidate uncertainty without diluting across inactive emotions
        # In multi-label emotion classification, 95% of classes are inactive for any given input.
        # Averaging all 35 classes artificially divides variance by 35, preventing abstention.
        # We compute emotion uncertainty from the top candidate emotions (max / top-k variance).
        emotion_var_per_class = emotion_probs_stack.var(dim=0)  # (B, num_emotions)
        emotion_var_max = emotion_var_per_class.max(dim=-1).values  # (B,)
        topk_k = min(3, emotion_var_per_class.shape[-1])
        emotion_var_topk = torch.topk(emotion_var_per_class, k=topk_k, dim=-1).values.mean(dim=-1)  # (B,)
        emotion_var = torch.maximum(emotion_var_max, emotion_var_topk)  # (B,)

        # Risk variance (categorical distribution over 3 risk states)
        risk_var_per_class = risk_probs_stack.var(dim=0)  # (B, num_risk_classes)
        risk_var = risk_var_per_class.max(dim=-1).values  # (B,)

        # Combined epistemic uncertainty reflects uncertainty if either emotion or risk is ambiguous
        uncertainty = torch.maximum(emotion_var, risk_var)  # (B,)

        # Abstention: evaluated on both direct variance and standard deviation scale
        # (Standard deviation reflects probability spread in [0, 1]; handles thresholds configured
        # in either variance scale e.g. 0.005 or std-dev spread scale e.g. 0.05 - 0.15)
        uncertainty_std = torch.sqrt(uncertainty)
        abstain = (uncertainty > uncertainty_threshold) | (uncertainty_std > uncertainty_threshold)


        # Final prediction from mean
        risk_prediction = mean_risk_probs.argmax(dim=-1)
        risk_probability = mean_risk_probs[:, -1:]
        emotion_intensity = mean_emotion_probs.max(dim=-1, keepdim=True).values

        # For multi-label sigmoid probabilities, logit is log(p / (1 - p)), NOT log(p)
        clamped_emotion_probs = mean_emotion_probs.clamp(min=1e-7, max=1.0 - 1e-7)
        emotion_logits = torch.log(clamped_emotion_probs / (1.0 - clamped_emotion_probs))

        return CognitiveOutput(
            emotion_logits=emotion_logits,
            emotion_probs=mean_emotion_probs,
            emotion_intensity=emotion_intensity,
            risk_logits=torch.log(mean_risk_probs.clamp(min=1e-8)),
            risk_probs=mean_risk_probs,
            risk_prediction=risk_prediction,
            risk_probability=risk_probability,
            latent_representation=latent,
            uncertainty=uncertainty,
            abstain=abstain,
            topic_decision=topic_decision,
            turn_function_info=turn_fn_out,
        )

    @classmethod
    def from_pretrained_backbone(
        cls,
        model_name: str = "distilbert-base-uncased",
        hidden_dim: Optional[int] = 768,
        num_emotions: int = 35,
        num_risk_classes: int = 3,
        emotion_head_hidden: int = 512,
        risk_head_hidden: int = 256,
        dropout: float = 0.1,
        freeze_backbone: bool = False,
        **kwargs,
    ) -> "CognitiveSLM":
        """
        Factory method to instantiate a CognitiveSLM with a pretrained backbone
        (e.g., DistilBERT), fine-tuning on affective and risk assessment tasks.
        """
        return cls(
            pretrained_backbone=model_name,
            hidden_dim=hidden_dim,
            num_emotions=num_emotions,
            num_risk_classes=num_risk_classes,
            emotion_head_hidden=emotion_head_hidden,
            risk_head_hidden=risk_head_hidden,
            dropout=dropout,
            freeze_backbone=freeze_backbone,
            **kwargs,
        )

    def get_num_parameters(self) -> Dict[str, Any]:
        """Return parameter counts broken down by component."""
        encoder_params = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
        emotion_params = sum(p.numel() for p in self.emotion_head.parameters() if p.requires_grad)
        risk_params = sum(p.numel() for p in self.risk_head.parameters() if p.requires_grad)
        boundary_params = sum(p.numel() for p in self.boundary_head.parameters() if p.requires_grad)
        turn_fn_params = sum(p.numel() for p in self.turn_function_head.parameters() if p.requires_grad)
        
        trainable_total = encoder_params + emotion_params + risk_params
        if getattr(self.boundary_head, "is_trained", False):
            trainable_total += boundary_params
        if getattr(self.turn_function_head, "is_trained", False):
            trainable_total += turn_fn_params

        total_all = sum(p.numel() for p in self.parameters())
        return {
            "encoder": encoder_params,
            "emotion_head": emotion_params,
            "risk_head": risk_params,
            "boundary_head": boundary_params,
            "turn_function_head": turn_fn_params,
            "trainable": trainable_total,
            "total": total_all,
            "boundary_head_trained": getattr(self.boundary_head, "is_trained", False),
            "turn_function_head_trained": getattr(self.turn_function_head, "is_trained", False),
        }
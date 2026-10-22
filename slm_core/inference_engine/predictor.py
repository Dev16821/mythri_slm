"""
Inference wrapper for the trained cognitive SLM.

Provides a clean interface for the rest of the MYTHRI pipeline
to consume the SLM's outputs.
"""

import re
import torch
import torch.nn as nn
from typing import Dict, Optional, List, Any, Union
from pathlib import Path

from model.cognitive_slm import CognitiveSLM, CognitiveOutput
from model.conversation_state import ConversationStateEngine, ConversationState
from data_utils.tokenizer import MythriTokenizer
from data_utils.synthetic_data import get_emotion_label_list


class CognitivePredictor:
    """
    Wraps the trained CognitiveSLM for inference.
    
    This is the interface that the MYTHRI pipeline would use:
    
        predictor = CognitivePredictor.load("checkpoints/best_model.pt")
        result = predictor.predict("I feel so alone today")
        
        result.emotions -> {"loneliness": 0.82, "sadness": 0.45, ...}
        result.risk_level -> "distressed"
        result.risk_probability -> 0.12
        result.uncertainty -> 0.08
        result.latent -> [512 floats]  # for future LLM control
    """

    def __init__(
        self,
        model: CognitiveSLM,
        tokenizer: MythriTokenizer,
        emotion_labels: List[str],
        device: torch.device = torch.device("cpu"),
        uncertainty_threshold: float = 0.15,
        emotion_threshold: float = 0.5,
        mc_samples: int = 10,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.emotion_labels = emotion_labels
        self.device = device
        self.uncertainty_threshold = uncertainty_threshold
        self.emotion_threshold = emotion_threshold
        self.mc_samples = mc_samples
        self.last_latent: Optional[torch.Tensor] = None
        self.state_engine = ConversationStateEngine()
        
        self.model.to(device)
        self.model.eval()
        self._warmup()

    def _warmup(self):
        """Warm up PyTorch/device execution kernels to prevent first-call latency."""
        try:
            with torch.no_grad():
                dummy_ids = torch.zeros((1, 8), dtype=torch.long, device=self.device)
                dummy_mask = torch.ones((1, 8), dtype=torch.long, device=self.device)
                _ = self.model.predict_with_uncertainty(
                    dummy_ids,
                    dummy_mask,
                    num_samples=min(self.mc_samples, 10),
                )
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
        except Exception:
            pass

    def reset_session(self):
        """Reset conversation session state."""
        self.last_latent = None
        self.state_engine.reset()


    @classmethod
    def load(
        cls,
        checkpoint_path: str,
        device: torch.device = torch.device("cpu"),
        uncertainty_threshold: Optional[float] = None,
        emotion_threshold: Optional[float] = None,
        mc_samples: Optional[int] = None,
    ) -> "CognitivePredictor":
        """Load a trained model from checkpoint (.pt or .safetensors)."""
        ckpt_p = Path(checkpoint_path)
        if ckpt_p.suffix == ".safetensors":
            from safetensors.torch import load_file
            state_dict = load_file(str(ckpt_p))
            
            # Retrieve configuration
            companion_pt = ckpt_p.with_suffix(".pt")
            companion_yaml = Path(__file__).resolve().parent.parent / "configs" / "experiment_0.yaml"
            if companion_pt.exists():
                pt_ckpt = torch.load(companion_pt, map_location="cpu", weights_only=False)
                config = pt_ckpt.get("config", {})
            elif companion_yaml.exists():
                import yaml
                with open(companion_yaml, "r") as f:
                    config = yaml.safe_load(f)
            else:
                from experiments.experiment_0 import CONFIG
                config = CONFIG
        else:
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                config = checkpoint.get("config", {})
                state_dict = checkpoint["model_state_dict"]
            elif isinstance(checkpoint, dict):
                config = {}
                state_dict = checkpoint
            else:
                raise ValueError("Unsupported checkpoint format")

            if not config:
                companion_yaml = Path(__file__).resolve().parent.parent / "configs" / "experiment_0.yaml"
                if companion_yaml.exists():
                    import yaml
                    with open(companion_yaml, "r") as f:
                        config = yaml.safe_load(f)
                else:
                    from experiments.experiment_0 import CONFIG
                    config = CONFIG
        
        model_config = config["model"]
        model = CognitiveSLM(
            vocab_size=model_config.get("vocab_size", 30522),
            hidden_dim=model_config.get("hidden_dim", 512),
            num_layers=model_config.get("num_layers", 6),
            num_heads=model_config.get("num_heads", 8),
            ffn_dim=model_config.get("ffn_dim", 2048),
            max_seq_len=model_config.get("max_seq_len", 128),
            dropout=model_config.get("dropout", 0.1),
            num_emotions=model_config.get("num_emotions", 35),
            num_risk_classes=model_config.get("num_risk_classes", 3),
            emotion_head_hidden=model_config.get("emotion_head_hidden", 512),
            risk_head_hidden=model_config.get("risk_head_hidden", 256),
            initializer_range=model_config.get("initializer_range", 0.02),
            pretrained_backbone=model_config.get("pretrained_backbone", None),
            freeze_backbone=model_config.get("freeze_backbone", False),
            pretrained_embeddings=model_config.get("pretrained_embeddings", None),
        )


        # Load weights gracefully (handles new heads like TopicBoundaryHead)
        model.load_state_dict(state_dict, strict=False)
        
        tokenizer = MythriTokenizer(config["tokenizer"]["type"])
        emotion_labels = config["data"]["emotion_labels"]
        
        eval_config = config.get("evaluation", {})
        thresh = uncertainty_threshold if uncertainty_threshold is not None else eval_config.get("uncertainty_threshold", 0.05)
        # Calibrate legacy unnormalized 0.15 threshold to 0.05 so abstention reliably activates on ambiguity
        if uncertainty_threshold is None and thresh == 0.15:
            thresh = 0.05

        return cls(
            model=model,
            tokenizer=tokenizer,
            emotion_labels=emotion_labels,
            device=device,
            uncertainty_threshold=thresh,
            emotion_threshold=emotion_threshold if emotion_threshold is not None else eval_config.get("emotion_threshold", 0.5),
            mc_samples=mc_samples if mc_samples is not None else eval_config.get("mc_dropout_samples", 10),
        )


    def predict(
        self,
        text: str,
        use_uncertainty: bool = True,
        previous_latent: Optional[Any] = None,
        track_session: bool = True,
    ) -> Dict:
        """
        Run inference on a single text input.
        
        Returns a structured dict with:
            emotions: Dict[str, float] — emotion -> probability
            active_emotions: List[str] — emotions above threshold
            emotion_intensity: float
            risk_state: str — "normal" | "distressed" | "risk"
            risk_probability: float
            uncertainty: float (if use_uncertainty)
            abstain: bool (if use_uncertainty)
            latent: List[float] — latent representation for LLM control
            topic_decision: Optional[Dict] — topic continuity / boundary decision if prior turn exists
        """
        encoded = self.tokenizer.encode(text, max_length=128, truncation=True, padding=True)
        input_ids = encoded["input_ids"].unsqueeze(0).to(self.device)
        attention_mask = encoded["attention_mask"].unsqueeze(0).to(self.device)

        # Resolve previous turn latent vector for topic boundary detection
        prev_tensor = None
        if previous_latent is not None:
            if isinstance(previous_latent, list):
                prev_tensor = torch.tensor(previous_latent, dtype=torch.float32, device=self.device)
            elif isinstance(previous_latent, torch.Tensor):
                prev_tensor = previous_latent.to(self.device)
        elif self.last_latent is not None:
            prev_tensor = self.last_latent.to(self.device)
        
        if use_uncertainty:
            output = self.model.predict_with_uncertainty(
                input_ids,
                attention_mask,
                previous_latent=prev_tensor,
                num_samples=self.mc_samples,
                uncertainty_threshold=self.uncertainty_threshold,
                text=text,
            )
            uncertainty = output.uncertainty.item() if output.uncertainty is not None else 0.0
            abstain = output.abstain.item() if output.abstain is not None else False
        else:
            with torch.no_grad():
                output = self.model(input_ids, attention_mask, previous_latent=prev_tensor, text=text)
            uncertainty = 0.0
            abstain = False

        # Session tracking
        if track_session:
            self.last_latent = output.latent_representation.detach()
        
        # Emotion predictions
        emotion_probs_list = output.emotion_probs.squeeze(0).detach().cpu().tolist()
        emotions = dict(zip(self.emotion_labels, emotion_probs_list))
        active_emotions = [
            label for label, prob in emotions.items()
            if prob >= self.emotion_threshold
        ]
        
        # Risk predictions
        risk_probs_list = output.risk_probs.squeeze(0).detach().cpu().tolist()
        risk_states = ["normal", "distressed", "risk"]
        pred_idx = int(output.risk_prediction.squeeze().item())
        risk_state = risk_states[pred_idx]
        risk_probability = risk_probs_list[-1]

        # Check if an affirmative crisis phrase is matched
        from safety.deterministic_policy import (
            DeterministicSafetyPolicy,
            DEFAULT_PATTERNS,
            LIFE_AFFIRMATION_CUES,
            NEGATION_CUES,
        )
        has_affirmative_crisis = DeterministicSafetyPolicy.has_affirmative_crisis_match(text, DEFAULT_PATTERNS)

        if not has_affirmative_crisis:
            # Positive affect safeguard:
            # If the utterance expresses genuine positive affect without distress indicators or danger cues,
            # calibrate spurious elevated risk predictions down to normal state
            positive_emotions = ["joy", "happiness", "gratitude", "love"]
            max_pos_prob = max([emotions.get(e, 0.0) for e in positive_emotions] + [0.0])
            distress_emotions = ["sadness", "grief", "loneliness", "anxiety", "fear", "remorse", "guilt"]
            max_distress_prob = max([emotions.get(e, 0.0) for e in distress_emotions] + [0.0])
            pos_regex = re.compile(
                r"\b(?:so happy|very happy|really happy|full of joy|so joyful|so grateful|wonderful day|"
                r"best day|great day|love (?:this|life|my)|feeling blessed|excited for|so excited|thrilled)\b",
                re.IGNORECASE
            )
            is_linguistic_pos = bool(pos_regex.search(text))

            if (max_pos_prob >= 0.40 or is_linguistic_pos) and max_distress_prob < 0.20:
                if risk_state == "risk":
                    risk_state = "normal"
                    risk_probability = min(risk_probability, 0.05)
                    risk_probs_list[0] = max(risk_probs_list[0], 0.95)
                    risk_probs_list[1] = min(risk_probs_list[1], 0.03)
                    risk_probs_list[2] = risk_probability

            # Clausal negation & life affirmation safeguard:
            # If the utterance expresses explicit negation of a crisis cue (e.g. "I do not want to die, I want to live"),
            # calibrate spurious elevated neural risk predictions down to normal state
            is_negated_crisis = False
            for p in DEFAULT_PATTERNS:
                esc_p = re.escape(p)
                if "don\\'t" in esc_p or "dont" in esc_p:
                    esc_p = esc_p.replace("don\\'t", r"(?:don'?t|do\s+not)").replace("dont", r"(?:don'?t|do\s+not)")
                for m in re.finditer(r"\b" + esc_p + r"\b", text, re.IGNORECASE):
                    if DeterministicSafetyPolicy.is_negated_match(text, m.start(), m.end(), m.group()):
                        is_negated_crisis = True
                        break
                if is_negated_crisis:
                    break

            if not is_negated_crisis and LIFE_AFFIRMATION_CUES.search(text) and NEGATION_CUES.search(text):
                is_negated_crisis = True

            if is_negated_crisis and max_distress_prob < 0.35:
                if risk_state == "risk":
                    risk_state = "normal"
                    risk_probability = min(risk_probability, 0.05)
                    risk_probs_list[0] = max(risk_probs_list[0], 0.95)
                    risk_probs_list[1] = min(risk_probs_list[1], 0.03)
                    risk_probs_list[2] = risk_probability

        # Latent representation
        latent = output.latent_representation.squeeze(0).detach().cpu().tolist()

        # Update conversation state & progression
        turn_fn_pred = None
        has_q = False
        if output.turn_function_info is not None:
            turn_fn_pred = output.turn_function_info.get("turn_function")
            has_q = output.turn_function_info.get("has_unresolved_question", False)

        conv_state = self.state_engine.update(
            text=text,
            emotions=emotions,
            risk_state=risk_state,
            topic_decision=output.topic_decision,
            turn_function_pred=turn_fn_pred,
            neural_has_question=has_q,
        )

        return {
            "text": text,
            "emotions": emotions,
            "active_emotions": active_emotions,
            "emotion_intensity": float(output.emotion_intensity.squeeze().item()),
            "risk_state": risk_state,
            "risk_probability": risk_probability,
            "risk_probs": {
                "normal": risk_probs_list[0],
                "distressed": risk_probs_list[1],
                "risk": risk_probs_list[2],
            },
            "uncertainty": uncertainty,
            "abstain": abstain,
            "latent": latent,
            "topic_decision": output.topic_decision,
            "conversation_state": conv_state.to_dict(),
            "turn_function_info": output.turn_function_info,
        }



    def predict_batch(self, texts: List[str]) -> List[Dict]:
        """Run inference on a batch of texts."""
        return [self.predict(text) for text in texts]
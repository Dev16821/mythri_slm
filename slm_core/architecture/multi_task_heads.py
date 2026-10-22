"""
Task-specific prediction heads for the cognitive SLM.

Each head takes the pooled encoder output and produces task-specific predictions.
Heads are intentionally simple MLPs — the heavy lifting is in the shared encoder.
"""

import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Any, List, Union, Tuple


class EmotionHead(nn.Module):
    """
    Multi-label emotion classification head.
    
    Outputs sigmoid probabilities for each of N emotion classes.
    Multiple emotions can be active simultaneously.
    
    Also produces an intensity estimate (max probability among active emotions).
    """

    def __init__(self, input_dim: int, hidden_dim: int, num_emotions: int, dropout: float = 0.1, initializer_range: float = 0.02):
        super().__init__()
        self.num_emotions = num_emotions
        self.initializer_range = initializer_range
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_emotions)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
        self._init_weights()

    def _init_weights(self):
        """Consistent Gaussian initialization with zero biases."""
        for linear in [self.fc1, self.fc2]:
            nn.init.normal_(linear.weight, mean=0.0, std=self.initializer_range)
            if linear.bias is not None:
                nn.init.zeros_(linear.bias)

    def forward(self, pooled: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        pooled: (batch, input_dim) — concatenated CLS + mean pooling

        Returns:
            logits: (batch, num_emotions)
            probabilities: (batch, num_emotions)
            intensity: (batch, 1) — max probability
        """
        x = self.dropout(pooled)
        x = self.activation(self.fc1(x))
        x = self.dropout(x)
        logits = self.fc2(x)
        probabilities = torch.sigmoid(logits)
        intensity = probabilities.max(dim=-1, keepdim=True).values
        return {
            "logits": logits,
            "probabilities": probabilities,
            "intensity": intensity,
        }


class RiskHead(nn.Module):
    """
    Multi-class risk classification head with affective conditioning.
    
    Classes:
        0: normal       — no safety concern
        1: distressed   — emotional distress, no explicit risk
        2: risk         — potential safety concern (self-harm, crisis, danger)
        
    Affective Conditioning:
    Directly incorporates predicted emotion probabilities and emotional intensity
    into risk feature representations:
        emotion -> emotional intensity -> possible risk
    Enables unified cognitive reasoning across affective state and risk evaluation.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_classes: int = 3,
        dropout: float = 0.1,
        initializer_range: float = 0.02,
        num_emotions: int = 35,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.initializer_range = initializer_range
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        # Affective projection: maps [emotion_probs (35), emotion_intensity (1)] to hidden_dim
        # Zero-initialized so existing checkpoints load and function seamlessly
        self.affective_proj = nn.Linear(num_emotions + 1, hidden_dim, bias=False)
        self.affective_proj._is_zero_init = True
        nn.init.zeros_(self.affective_proj.weight)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
        self._init_weights()

    def _init_weights(self):
        """Consistent Gaussian initialization with zero biases."""
        for linear in [self.fc1, self.fc2]:
            nn.init.normal_(linear.weight, mean=0.0, std=self.initializer_range)
            if linear.bias is not None:
                nn.init.zeros_(linear.bias)
        if hasattr(self, "affective_proj") and self.affective_proj is not None:
            nn.init.zeros_(self.affective_proj.weight)

    def forward(
        self,
        pooled: torch.Tensor,
        emotion_probs: Optional[torch.Tensor] = None,
        emotion_intensity: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        pooled: (batch, input_dim)
        emotion_probs: Optional (batch, num_emotions)
        emotion_intensity: Optional (batch, 1)

        Returns:
            logits: (batch, num_classes)
            probabilities: (batch, num_classes)
            prediction: (batch,) — argmax class
            risk_probability: (batch, 1) — P(class=risk)
        """
        x = self.dropout(pooled)
        feat = self.activation(self.fc1(x))
        if emotion_probs is not None and self.affective_proj is not None:
            if emotion_intensity is None:
                emotion_intensity = emotion_probs.max(dim=-1, keepdim=True).values
            affect_in = torch.cat([emotion_probs, emotion_intensity], dim=-1)
            feat = feat + self.affective_proj(affect_in)
        feat = self.dropout(feat)
        logits = self.fc2(feat)
        probabilities = F.softmax(logits, dim=-1)
        prediction = logits.argmax(dim=-1)
        risk_probability = probabilities[:, -1:]  # P(risk) = last class
        return {
            "logits": logits,
            "probabilities": probabilities,
            "prediction": prediction,
            "risk_probability": risk_probability,
        }



class TopicBoundaryHead(nn.Module):
    """
    Topic boundary and conversational continuity head.

    Evaluates whether the user's current utterance continues the active conversational
    topic or initiates a sharp topic pivot (e.g. from 'cooking pasta' to 'relationship distress').

    Takes latent representations of consecutive turns:
        h_prev: (B, latent_dim)
        h_curr: (B, latent_dim)

    Computes:
        1. Geometric cosine similarity and calibrated continuity score.
        2. Neural interaction projection (when trained):
           v_trans = [h_prev, h_curr, |h_curr - h_prev|, h_curr * h_prev] (B, 4 * latent_dim)
           When trained, fuses neural continuity with geometric continuity.
           When untrained (default for baseline SLM), safely bypasses the untrained ~1.05M
           parameters to prevent wasteful matrix multiplications and ungrounded predictions.
    """

    def __init__(
        self,
        latent_dim: int = 1024,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        boundary_threshold: float = 0.50,
        initializer_range: float = 0.02,
        is_trained: bool = False,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.boundary_threshold = boundary_threshold
        self.initializer_range = initializer_range
        self.is_trained = is_trained

        # Feature interaction dimension: [h_prev, h_curr, diff, prod]
        input_dim = latent_dim * 4
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
        self._init_weights()

    def set_trained(self, trained: bool = True):
        """Set whether the neural weights have been trained."""
        self.is_trained = trained

    def _init_weights(self):
        """Consistent Gaussian initialization with zero biases."""
        for linear in [self.fc1, self.fc2]:
            nn.init.normal_(linear.weight, mean=0.0, std=self.initializer_range)
            if linear.bias is not None:
                nn.init.zeros_(linear.bias)

    def forward(
        self,
        h_curr: torch.Tensor,
        h_prev: torch.Tensor,
        threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Args:
            h_curr: (B, latent_dim) or (latent_dim,)
            h_prev: (B, latent_dim) or (latent_dim,)
            threshold: Optional override for boundary decision threshold.

        Returns:
            Dict containing:
                topic_continuity: float in [0.0, 1.0]
                topic_boundary: bool (True if topic shifted)
                cosine_similarity: float in [-1.0, 1.0]
                semantic_distance: float in [0.0, 2.0]
                neural_continuity: Optional[float] (None if untrained)
                is_trained: bool
                recommended_memory_action: "FLUSH_RAG" | "PRUNE_HISTORY" | "RETAIN_CONTEXT"
                reason: str explaining the decision
        """
        thresh = threshold if threshold is not None else self.boundary_threshold

        # Ensure 2D (B, D)
        if h_curr.dim() == 1:
            h_curr = h_curr.unsqueeze(0)
        if h_prev.dim() == 1:
            h_prev = h_prev.unsqueeze(0)

        # 1. Geometric cosine similarity
        cos_sim = F.cosine_similarity(h_curr, h_prev, dim=-1)  # (B,)
        sem_dist = (1.0 - cos_sim).clamp(min=0.0)

        # In high-dimensional transformer latent space (d=1024), unrelated sentences have
        # an empirical baseline cosine similarity around 0.35-0.40 due to syntactic token overlap.
        # We calibrate the continuity metric over the active semantic range [0.38, 0.90]:
        #   cos >= 0.85 -> high continuity (same topic)
        #   cos 0.55 - 0.84 -> evolving/related topic
        #   cos < 0.55 -> sharp topic boundary shift
        cos_continuity = ((cos_sim - 0.38) / (0.90 - 0.38)).clamp(min=0.0, max=1.0)

        # 2. Neural interaction projection (only executed when trained)
        if self.is_trained:
            diff = torch.abs(h_curr - h_prev)
            prod = h_curr * h_prev
            v_trans = torch.cat([h_prev, h_curr, diff, prod], dim=-1)  # (B, 4*latent_dim)

            x = self.dropout(v_trans)
            x = self.activation(self.fc1(x))
            x = self.dropout(x)
            logits = self.fc2(x).squeeze(-1)  # (B,)
            neural_continuity = torch.sigmoid(logits)  # (B,)
            
            # Calibrated blend: fuse neural prediction with geometric cosine continuity
            continuity = 0.5 * neural_continuity + 0.5 * cos_continuity
        else:
            # Safely bypass untrained ~1.05M parameters: eliminate wasted matrix multiplications
            # and prevent random initialization noise from distorting continuity.
            neural_continuity = None
            continuity = cos_continuity

        is_boundary = continuity < thresh

        # Determine recommended RAG & memory actions
        actions = []
        reasons = []
        for i in range(len(continuity)):
            c_val = float(continuity[i].item())
            if c_val < 0.40:
                actions.append("FLUSH_RAG")
                reasons.append(
                    f"Sharp topic shift detected (continuity={c_val:.2%}). "
                    "Evict previous RAG knowledge to prevent cross-topic contamination."
                )
            elif c_val < 0.65:
                actions.append("PRUNE_HISTORY")
                reasons.append(
                    f"Topic is evolving/pivoting (continuity={c_val:.2%}). "
                    "Decay previous RAG relevance and archive older working context."
                )
            else:
                actions.append("RETAIN_CONTEXT")
                reasons.append(
                    f"Continuous topic thread maintained (continuity={c_val:.2%}). "
                    "Keep existing RAG context and conversational working memory."
                )

        if len(continuity) == 1:
            return {
                "topic_continuity": float(continuity[0].item()),
                "topic_boundary": bool(is_boundary[0].item()),
                "cosine_similarity": float(cos_sim[0].item()),
                "semantic_distance": float(sem_dist[0].item()),
                "neural_continuity": float(neural_continuity[0].item()) if neural_continuity is not None else None,
                "is_trained": self.is_trained,
                "recommended_memory_action": actions[0],
                "reason": reasons[0],
            }

        return {
            "topic_continuity": continuity,
            "topic_boundary": is_boundary,
            "cosine_similarity": cos_sim,
            "semantic_distance": sem_dist,
            "neural_continuity": neural_continuity,
            "is_trained": self.is_trained,
            "recommended_memory_action": actions,
            "reason": reasons,
        }


class TurnFunctionHead(nn.Module):
    """
    Pragmatic turn function classification head.

    Predicts the communicative speech act / pragmatic function of the utterance:
    0: introducing_problem — User presents a new difficulty, symptom, or emotional pain
    1: elaborating_venting — User deepens the narrative, gives details, or vents emotions
    2: seeking_advice      — User asks what to do, how to cope, or requests guidance
    3: evaluating_advice   — User reacts to proposed advice (agrees, pushes back, reflects)
    4: casual_chat         — Chit-chat, pleasantries, greetings, factual questions
    5: closing_grounding   — De-escalation, gratitude, wrapping up, feeling grounded

    Question Detection:
    Features robust syntactic and linguistic question detection.
    When untrained (is_trained=False), bypasses the untrained random projection (~264K params)
    to guarantee that purely declarative statements (e.g. 'I feel so lonely and depressed today')
    are NEVER falsely flagged as unresolved questions.
    """

    TURN_FUNCTIONS = [
        "introducing_problem",
        "elaborating_venting",
        "seeking_advice",
        "evaluating_advice",
        "casual_chat",
        "closing_grounding",
    ]

    ADVICE_CUES = [
        r"\bwhat (should|can|do|must)\b.*?\b(do|say|try)\b",
        r"\bhow (do|can|should)\b.*?\b(handle|deal with|stop|fix|manage|respond|cope)\b",
        r"\bany advice\b",
        r"\bhelp me (with|understand|figure out)\b",
        r"\bwhat do you suggest\b",
        r"\bwhat would you do\b",
        r"\bcan you help me\b",
        r"\bwhat to do\b",
        r"\bgive me (some |any )?advice\b",
        r"\bneed (some |any )?advice\b",
    ]

    EVAL_CUES = [
        r"\bi (already )?tried\b",
        r"\bthat (won't|does not|didn't) work\b",
        r"\bthat sounds.*?(hard|scary|difficult|impossible)\b",
        r"\bthat makes sense\b",
        r"\bmaybe that could work\b",
        r"\bi guess i could try\b",
        r"\bbut what if\b",
        r"\bmade (him|her|them) (angrier|upset|mad)\b",
    ]

    CLOSURE_CUES = [
        r"\bthank you\b",
        r"\bthanks for (listening|helping|talking)\b",
        r"\bi feel (a bit|much|a little) better\b",
        r"\bi will try (that|this)\b",
        r"\btalk to you later\b",
        r"\bgoodbye|bye\b",
        r"\bgood night\b",
    ]

    INTERROGATIVE_CUES = [
        r"^\s*(what|why|how|when|where|who|whom|which|whose)\b",
        r"^\s*(can|could|would|should|will|shall|may|might|do|does|did|is|are|am|was|were|have|has|had)\s+(i|you|we|he|she|it|they|there|someone|anyone|anybody)\b",
    ]

    def __init__(
        self,
        input_dim: int = 1024,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        initializer_range: float = 0.02,
        is_trained: bool = False,
    ):
        super().__init__()
        self.num_classes = len(self.TURN_FUNCTIONS)
        self.initializer_range = initializer_range
        self.is_trained = is_trained
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, self.num_classes)
        self.fc_question = nn.Linear(hidden_dim, 1)  # Unresolved question detector
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
        self._init_weights()

    def set_trained(self, trained: bool = True):
        """Set whether the neural weights have been trained."""
        self.is_trained = trained

    def _init_weights(self):
        """Consistent Gaussian initialization with zero biases."""
        for linear in [self.fc1, self.fc2, self.fc_question]:
            nn.init.normal_(linear.weight, mean=0.0, std=self.initializer_range)
            if linear.bias is not None:
                nn.init.zeros_(linear.bias)

    @classmethod
    def detect_question_linguistic(cls, text: str) -> Dict[str, Any]:
        """
        Linguistically verified question and advice detection.
        Examines punctuation, interrogative syntax, inverted modals, and advice cues.
        Guarantees that declarative statements like 'I feel so lonely and depressed today'
        evaluate to has_question=False.
        """
        if not text or not str(text).strip():
            return {
                "has_question": False,
                "question_probability": 0.0,
                "question_text": None,
            }

        text_str = str(text).strip()
        text_lower = text_str.lower()

        # 1. Direct question mark
        if "?" in text_str:
            sentences = re.split(r'[.!?\n]+', text_str)
            q_part = None
            for s in sentences:
                s_strip = s.strip()
                if s_strip and any(re.search(p, s_strip.lower()) for p in cls.ADVICE_CUES + cls.INTERROGATIVE_CUES):
                    q_part = s_strip
                    break
            if not q_part:
                match = re.search(r'([^.!?\n]+)\?', text_str)
                q_part = match.group(1).strip() if match else text_str

            return {
                "has_question": True,
                "question_probability": 1.0,
                "question_text": q_part,
            }

        # 2. Advice seeking cues
        for cue in cls.ADVICE_CUES:
            if re.search(cue, text_lower):
                sentences = re.split(r'[.!;\n]+', text_str)
                q_part = None
                for s in sentences:
                    if re.search(cue, s.lower()):
                        q_part = s.strip()
                        break
                return {
                    "has_question": True,
                    "question_probability": 0.95,
                    "question_text": q_part or text_str,
                }

        # 3. Interrogative clause starters (wh-words or modal/aux inversion at clause start)
        sentences = re.split(r'[.!;\n]+', text_str)
        for s in sentences:
            s_clean = s.strip()
            s_clean_lower = s_clean.lower()
            for cue in cls.INTERROGATIVE_CUES:
                if re.search(cue, s_clean_lower):
                    return {
                        "has_question": True,
                        "question_probability": 0.90,
                        "question_text": s_clean,
                    }

        # 4. Purely declarative statement
        return {
            "has_question": False,
            "question_probability": 0.0,
            "question_text": None,
        }

    @classmethod
    def determine_turn_function_heuristic(
        cls,
        text: str,
        emotions: Optional[Dict[str, float]] = None,
        risk_state: Optional[str] = None,
    ) -> str:
        text_lower = text.lower().strip() if text else ""
        if any(re.search(p, text_lower) for p in cls.CLOSURE_CUES):
            return "closing_grounding"
        if any(re.search(p, text_lower) for p in cls.EVAL_CUES):
            return "evaluating_advice"

        q_info = cls.detect_question_linguistic(text)
        if q_info["has_question"]:
            return "seeking_advice"

        if emotions:
            elevated = (
                emotions.get("sadness", 0.0) > 0.20
                or emotions.get("anxiety", 0.0) > 0.20
                or emotions.get("anger", 0.0) > 0.20
                or emotions.get("grief", 0.0) > 0.20
                or emotions.get("loneliness", 0.0) > 0.20
            )
            if elevated or (risk_state and risk_state != "normal"):
                return "introducing_problem"

        return "casual_chat"

    def forward(
        self,
        pooled: torch.Tensor,
        text: Optional[Union[str, List[str]]] = None,
        emotions: Optional[Union[Dict[str, float], List[Dict[str, float]]]] = None,
        risk_state: Optional[Union[str, List[str]]] = None,
    ) -> Dict[str, Any]:
        """
        pooled: (B, input_dim) — h_cog
        text: Optional string or list of strings corresponding to batch.
        """
        if pooled.dim() == 1:
            pooled = pooled.unsqueeze(0)
        B = pooled.shape[0]

        text_list: List[Optional[str]] = []
        if isinstance(text, str):
            text_list = [text] * B
        elif isinstance(text, (list, tuple)):
            text_list = list(text)
            if len(text_list) < B:
                text_list.extend([None] * (B - len(text_list)))
        else:
            text_list = [None] * B

        if self.is_trained:
            x = self.dropout(pooled)
            feat = self.activation(self.fc1(x))
            feat = self.dropout(feat)

            logits = self.fc2(feat)
            probs = F.softmax(logits, dim=-1)
            pred_idx = logits.argmax(dim=-1)

            q_logits = self.fc_question(feat).squeeze(-1)
            q_prob = torch.sigmoid(q_logits)

            has_questions = []
            final_q_probs = []
            q_texts = []
            for i in range(B):
                t = text_list[i]
                qp = float(q_prob[i].item()) if B > 1 else float(q_prob.item())
                if t is not None:
                    ling = self.detect_question_linguistic(t)
                    if not ling["has_question"] and qp < 0.85:
                        has_questions.append(False)
                        final_q_probs.append(0.0)
                        q_texts.append(None)
                    else:
                        has_questions.append(ling["has_question"] or (qp > 0.50))
                        final_q_probs.append(max(ling["question_probability"], qp))
                        q_texts.append(ling["question_text"])
                else:
                    has_questions.append(qp > 0.50)
                    final_q_probs.append(qp)
                    q_texts.append(None)

            has_question_out = torch.tensor(has_questions, device=pooled.device)
            q_prob_out = torch.tensor(final_q_probs, device=pooled.device)
            pred_functions = [self.TURN_FUNCTIONS[i] for i in pred_idx.tolist()]
        else:
            # Bypass untrained ~264K weights to eliminate random inference & noise
            pred_functions = []
            has_questions = []
            final_q_probs = []
            q_texts = []
            probs_list = []

            for i in range(B):
                t = text_list[i] or ""
                emo_i = emotions if isinstance(emotions, dict) else (emotions[i] if isinstance(emotions, list) and i < len(emotions) else None)
                risk_i = risk_state if isinstance(risk_state, str) else (risk_state[i] if isinstance(risk_state, list) and i < len(risk_state) else None)

                fn = self.determine_turn_function_heuristic(t, emo_i, risk_i)
                pred_functions.append(fn)

                ling = self.detect_question_linguistic(t)
                has_questions.append(ling["has_question"])
                final_q_probs.append(ling["question_probability"])
                q_texts.append(ling["question_text"])

                fn_idx = self.TURN_FUNCTIONS.index(fn)
                prob_vec = [0.03] * self.num_classes
                prob_vec[fn_idx] = 1.0 - (0.03 * (self.num_classes - 1))
                probs_list.append(prob_vec)

            has_question_out = torch.tensor(has_questions, device=pooled.device)
            q_prob_out = torch.tensor(final_q_probs, device=pooled.device)
            probs = torch.tensor(probs_list, device=pooled.device)

        if B == 1:
            return {
                "turn_function": pred_functions[0],
                "turn_function_probs": dict(zip(self.TURN_FUNCTIONS, probs[0].tolist())),
                "has_unresolved_question": bool(has_question_out[0].item()),
                "question_probability": float(q_prob_out[0].item()),
                "unresolved_question_text": q_texts[0],
                "is_trained": self.is_trained,
            }

        return {
            "turn_function": pred_functions,
            "turn_function_probs": probs,
            "has_unresolved_question": has_question_out,
            "question_probability": q_prob_out,
            "unresolved_question_text": q_texts,
            "is_trained": self.is_trained,
        }
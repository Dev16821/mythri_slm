"""
Combined dataset for multi-task training of the cognitive SLM.

Loads emotion and risk examples from JSONL files in the datasets/ directory.
Implements joint multi-task supervision:
- Emotion examples receive inferred risk labels (emotion -> intensity -> risk).
- Risk examples receive inferred affective emotion labels.
- Both emotion_mask and risk_mask provide active gradients for joint learning.
"""

import json
import random
import re
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Optional, Set
from pathlib import Path

from data_utils.tokenizer import MythriTokenizer
from data_utils.synthetic_data import get_emotion_label_list, EMOTION_TO_IDX
from safety.deterministic_policy import DeterministicSafetyPolicy


def augment_text(text: str, rng: random.Random, prob: float = 0.15) -> str:
    """
    On-the-fly text augmentation during training to prevent memorization:
    - Random word dropout: randomly masks or drops 5-8% of words
    - Contraction variations
    - Casing jitter
    """
    if rng.random() > prob:
        return text

    words = text.split()
    if len(words) <= 3:
        return text

    # Word dropout / substitution
    augmented_words = []
    for w in words:
        r = rng.random()
        if r < 0.05:
            continue  # drop word
        elif r < 0.08:
            augmented_words.append("[MASK]")  # mask word
        else:
            augmented_words.append(w)

    if not augmented_words:
        augmented_words = words

    augmented_text = " ".join(augmented_words)

    # Occasional case jitter
    if rng.random() < 0.05:
        augmented_text = augmented_text.lower()

    return augmented_text


# High-distress emotion categories that indicate risk class 1 (distressed)
HIGH_DISTRESS_EMOTIONS = {
    "sadness", "grief", "loneliness", "anxiety", "fear", "remorse", "guilt", "nervousness"
}
HIGH_DISTRESS_INDICES = {
    EMOTION_TO_IDX[e] for e in HIGH_DISTRESS_EMOTIONS if e in EMOTION_TO_IDX
}

# Explicit crisis cue phrases for detecting risk class 2 (risk / crisis)
CRISIS_CUES = [
    re.compile(r"\b" + re.escape(p) + r"\b", re.IGNORECASE) for p in [
        "kill myself", "killing myself", "take my own life", "end my life",
        "end it all", "commit suicide", "want to die", "suicide note",
        "hang myself", "slit my wrists", "overdose", "jump off the bridge",
        "better off dead", "final goodbye", "goodbye letter", "cut myself deeply"
    ]
]


def infer_risk_from_emotion_data(text: str, emotion_labels: List[int]) -> int:
    """
    Infers the ground-truth risk category for an emotion-annotated utterance.
    Establishes cognitive joint supervision: emotion -> emotional intensity -> risk.
    
    Classes:
        0: normal (everyday, positive, mild, neutral)
        1: distressed (acute emotional distress: sadness, grief, anxiety, loneliness)
        2: risk (explicit crisis/safety concern)
    """
    # 1. Check for explicit crisis phrases (with clausal negation understanding)
    for pattern in CRISIS_CUES:
        for match in pattern.finditer(text):
            if not DeterministicSafetyPolicy.is_negated_match(text, match.start(), match.end(), match.group()):
                return 2  # class 2: risk / crisis

    # 2. Check for emotional distress indicators
    if any(idx in HIGH_DISTRESS_INDICES for idx in emotion_labels):
        return 1  # class 1: distressed

    return 0  # class 0: normal


def infer_emotions_from_risk_data(text: str, risk_label: int, risk_type: str = "n/a") -> List[int]:
    """
    Infers affective emotion labels for risk-annotated utterances.
    Provides multi-task emotion supervision on risk corpus:
    - Normal / hard-negative queries -> true negative emotion anchors (all zeros)
    - Distressed disclosures -> anxiety, nervousness, grief, loneliness, guilt, sadness
    - Crisis utterances -> severe distress (grief, sadness, loneliness)
    """
    if risk_label == 0:
        return []

    text_lower = text.lower()
    emotions: Set[int] = set()

    # Anxiety & panic cues
    if any(w in text_lower for w in ["anxi", "panic", "worry", "racing thoughts", "chest feels tight", "suffocating"]):
        if "anxiety" in EMOTION_TO_IDX:
            emotions.add(EMOTION_TO_IDX["anxiety"])
        if any(w in text_lower for w in ["nervous", "panic", "racing"]):
            if "nervousness" in EMOTION_TO_IDX:
                emotions.add(EMOTION_TO_IDX["nervousness"])

    # Grief & loss cues
    if any(w in text_lower for w in ["grief", "mourning", "loss", "carrying so much grief", "miss them", "dying"]):
        if "grief" in EMOTION_TO_IDX:
            emotions.add(EMOTION_TO_IDX["grief"])

    # Loneliness & isolation cues
    if any(w in text_lower for w in ["alone", "lonel", "isolated", "disconnected", "nobody will miss", "ghost", "no one", "disappear"]):
        if "loneliness" in EMOTION_TO_IDX:
            emotions.add(EMOTION_TO_IDX["loneliness"])

    # Guilt, remorse & feeling like a burden/failure
    if any(w in text_lower for w in ["failure", "guilt", "my fault", "regret", "burden", "deserve"]):
        if "guilt" in EMOTION_TO_IDX:
            emotions.add(EMOTION_TO_IDX["guilt"])
        if "disappointment" in EMOTION_TO_IDX:
            emotions.add(EMOTION_TO_IDX["disappointment"])

    # Fear cues
    if any(w in text_lower for w in ["scared", "terrified", "afraid", "fear"]):
        if "fear" in EMOTION_TO_IDX:
            emotions.add(EMOTION_TO_IDX["fear"])

    # Burnout & frustration
    if any(w in text_lower for w in ["burnout", "frustrat", "exhausting", "draining", "never enough"]):
        if "frustration" in EMOTION_TO_IDX:
            emotions.add(EMOTION_TO_IDX["frustration"])

    # Crisis states (class 2) always anchor in grief and sadness
    if risk_label == 2:
        if "sadness" in EMOTION_TO_IDX:
            emotions.add(EMOTION_TO_IDX["sadness"])
        if "grief" in EMOTION_TO_IDX:
            emotions.add(EMOTION_TO_IDX["grief"])
        if any(w in text_lower for w in ["nobody", "alone", "no one", "disappear", "burden", "fade away"]):
            if "loneliness" in EMOTION_TO_IDX:
                emotions.add(EMOTION_TO_IDX["loneliness"])

    # Distressed states (class 1) involve sadness as foundational affective state
    if risk_label == 1:
        if "sadness" in EMOTION_TO_IDX:
            emotions.add(EMOTION_TO_IDX["sadness"])

    return sorted(list(emotions))


class CombinedCognitiveDataset(Dataset):
    """
    Combines emotion and risk examples from JSONL files into a single dataset.
    Supports on-the-fly data augmentation during training to prevent overfitting.
    Supports joint multi-task supervision: emotion -> emotional intensity -> possible risk.
    
    JSONL format expected:
        Emotion: {"text": "...", "labels": [0, 5], "source": "synthetic"}
        Risk:    {"text": "...", "label": 2, "risk_type": "implicit", "source": "synthetic"}
    """

    def __init__(
        self,
        emotion_filepath: str,
        risk_filepath: str,
        tokenizer: MythriTokenizer,
        max_length: int = 128,
        num_emotions: int = 35,
        augment: bool = False,
        augmentation_prob: float = 0.20,
        joint_supervision: bool = True,
        seed: int = 42,
        max_samples: Optional[int] = None,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_emotions = num_emotions
        self.augment = augment
        self.augmentation_prob = augmentation_prob
        self.joint_supervision = joint_supervision
        self.rng = random.Random(seed)
        self.examples: List[Dict] = []
        
        # Load emotion data
        if emotion_filepath and Path(emotion_filepath).exists():
            with open(emotion_filepath, "r", encoding="utf-8") as f:
                for line in f:
                    ex = json.loads(line)
                    text = ex["text"]
                    emo_labels = ex.get("labels", [])
                    if self.joint_supervision:
                        r_label = infer_risk_from_emotion_data(text, emo_labels)
                    else:
                        r_label = -1
                    self.examples.append({
                        "text": text,
                        "task": "emotion",
                        "emotion_labels": emo_labels,
                        "risk_label": r_label,
                        "risk_type": "inferred" if self.joint_supervision else "n/a",
                    })
        
        # Load risk data
        if risk_filepath and Path(risk_filepath).exists():
            with open(risk_filepath, "r", encoding="utf-8") as f:
                for line in f:
                    ex = json.loads(line)
                    text = ex["text"]
                    r_label = ex["label"]
                    r_type = ex.get("risk_type", "n/a")
                    if self.joint_supervision:
                        emo_labels = infer_emotions_from_risk_data(text, r_label, r_type)
                    else:
                        emo_labels = []
                    self.examples.append({
                        "text": text,
                        "task": "risk",
                        "emotion_labels": emo_labels,
                        "risk_label": r_label,
                        "risk_type": r_type,
                    })

        # Apply stratified sub-sampling if max_samples is specified and smaller than total
        if max_samples is not None and len(self.examples) > max_samples:
            rare_emotion_indices = {
                EMOTION_TO_IDX[e] for e in [
                    "grief", "nervousness", "remorse", "desire", "caring",
                    "relief", "pride", "optimism"
                ] if e in EMOTION_TO_IDX
            }
            mandatory = []
            standard = []
            for ex in self.examples:
                is_rare = any(idx in rare_emotion_indices for idx in ex.get("emotion_labels", []))
                is_risk = ex.get("risk_label", -1) >= 1
                if is_rare or is_risk:
                    mandatory.append(ex)
                else:
                    standard.append(ex)

            if len(mandatory) >= max_samples:
                self.rng.shuffle(mandatory)
                selected = mandatory[:max_samples]
            else:
                needed = max_samples - len(mandatory)
                self.rng.shuffle(standard)
                selected = mandatory + standard[:needed]

            self.rng.shuffle(selected)
            print(f"  Subsampled dataset: {len(self.examples):,} -> {len(selected):,} (stratified preservation of critical minority/risk examples)")
            self.examples = selected

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict:
        ex = self.examples[idx]
        text = ex["text"]
        
        # Apply dynamic text augmentation during training to prevent memorization
        if self.augment:
            text = augment_text(text, self.rng, prob=self.augmentation_prob)
        
        encoded = self.tokenizer.encode(
            text,
            max_length=self.max_length,
            truncation=True,
            padding=True,
        )
        
        # Multi-hot emotion label and mask
        if self.joint_supervision:
            # Active joint supervision across 100% of samples
            emotion_label = torch.zeros(self.num_emotions, dtype=torch.float32)
            if ex["emotion_labels"]:
                for label_idx in ex["emotion_labels"]:
                    if 0 <= label_idx < self.num_emotions:
                        emotion_label[label_idx] = 1.0
            emotion_mask = 1.0

            risk_label = ex["risk_label"]
            risk_mask = 1.0 if risk_label >= 0 else 0.0
        else:
            # Legacy single-task masking
            if ex["task"] == "emotion":
                emotion_label = torch.zeros(self.num_emotions, dtype=torch.float32)
                if ex["emotion_labels"]:
                    for label_idx in ex["emotion_labels"]:
                        if 0 <= label_idx < self.num_emotions:
                            emotion_label[label_idx] = 1.0
                emotion_mask = 1.0
            elif ex["task"] == "risk" and ex["risk_type"] in ("normal", "hard_negative", "safe_dangerous_topic"):
                emotion_label = torch.zeros(self.num_emotions, dtype=torch.float32)
                emotion_mask = 1.0
            else:
                emotion_label = torch.full((self.num_emotions,), -1.0, dtype=torch.float32)
                emotion_mask = 0.0

            risk_label = ex["risk_label"]
            risk_mask = 1.0 if (ex["task"] == "risk" and risk_label >= 0) else 0.0
        
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "emotion_labels": emotion_label,
            "emotion_mask": torch.tensor(emotion_mask, dtype=torch.float32),
            "risk_label": torch.tensor(risk_label, dtype=torch.long),
            "risk_mask": torch.tensor(risk_mask, dtype=torch.float32),
            "risk_type": ex["risk_type"],
            "task": ex["task"],
            "text": text,
        }


def collate_fn(batch: List[Dict]) -> Dict:
    """Custom collate function for the combined dataset."""
    input_ids = torch.stack([b["input_ids"] for b in batch])
    attention_mask = torch.stack([b["attention_mask"] for b in batch])
    emotion_labels = torch.stack([b["emotion_labels"] for b in batch])
    emotion_mask = torch.stack([b["emotion_mask"] for b in batch])
    risk_label = torch.stack([b["risk_label"] for b in batch])
    risk_mask = torch.stack([b["risk_mask"] for b in batch])
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "emotion_labels": emotion_labels,
        "emotion_mask": emotion_mask,
        "risk_label": risk_label,
        "risk_mask": risk_mask,
        "risk_types": [b["risk_type"] for b in batch],
        "tasks": [b["task"] for b in batch],
        "texts": [b["text"] for b in batch],
    }
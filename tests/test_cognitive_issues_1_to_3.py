"""
Unit and integration tests verifying the fixes for Issues 1, 2, and 3:
- Issue 1: False Crisis Detection on positive emotions prevented
- Issue 2: Risk dataset expanded beyond 88 repetitive templates to diverse root templates
- Issue 3: Stratified sampling ensures all 35 emotion classes are actively represented
"""

import sys
import json
from pathlib import Path
from collections import Counter

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_utils.synthetic_data import (
    RISK_TEMPLATES,
    _get_partitioned_templates,
    EMOTION_LABELS,
)
from safety.deterministic_policy import HybridSafetyLayer, DeterministicSafetyPolicy
from inference.predictor import CognitivePredictor


def test_issue_1_false_crisis_detection_prevented():
    """
    Issue 1: False Crisis Detection.
    Normal positive emotional disclosures like 'I'm so happy today!' must never be
    classified as a crisis.
    """
    safety = HybridSafetyLayer(
        DeterministicSafetyPolicy(
            patterns=["want to die", "kill myself", "end my life"],
            high_risk_emotions=[],
        )
    )

    # Positive emotion test case
    positive_text = "I'm so happy today!"
    mock_emotions = {"joy": 0.45, "happiness": 0.40, "sadness": 0.01, "grief": 0.00}
    
    # Even if SLM raw risk probability were elevated, safety layer must veto false crisis on positive affect
    decision = safety.evaluate(
        text=positive_text,
        slm_risk_probability=0.72,
        slm_uncertainty=0.01,
        emotion_probs=mock_emotions,
    )

    assert not decision.crisis_detected, "Positive emotional disclosure must not be flagged as crisis"
    assert decision.risk_level == "normal", "Risk level should be normal for joyful disclosure"
    assert decision.recommended_action == "normal", "Action should be normal"
    assert decision.vetoed, "Positive affect veto should be triggered"

    # Affirmative crisis test case must still trigger
    crisis_text = "I want to die tonight."
    crisis_decision = safety.evaluate(
        text=crisis_text,
        slm_risk_probability=0.85,
        slm_uncertainty=0.01,
        emotion_probs={"sadness": 0.80},
    )
    assert crisis_decision.crisis_detected, "Affirmative crisis phrase must trigger crisis"


def test_issue_2_risk_dataset_expanded_beyond_88_templates():
    """
    Issue 2: Training Data Is Too Small and Repetitive.
    Verifies that RISK_TEMPLATES has been expanded far beyond 88 templates
    and includes positive emotional statements and disjoint partitions.
    """
    total_templates = sum(len(templates) for templates in RISK_TEMPLATES.values())
    assert total_templates >= 300, f"Expected at least 300 templates across categories, found {total_templates}"

    # Verify normal category includes positive emotions
    normal_templates = RISK_TEMPLATES["normal"]
    assert any("happy" in t.lower() for t in normal_templates), "Normal templates must include happy disclosures"
    assert any("grateful" in t.lower() for t in normal_templates), "Normal templates must include grateful disclosures"
    assert any("joy" in t.lower() for t in normal_templates), "Normal templates must include joy disclosures"

    # Verify split partitions are disjoint
    train_templates = set(_get_partitioned_templates("normal", split="train"))
    val_templates = set(_get_partitioned_templates("normal", split="val"))
    test_templates = set(_get_partitioned_templates("normal", split="test"))

    assert len(train_templates.intersection(val_templates)) == 0, "Train and Val templates must be disjoint"
    assert len(train_templates.intersection(test_templates)) == 0, "Train and Test templates must be disjoint"
    assert len(val_templates.intersection(test_templates)) == 0, "Val and Test templates must be disjoint"


def test_issue_3_all_35_emotion_classes_actively_represented():
    """
    Issue 3: Most Emotion Classes Are Not Learned (25/35 zero F1).
    Verifies that stratified sampling guarantees active training and validation support
    for ALL 35 emotion classes without missing classes.
    """
    train_file = Path("datasets/raw/emotion_train.jsonl")
    val_file = Path("datasets/raw/emotion_val.jsonl")

    assert train_file.exists(), "emotion_train.jsonl must exist"
    assert val_file.exists(), "emotion_val.jsonl must exist"

    train_counts = Counter()
    with open(train_file, "r") as f:
        for line in f:
            for l in json.loads(line).get("labels", []):
                train_counts[l] += 1

    val_counts = Counter()
    with open(val_file, "r") as f:
        for line in f:
            for l in json.loads(line).get("labels", []):
                val_counts[l] += 1

    # Verify 100% of the 35 classes have positive examples in train
    for idx in range(len(EMOTION_LABELS)):
        c_name = EMOTION_LABELS[idx]
        assert train_counts[idx] >= 100, f"Class '{c_name}' (idx {idx}) has only {train_counts[idx]} training samples (must be >= 100)"
        assert val_counts[idx] >= 20, f"Class '{c_name}' (idx {idx}) has only {val_counts[idx]} validation samples (must be >= 20)"


if __name__ == "__main__":
    test_issue_1_false_crisis_detection_prevented()
    print("✓ test_issue_1_false_crisis_detection_prevented passed")
    test_issue_2_risk_dataset_expanded_beyond_88_templates()
    print("✓ test_issue_2_risk_dataset_expanded_beyond_88_templates passed")
    test_issue_3_all_35_emotion_classes_actively_represented()
    print("✓ test_issue_3_all_35_emotion_classes_actively_represented passed")
    print("\nAll Issue 1, 2, and 3 tests passed successfully!")

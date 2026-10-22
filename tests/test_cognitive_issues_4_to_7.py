"""
Unit and integration tests verifying the fixes for Issues 4, 5, 6, and 7:
- Issue 4: Joint emotion and risk supervision & affective conditioning
- Issue 5: Untrained components bypassed (linguistic question detector)
- Issue 6: Neural boundary head fused when trained, bypassed when untrained
- Issue 7: Clausal negation understanding in safety policy
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from dataset.combined_dataset import (
    CombinedCognitiveDataset,
    infer_risk_from_emotion_data,
    infer_emotions_from_risk_data,
)
from data_utils.synthetic_data import EMOTION_TO_IDX
from model.multi_task_heads import RiskHead, TopicBoundaryHead, TurnFunctionHead
from model.cognitive_slm import CognitiveSLM
from safety.deterministic_policy import DeterministicSafetyPolicy


def test_issue_7_negation_understanding():
    """
    Issue 7: Negation is not understood.
    Statements like 'I do not want to die, I want to live' should be recognized as safe,
    while affirmative crisis expressions must trigger alerts.
    """
    policy = DeterministicSafetyPolicy(
        patterns=["want to die", "end my life", "kill myself", "dont want to live"],
        high_risk_emotions=[],
    )
    
    # Negated life-affirming statement
    r_neg = policy.check("I do not want to die, I want to live.", {})
    assert not r_neg["matched"], "Life affirmation with negation must not trigger crisis"
    assert r_neg["negated_pattern"] is not None, "Negated phrase should be recorded"

    # Affirmative crisis statement
    r_pos = policy.check("I want to die tonight.", {})
    assert r_pos["matched"], "Affirmative crisis phrase must trigger crisis"

    # Inherently negative crisis statement
    r_inh = policy.check("I do not want to live anymore.", {})
    assert r_inh["matched"], "Inherently negative pattern ('do not want to live') must trigger crisis"


def test_issue_6_neural_boundary_head_fusion():
    """
    Issue 6: Neural component in TopicBoundaryHead was ignored.
    - Untrained: neural forward pass is bypassed, saving compute.
    - Trained: neural continuity is computed and fused with cosine continuity.
    """
    h_prev = torch.randn(2, 64)
    h_curr = torch.randn(2, 64)

    # Untrained mode
    head_untrained = TopicBoundaryHead(latent_dim=64, is_trained=False)
    out_untrained = head_untrained(h_curr, h_prev)
    assert out_untrained["neural_continuity"] is None
    cont_un = out_untrained["topic_continuity"]
    assert ((cont_un >= 0.0) & (cont_un <= 1.0)).all()

    # Trained mode
    head_trained = TopicBoundaryHead(latent_dim=64, is_trained=True)
    out_trained = head_trained(h_curr, h_prev)
    assert out_trained["neural_continuity"] is not None
    cont_tr = out_trained["topic_continuity"]
    assert ((cont_tr >= 0.0) & (cont_tr <= 1.0)).all()


def test_issue_5_untrained_question_detector():
    """
    Issue 5: Untrained TurnFunctionHead randomly flagged declarative statements as questions.
    Declarative disclosures must never be flagged as unresolved questions.
    """
    res_decl = TurnFunctionHead.detect_question_linguistic("I feel so lonely and depressed today")
    assert not res_decl["has_question"]
    assert res_decl["question_probability"] == 0.0

    res_q = TurnFunctionHead.detect_question_linguistic("What should I do to feel better?")
    assert res_q["has_question"]
    assert res_q["question_probability"] > 0.5


def test_issue_4_joint_supervision_and_affective_conditioning():
    """
    Issue 4: Risk and Emotion are trained separately.
    - Emotion examples provide inferred risk supervision.
    - Risk examples provide inferred emotion supervision.
    - RiskHead conditions on emotion probabilities and intensity.
    """
    # 1. Inferred risk from emotion labels
    risk_joy = infer_risk_from_emotion_data("I am so joyful today!", [EMOTION_TO_IDX["joy"]])
    assert risk_joy == 0  # normal

    risk_sad = infer_risk_from_emotion_data("I am crying and feel so sad.", [EMOTION_TO_IDX["sadness"]])
    assert risk_sad == 1  # distressed

    risk_crisis = infer_risk_from_emotion_data("I have written my suicide note already.", [EMOTION_TO_IDX["sadness"]])
    assert risk_crisis == 2  # risk

    # 2. Inferred emotions from risk labels
    emo_r0 = infer_emotions_from_risk_data("What is the capital of France?", 0)
    assert emo_r0 == []  # True negative emotion anchor

    emo_r1 = infer_emotions_from_risk_data("My anxiety has been through the roof since this morning.", 1)
    assert EMOTION_TO_IDX["anxiety"] in emo_r1
    assert EMOTION_TO_IDX["sadness"] in emo_r1

    emo_r2 = infer_emotions_from_risk_data("I bought a rope, I know what I am going to do.", 2)
    assert EMOTION_TO_IDX["sadness"] in emo_r2
    assert EMOTION_TO_IDX["grief"] in emo_r2

    # 3. Model affective conditioning
    model = CognitiveSLM(hidden_dim=128, ffn_dim=256, num_layers=2, num_heads=4)
    # Checkpoint backward compatibility: affective_proj weight is zero-initialized
    assert model.risk_head.affective_proj.weight.norm().item() == 0.0

    # Forward pass executes cleanly through affective conditioning
    x = torch.randint(0, 1000, (2, 16))
    out = model(x)
    assert out.risk_probs.shape == (2, 3)
    assert out.emotion_probs.shape == (2, 35)


if __name__ == "__main__":
    test_issue_7_negation_understanding()
    print("✓ test_issue_7_negation_understanding passed")
    test_issue_6_neural_boundary_head_fusion()
    print("✓ test_issue_6_neural_boundary_head_fusion passed")
    test_issue_5_untrained_question_detector()
    print("✓ test_issue_5_untrained_question_detector passed")
    test_issue_4_joint_supervision_and_affective_conditioning()
    print("✓ test_issue_4_joint_supervision_and_affective_conditioning passed")
    print("\nAll 4 issue test suites passed successfully!")


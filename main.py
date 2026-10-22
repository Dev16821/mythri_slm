"""
MYTHRI SLM — Interactive Testing & Diagnostic CLI

Run cognitive SLM inference to test:
1. Multi-label Emotion Classification & Intensity
2. Multi-class Risk Assessment (Normal, Distressed, Risk)
3. MC Dropout Epistemic Uncertainty & Abstention
4. Hybrid Safety Policy (Regex + Neural Risk + False Positive Veto)
5. Extracted Cognitive Latent Representation (h_cog, 1024-dim)

Usage:
    # Interactive REPL mode:
    python main.py

    # Single text evaluation:
    python main.py --text "I feel completely exhausted and lonely today"

    # Run diagnostic demo benchmark:
    python main.py --demo

    # Specify custom checkpoint or device:
    python main.py --checkpoint checkpoints/exp_0/best_model.pt --device cpu
"""

import os
# Auto-configure ROCm target architecture for AMD GPUs
if "HSA_OVERRIDE_GFX_VERSION" not in os.environ:
    os.environ["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"
if "TORCH_BLAS_PREFER_HIPBLASLT" not in os.environ:
    os.environ["TORCH_BLAS_PREFER_HIPBLASLT"] = "0"

import sys
import time
import argparse
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import torch

# Ensure repository root is on sys.path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from inference.predictor import CognitivePredictor
from safety.deterministic_policy import DeterministicSafetyPolicy, HybridSafetyLayer


# ANSI Terminal Colors
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'
    DIM = '\033[2m'


DEFAULT_PATTERNS = [
    "want to die",
    "kill myself",
    "end it all",
    "commit suicide",
    "better off dead",
    "don't want to live",
    "no point in living",
    "hurt myself",
    "take my own life",
    "suicide plan",
]

DEFAULT_HIGH_RISK_EMOTIONS = ["grief"]  # Aligned with 35 classes; 'sadness' excluded to prevent normal sad statements from escalating


def init_safety_layer(config: Optional[Dict] = None) -> HybridSafetyLayer:
    """Initializes the deterministic policy and hybrid safety layer."""
    safety_cfg = (config or {}).get("safety", {})
    patterns = safety_cfg.get("deterministic_patterns", DEFAULT_PATTERNS)
    high_risk_emotions = safety_cfg.get("high_risk_emotions", DEFAULT_HIGH_RISK_EMOTIONS)
    threshold = safety_cfg.get("high_risk_emotion_threshold", 0.85)

    det_policy = DeterministicSafetyPolicy(
        patterns=patterns,
        high_risk_emotions=high_risk_emotions,
        high_risk_emotion_threshold=threshold,
    )
    return HybridSafetyLayer(
        deterministic_policy=det_policy,
        slm_risk_threshold=0.70,
        slm_elevated_risk_threshold=0.40,
        slm_veto_risk_threshold=0.10,
        slm_veto_uncertainty_threshold=0.05,
        emotion_risk_threshold=0.70,
    )



def print_diagnostic_card(result: Dict, safety: Optional[object] = None, latency_ms: float = 0.0):
    """Prints a structured, formatted diagnostic card for an SLM prediction."""
    text = result["text"]
    emotions = result["emotions"]
    active_emotions = result["active_emotions"]
    intensity = result["emotion_intensity"]
    risk_state = result["risk_state"]
    risk_prob = result["risk_probability"]
    risk_probs = result.get("risk_probs", {})
    uncertainty = result["uncertainty"]
    abstain = result["abstain"]
    latent = result["latent"]
    latent_arr = np.array(latent)

    sep_double = "=" * 80
    sep_single = "-" * 80

    print(f"\n{Colors.BOLD}{sep_double}{Colors.END}")
    print(f"{Colors.HEADER}{Colors.BOLD} MYTHRI COGNITIVE SLM — INFERENCE REPORT{Colors.END}")
    print(f"{Colors.BOLD}{sep_double}{Colors.END}")
    print(f"{Colors.BOLD}Input Text:{Colors.END} \"{Colors.CYAN}{text}{Colors.END}\"")
    print(f"{Colors.DIM}Latency: {latency_ms:.2f} ms | Device: CPU/GPU | MC Dropout Samples: 10{Colors.END}")
    print(sep_single)

    # 1. Affective Profile
    sorted_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)
    top_5 = sorted_emotions[:5]

    print(f"{Colors.BOLD}[1] AFFECTIVE PROFILE (35-Class Multi-Label){Colors.END}")
    print(f"  Dominant Intensity : {Colors.BOLD}{intensity:.4f}{Colors.END}")
    print(f"  Active Emotions (≥0.50): {Colors.GREEN if active_emotions else Colors.DIM}{active_emotions or 'None'}{Colors.END}")
    print(f"  Top 5 Predicted Emotions:")
    for emo, prob in top_5:
        bar_len = int(prob * 25)
        bar = "█" * bar_len + "░" * (25 - bar_len)
        color = Colors.GREEN if prob >= 0.50 else (Colors.YELLOW if prob >= 0.20 else Colors.DIM)
        print(f"    - {emo:<16} : {color}{prob:6.2%} {bar}{Colors.END}")

    print(sep_single)

    # 2. Risk Assessment
    risk_color = (
        Colors.RED if risk_state == "risk"
        else (Colors.YELLOW if risk_state == "distressed" else Colors.GREEN)
    )
    print(f"{Colors.BOLD}[2] RISK ASSESSMENT (Multi-Class){Colors.END}")
    print(f"  Predicted Risk State : {risk_color}{Colors.BOLD}{risk_state.upper()}{Colors.END}")
    print(f"  P(Risk / Crisis)     : {risk_color}{risk_prob:6.2%}{Colors.END}")
    if risk_probs:
        print(f"  Full Probability Distribution:")
        print(f"    - Normal     : {risk_probs.get('normal', 0.0):6.2%}")
        print(f"    - Distressed : {risk_probs.get('distressed', 0.0):6.2%}")
        print(f"    - Risk       : {risk_probs.get('risk', 0.0):6.2%}")

    print(sep_single)

    # 3. Uncertainty & Epistemic Calibration
    unc_color = Colors.RED if abstain else (Colors.YELLOW if uncertainty > 0.08 else Colors.GREEN)
    print(f"{Colors.BOLD}[3] UNCERTAINTY & ABSTENTION (MC Dropout){Colors.END}")
    print(f"  Epistemic Variance   : {unc_color}{uncertainty:.6f}{Colors.END}")
    print(f"  Abstain Decision     : {Colors.RED if abstain else Colors.GREEN}{'YES (Model is Uncertain)' if abstain else 'NO (High Confidence)'}{Colors.END}")

    print(sep_single)

    # 4. Hybrid Safety Policy
    if safety is not None:
        decision = safety.evaluate(
            text=text,
            slm_risk_probability=risk_prob,
            slm_uncertainty=uncertainty,
            emotion_probs=emotions,
        )
        crisis_color = Colors.RED if decision.crisis_detected else Colors.GREEN
        veto_str = f" {Colors.CYAN}[SLM VETOED REGEX MATCH]{Colors.END}" if decision.vetoed else ""

        print(f"{Colors.BOLD}[4] HYBRID SAFETY GATEWAY (Regex + SLM + Veto){Colors.END}")
        print(f"  Crisis Detected      : {crisis_color}{Colors.BOLD}{'CRISIS FLAGGED' if decision.crisis_detected else 'SAFE / NON-CRISIS'}{Colors.END}{veto_str}")
        print(f"  Recommended Action   : {Colors.BOLD}{decision.recommended_action.upper()}{Colors.END} (Action Mode)")
        print(f"  Trigger Source       : {decision.trigger_source}")
        if decision.regex_match:
            print(f"  Matched Regex Rule   : '{decision.regex_match}'")
        print(f"  High-Distress Emo Risk: {decision.emotion_risk:.4f}")
        print(sep_single)

    # 5. Cognitive Latent Vector (h_cog)
    norm = np.linalg.norm(latent_arr)
    mean_val = float(np.mean(latent_arr))
    std_val = float(np.std(latent_arr))
    preview = [round(float(x), 4) for x in latent_arr[:6]]

    print(f"{Colors.BOLD}[5] COGNITIVE LATENT VECTOR (h_cog for Downstream LLM Control){Colors.END}")
    print(f"  Tensor Shape         : torch.Size([{len(latent)}]) — (Concatenated CLS + Mean Pool)")
    print(f"  Vector L2 Norm       : {norm:.4f}")
    print(f"  Mean: {mean_val:.4f} | Std: {std_val:.4f} | Min: {np.min(latent_arr):.4f} | Max: {np.max(latent_arr):.4f}")
    print(f"  First 6 Dimensions   : {preview} ...")
    print(sep_single)

    # 6. Topic Boundary & RAG Gate
    topic_decision = result.get("topic_decision")
    print(f"{Colors.BOLD}[6] TOPIC BOUNDARY & RAG WORKING-MEMORY GATE{Colors.END}")
    if topic_decision is not None:
        continuity = topic_decision["topic_continuity"]
        is_boundary = topic_decision["topic_boundary"]
        cos_sim = topic_decision["cosine_similarity"]
        action = topic_decision["recommended_memory_action"]
        reason = topic_decision["reason"]

        c_color = Colors.GREEN if continuity >= 0.65 else (Colors.YELLOW if continuity >= 0.40 else Colors.RED)
        act_color = Colors.RED if action == "FLUSH_RAG" else (Colors.YELLOW if action == "PRUNE_HISTORY" else Colors.GREEN)
        b_color = Colors.RED if is_boundary else Colors.GREEN

        print(f"  Topic Continuity     : {c_color}{Colors.BOLD}{continuity:6.2%}{Colors.END}")
        print(f"  Topic Boundary Shift : {b_color}{Colors.BOLD}{'TRUE (TOPIC PIVOT DETECTED)' if is_boundary else 'FALSE (CONTINUOUS THREAD)'}{Colors.END}")
        print(f"  Semantic Cosine Sim  : {cos_sim:+.4f} (Distance: {topic_decision['semantic_distance']:.4f})")
        print(f"  RAG & Memory Action  : {act_color}{Colors.BOLD}{action}{Colors.END}")
        print(f"  Gate Directive       : {Colors.DIM}{reason}{Colors.END}")
    else:
        print(f"  Status               : {Colors.CYAN}Initial turn in session — established topic baseline in working memory.{Colors.END}")

    # 7. Conversation State & Dialogue Progression
    conv_state = result.get("conversation_state")
    if conv_state is not None:
        print(sep_single)
        stage = conv_state["dialogue_stage"]
        turn_fn = conv_state["current_turn_function"]
        unresolved_q = conv_state["unresolved_question"]
        q_text = conv_state["unresolved_question_text"]
        goal = conv_state["previous_goal"]
        topic_anchor = conv_state["topic"]
        history = conv_state.get("stage_history", [])

        stage_color = Colors.CYAN if "ADVICE" in stage else (Colors.YELLOW if "ELABORATION" in stage else Colors.GREEN)
        q_color = Colors.RED if unresolved_q else Colors.GREEN

        print(f"{Colors.BOLD}[7] CONVERSATION STATE & DIALOGUE PROGRESSION{Colors.END}")
        print(f"  Dialogue Stage       : {stage_color}{Colors.BOLD}{stage}{Colors.END}")
        print(f"  Current Turn Function: {Colors.BOLD}{turn_fn.upper()}{Colors.END}")
        print(f"  Active Topic Anchor  : {Colors.DIM}{topic_anchor}{Colors.END}")
        print(f"  Unresolved Question  : {q_color}{Colors.BOLD}{'YES — Pending user question' if unresolved_q else 'NO — Statement / Reflection'}{Colors.END}")
        if q_text:
            print(f"  Pending Question Text: \"{q_text}\"")
        print(f"  Inferred User Goal   : {Colors.BOLD}{goal}{Colors.END}")
        if history:
            arc_str = " ──> ".join([s.split('_')[-1] for s in history[-5:]])
            print(f"  Dialogue Arc History : {Colors.DIM}{arc_str}{Colors.END}")

    print(f"{Colors.BOLD}{sep_double}{Colors.END}\n")


def run_benchmark_demo(predictor: CognitivePredictor, safety: HybridSafetyLayer):
    """Runs a curated test battery across typical clinical, edge-case, and topic boundary shifts."""
    test_cases = [
        # Explicit crisis
        ("I want to die, I can't take this pain anymore.", "Explicit Crisis"),
        # Implicit risk
        ("I've been thinking about permanent solutions to end this.", "Implicit Risk"),
        # Hard negative (dangerous words, safe context)
        ("I read an educational article about suicide prevention today.", "Hard Negative (Educational)"),
        # Third person reference
        ("My friend told me she wants to end it all, I am really worried about her.", "Third-Person Risk"),
        # Emotional distress (high distress, no self-harm)
        ("I feel so overwhelmed, lonely and completely hopeless about my career.", "High Distress / Non-Crisis"),
        # Positive / Everyday
        ("I'm feeling much better today, went for a morning run and had coffee.", "Normal / Positive"),
        # Ambiguous / Low confidence
        ("I don't know what to do next with my life.", "Ambiguous / Mild Distress"),
    ]

    print(f"\n{Colors.BOLD}{'=' * 80}")
    print(f" RUNNING MYTHRI SLM CURATED BENCHMARK DEMO ({len(test_cases)} single-turn cases)")
    print(f"{'=' * 80}{Colors.END}\n")

    predictor.reset_session()
    for i, (text, label) in enumerate(test_cases, 1):
        print(f"{Colors.BOLD}Test Case {i}/{len(test_cases)}: [{label}]{Colors.END}")
        t0 = time.perf_counter()
        result = predictor.predict(text, use_uncertainty=True, track_session=False)
        latency_ms = (time.perf_counter() - t0) * 1000
        print_diagnostic_card(result, safety=safety, latency_ms=latency_ms)
        time.sleep(0.05)

    # Multi-turn Topic Boundary & Progression Tests
    print(f"\n{Colors.BOLD}{'=' * 80}")
    print(f" MULTI-TURN TOPIC BOUNDARY & DIALOGUE ARC PROGRESSION SUITE")
    print(f"{'=' * 80}{Colors.END}\n")

    multi_turn_dialogues = [
        (
            "Demo 1: Seamless In-Topic Flow (Cooking -> Cooking)",
            [
                "Can you give me an easy recipe for chicken pasta?",
                "I also need a recipe for garlic bread and homemade tomato sauce.",
            ]
        ),
        (
            "Demo 2: Sharp Topic Pivot / RAG Contamination Circuit Breaker (Cooking -> Relationship Crisis)",
            [
                "Can you give me an easy recipe for chicken pasta?",
                "Actually, my partner and I had a terrible fight and I cannot stop crying.",
            ]
        ),
        (
            "Demo 3: Complete 5-Turn Clinical Dialogue Arc Progression",
            [
                # Turn 1: Introducing problem
                "I've been feeling terrible about my family lately.",
                # Turn 2: Elaborating and venting
                "They never listen to my boundaries, and my dad yelled at me yesterday in front of everyone.",
                # Turn 3: Actively seeking advice
                "What should I actually say to him the next time he starts shouting at me?",
                # Turn 4: Evaluating advice
                "I tried setting boundaries before, but that sounds really hard and just made him angrier.",
                # Turn 5: Closure and grounded integration
                "That grounding step makes sense. I will step into another room next time. Thank you.",
            ]
        ),
    ]

    for title, turns in multi_turn_dialogues:
        print(f"\n{Colors.HEADER}{Colors.BOLD}>>> {title} <<<{Colors.END}")
        predictor.reset_session()
        for t_idx, turn in enumerate(turns, 1):
            print(f"\n{Colors.BOLD}Turn {t_idx}: \"{turn}\"{Colors.END}")
            t0 = time.perf_counter()
            result = predictor.predict(turn, use_uncertainty=True, track_session=True)
            latency_ms = (time.perf_counter() - t0) * 1000
            print_diagnostic_card(result, safety=safety, latency_ms=latency_ms)



def run_interactive_repl(predictor: CognitivePredictor, safety: HybridSafetyLayer):
    """Runs an interactive read-eval-print loop with multi-turn session tracking."""
    print(f"\n{Colors.GREEN}{Colors.BOLD}MYTHRI SLM Interactive Console Ready.{Colors.END}")
    print("Multi-turn topic boundary tracking is ACTIVE (tracks consecutive conversational shifts).")
    print("Commands:")
    print("  'reset' : Clear session history & reset topic baseline")
    print("  'demo'  : Run benchmark test suite & topic transition demos")
    print("  'exit'  : Quit console\n")

    predictor.reset_session()

    while True:
        try:
            text = input(f"{Colors.BOLD}mythri-slm > {Colors.END}").strip()
            if not text:
                continue
            if text.lower() in ["exit", "quit", "q"]:
                print(f"{Colors.DIM}Exiting MYTHRI SLM console. Goodbye!{Colors.END}")
                break
            if text.lower() == "reset":
                predictor.reset_session()
                print(f"{Colors.CYAN}Session memory reset. Next utterance will establish a new topic baseline.{Colors.END}\n")
                continue
            if text.lower() == "demo":
                run_benchmark_demo(predictor, safety)
                continue

            t0 = time.perf_counter()
            result = predictor.predict(text, use_uncertainty=True, track_session=True)
            latency_ms = (time.perf_counter() - t0) * 1000
            print_diagnostic_card(result, safety=safety, latency_ms=latency_ms)

        except (KeyboardInterrupt, EOFError):
            print(f"\n{Colors.DIM}Session interrupted. Exiting.{Colors.END}")
            break
        except Exception as e:
            print(f"{Colors.RED}Inference Error: {e}{Colors.END}")



def main():
    parser = argparse.ArgumentParser(description="MYTHRI SLM Diagnostic & Testing CLI")
    parser.add_argument(
        "--text", type=str, default=None,
        help="Single text utterance to evaluate"
    )
    parser.add_argument(
        "--demo", action="store_true",
        help="Run the benchmark test suite across diverse clinical and edge cases"
    )
    parser.add_argument(
        "--checkpoint", type=str, default="checkpoints/exp_0/best_model.pt",
        help="Path to trained model checkpoint (default: checkpoints/exp_0/best_model.pt)"
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Execution device: 'cuda', 'cpu', or None for auto-detect"
    )
    parser.add_argument(
        "--no-uncertainty", action="store_true",
        help="Disable MC Dropout uncertainty estimation for faster single-pass inference"
    )
    args = parser.parse_args()

    # Determine device
    if args.device:
        dev_str = args.device
        if dev_str.startswith("rocm") or dev_str.startswith("hip"):
            dev_str = dev_str.replace("rocm", "cuda").replace("hip", "cuda")
        device = torch.device(dev_str)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_absolute():
        ckpt_path = ROOT_DIR / ckpt_path

    if not ckpt_path.exists():
        print(f"{Colors.RED}Error: Checkpoint not found at: {ckpt_path}{Colors.END}")
        print("Please check that 'checkpoints/exp_0/best_model.pt' exists or specify via --checkpoint.")
        sys.exit(1)

    print(f"{Colors.DIM}Loading MYTHRI SLM from: {ckpt_path.name} (device: {device})...{Colors.END}")
    t_load = time.perf_counter()
    try:
        predictor = CognitivePredictor.load(str(ckpt_path), device=device)
    except Exception as e:
        print(f"{Colors.RED}Failed to load checkpoint: {e}{Colors.END}")
        sys.exit(1)

    load_time_s = time.perf_counter() - t_load
    print(f"{Colors.GREEN}Model loaded successfully in {load_time_s:.2f}s.{Colors.END}")

    safety = init_safety_layer()

    # Mode 1: Single text argument
    if args.text:
        t0 = time.perf_counter()
        result = predictor.predict(args.text, use_uncertainty=not args.no_uncertainty)
        latency_ms = (time.perf_counter() - t0) * 1000
        print_diagnostic_card(result, safety=safety, latency_ms=latency_ms)
        return

    # Mode 2: Benchmark demo
    if args.demo:
        run_benchmark_demo(predictor, safety)
        return

    # Mode 3: Interactive REPL
    run_interactive_repl(predictor, safety)


if __name__ == "__main__":
    main()

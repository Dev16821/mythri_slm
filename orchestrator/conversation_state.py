"""
Conversation State & Dialogue Progression Engine for MYTHRI SLM.

Tracks where the conversation currently is across turns, modeling the natural
clinical dialogue arc:

    [1] PROBLEM_INTRODUCTION
             ↓
    [2] PROBLEM_ELABORATION
             ↓
    [3] ADVICE_SEEKING
             ↓
    [4] SOLUTION_EVALUATION
             ↓
    [5] CLOSURE_INTEGRATION

Prevents treating each turn as an isolated query, coordinating:
- Current dialogue stage
- Pragmatic speech-act turn function
- Pending unresolved questions
- Inferred user goal
- Topic boundaries and working-memory invalidation
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

from model.multi_task_heads import TurnFunctionHead


class DialogueStage:
    OPENING_GREETING = "OPENING_GREETING"
    PROBLEM_INTRODUCTION = "PROBLEM_INTRODUCTION"
    PROBLEM_ELABORATION = "PROBLEM_ELABORATION"
    ADVICE_SEEKING = "ADVICE_SEEKING"
    SOLUTION_EVALUATION = "SOLUTION_EVALUATION"
    CLOSURE_INTEGRATION = "CLOSURE_INTEGRATION"

    ORDER = [
        OPENING_GREETING,
        PROBLEM_INTRODUCTION,
        PROBLEM_ELABORATION,
        ADVICE_SEEKING,
        SOLUTION_EVALUATION,
        CLOSURE_INTEGRATION,
    ]


@dataclass
class ConversationState:
    """Structured representation of active conversation state."""
    topic: str
    dialogue_stage: str
    current_turn_function: str
    unresolved_question: bool
    unresolved_question_text: Optional[str]
    previous_goal: str
    topic_boundary: bool
    topic_continuity: float
    turn_count: int
    stage_history: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "topic": self.topic,
            "dialogue_stage": self.dialogue_stage,
            "current_turn_function": self.current_turn_function,
            "unresolved_question": self.unresolved_question,
            "unresolved_question_text": self.unresolved_question_text,
            "previous_goal": self.previous_goal,
            "topic_boundary": self.topic_boundary,
            "topic_continuity": self.topic_continuity,
            "turn_count": self.turn_count,
            "stage_history": list(self.stage_history),
        }


class ConversationStateEngine:
    """
    Stateful engine maintaining and updating conversation progression.
    
    Combines SLM cognitive outputs (emotions, risk, boundary, speech-act)
    with conversation history to manage state transitions.
    """

    # Pragmatic cues
    ADVICE_CUES = [
        r"\bwhat (should|can|do|must)\b.*?\b(do|say|try)\b",
        r"\bhow (do|can|should)\b.*?\b(handle|deal with|stop|fix|manage|respond)\b",
        r"\bany advice\b",
        r"\bhelp me (with|understand|figure out)\b",
        r"\bwhat do you suggest\b",
        r"\bwhat would you do\b",
        r"\bcan you help me\b",
        r"\bwhat to do\b",
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

    def __init__(self):
        self.state: Optional[ConversationState] = None
        self.pending_question: Optional[str] = None
        self.current_topic: str = "general"

    def reset(self):
        """Reset the conversation state for a new session."""
        self.state = None
        self.pending_question = None
        self.current_topic = "general"

    def update(
        self,
        text: str,
        emotions: Dict[str, float],
        risk_state: str,
        topic_decision: Optional[Dict[str, Any]] = None,
        turn_function_pred: Optional[str] = None,
        neural_has_question: bool = False,
    ) -> ConversationState:
        """
        Updates the active conversation state given new cognitive outputs.
        """
        text_lower = text.lower().strip()

        # 1. Topic boundary & continuity
        is_boundary = False
        continuity = 1.0
        if topic_decision is not None:
            raw_boundary = topic_decision.get("topic_boundary", False)
            continuity = topic_decision.get("topic_continuity", 1.0)
            
            # If the user is currently in problem disclosure and continues with sadness/distress,
            # it is a natural narrative elaboration, not an external topic pivot.
            if self.state and self.state.dialogue_stage in [DialogueStage.PROBLEM_INTRODUCTION, DialogueStage.PROBLEM_ELABORATION]:
                has_elevated_distress = (emotions.get("sadness", 0) > 0.15 or emotions.get("anger", 0) > 0.15 or emotions.get("anxiety", 0) > 0.15 or risk_state != "normal")
                is_boundary = raw_boundary and not has_elevated_distress
            else:
                is_boundary = raw_boundary


        # 2. Extract unresolved question with linguistic verification
        q_info = TurnFunctionHead.detect_question_linguistic(text)
        # Only trust neural_has_question if there is interrogative support, preventing
        # ungrounded false positives on declarative emotional venting
        has_unresolved_q = q_info["has_question"] or (neural_has_question and q_info["question_probability"] > 0)
        q_text = q_info["question_text"] if has_unresolved_q else None

        # 3. Determine Pragmatic Turn Function
        if any(re.search(p, text_lower) for p in self.CLOSURE_CUES):
            turn_fn = "closing_grounding"
        elif any(re.search(p, text_lower) for p in self.EVAL_CUES):
            turn_fn = "evaluating_advice"
        elif has_unresolved_q:
            turn_fn = "seeking_advice"
        elif emotions.get("sadness", 0) > 0.20 or emotions.get("anxiety", 0) > 0.20 or emotions.get("anger", 0) > 0.20 or risk_state != "normal":
            turn_fn = "elaborating_venting" if (self.state and self.state.turn_count >= 1 and not is_boundary) else "introducing_problem"
        elif turn_function_pred:
            turn_fn = turn_function_pred
        else:
            turn_fn = "casual_chat"



        # 4. Infer Topic Category
        if is_boundary or self.state is None:
            # Establish new topic label
            top_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)
            dominant_emo = top_emotions[0][0] if top_emotions and top_emotions[0][1] > 0.30 else "neutral"
            if risk_state == "risk":
                self.current_topic = f"crisis_intervention ({dominant_emo})"
            elif dominant_emo in ["sadness", "loneliness", "grief"]:
                self.current_topic = f"emotional_distress ({dominant_emo})"
            elif dominant_emo in ["anxiety", "fear"]:
                self.current_topic = f"anxiety_stress ({dominant_emo})"
            elif dominant_emo in ["anger", "annoyance"]:
                self.current_topic = f"interpersonal_conflict ({dominant_emo})"
            else:
                self.current_topic = f"conversational ({dominant_emo})"

        # 5. State Machine: Progress Dialogue Stage
        if self.state is None:
            # First turn in session
            turn_count = 1
            stage_history = []
            if turn_fn == "introducing_problem" or risk_state != "normal":
                stage = DialogueStage.PROBLEM_INTRODUCTION
            else:
                stage = DialogueStage.OPENING_GREETING
        else:
            turn_count = self.state.turn_count + 1
            stage_history = list(self.state.stage_history)
            prev_stage = self.state.dialogue_stage

            # Stage transitions based on pragmatic progression
            if turn_fn == "closing_grounding":
                stage = DialogueStage.CLOSURE_INTEGRATION
            elif turn_fn == "evaluating_advice":
                stage = DialogueStage.SOLUTION_EVALUATION
            elif turn_fn == "seeking_advice" or has_unresolved_q:
                stage = DialogueStage.ADVICE_SEEKING
            elif prev_stage == DialogueStage.PROBLEM_INTRODUCTION:
                stage = DialogueStage.PROBLEM_ELABORATION
            elif prev_stage == DialogueStage.PROBLEM_ELABORATION:
                if is_boundary and "conversational" in self.current_topic:
                    stage = DialogueStage.OPENING_GREETING
                else:
                    stage = DialogueStage.PROBLEM_ELABORATION
            elif prev_stage == DialogueStage.ADVICE_SEEKING:
                stage = DialogueStage.SOLUTION_EVALUATION
            elif prev_stage == DialogueStage.SOLUTION_EVALUATION and any(re.search(p, text_lower) for p in self.CLOSURE_CUES):
                stage = DialogueStage.CLOSURE_INTEGRATION
            else:
                stage = prev_stage

        stage_history.append(stage)


        # 6. Infer User Goal
        if risk_state == "risk":
            goal = "CRISIS_STABILIZATION"
        elif stage in [DialogueStage.PROBLEM_INTRODUCTION, DialogueStage.PROBLEM_ELABORATION]:
            goal = "EMOTIONAL_VENTING_AND_VALIDATION"
        elif stage == DialogueStage.ADVICE_SEEKING:
            goal = "ACTIONABLE_COPING_GUIDANCE"
        elif stage == DialogueStage.SOLUTION_EVALUATION:
            goal = "EXPLORATION_AND_BARRIER_REDUCTION"
        elif stage == DialogueStage.CLOSURE_INTEGRATION:
            goal = "GROUNDING_AND_WRAP_UP"
        else:
            goal = "CASUAL_CONNECTION"

        self.state = ConversationState(
            topic=self.current_topic,
            dialogue_stage=stage,
            current_turn_function=turn_fn,
            unresolved_question=has_unresolved_q,
            unresolved_question_text=q_text,
            previous_goal=goal,
            topic_boundary=is_boundary,
            topic_continuity=continuity,
            turn_count=turn_count,
            stage_history=stage_history,
        )

        return self.state

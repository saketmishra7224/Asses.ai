"""Adaptive questioning state machine (Phase 6). Deliberately lightweight.

Per technical turn: assess the candidate's answer -> update a small
persistent state (difficulty, coverage, weak/strong topics, asked-question
fingerprints, turn budget) -> feed that state into the Phase 4 planner.

Design rules:
* Correctness is judged by a tiny LLM-as-judge call with strict JSON
  validation; ANY failure falls back to a transparent heuristic. The model
  output never touches prompts unvalidated.
* No psychological claims: AnswerAssessment has exactly the specified
  fields (correctness, understanding, follow-up need, recommended action).
* Difficulty moves ONE step per turn, never jumps.
* Repetition is prevented with normalized question fingerprints persisted
  in the existing session (no second database).
* A turn budget (RAG_MAX_TURNS) forces a closing plan so follow-up loops
  cannot run forever.

Importing this module performs no network I/O.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator

from .planner import Difficulty, InterviewState, QuestionPlan

Correctness = Literal["correct", "partial", "incorrect", "unknown"]
Understanding = Literal["basic", "intermediate", "advanced", "unknown"]
RecommendedAction = Literal["advance", "deepen", "followup", "simplify", "change_topic"]

DIFFICULTY_ORDER = ("basic", "intermediate", "advanced")
MAX_TURNS_DEFAULT = 20
MAX_DEPTH_PER_TOPIC = 2  # turns on one topic before moving on (when not struggling)

_UNCERTAIN_RE = re.compile(r"\b(don'?t know|not sure|no idea|no clue|skip\b|i give up|never heard)\b")


class AnswerAssessment(BaseModel):
    """Structured judgment of one candidate answer. Factual only."""

    correctness: Correctness = "unknown"
    understanding: Understanding = "unknown"
    needs_followup: bool = False
    recommended_action: RecommendedAction = "advance"
    rationale: str = ""


class AdaptiveState(BaseModel):
    """Persisted per-session adaptive interview state (session["adaptive"])."""

    current_topic: str = ""
    current_difficulty: Difficulty = "basic"
    current_project: str = ""
    depth: int = 0
    questions_asked: List[str] = []  # recent raw questions (bounded, for prompts)
    asked_fingerprints: List[str] = []
    topics_covered: List[str] = []
    weak_topics: List[str] = []
    strong_topics: List[str] = []
    last_assessment: Optional[AnswerAssessment] = None
    assessment_history: List[Dict[str, Any]] = []  # per-turn {topic, correctness, needs_followup}
    technical_turns: int = 0
    max_turns: int = MAX_TURNS_DEFAULT
    should_wrap_up: bool = False

    def fingerprint_known(self, question: str) -> bool:
        return fingerprint(question) in set(self.asked_fingerprints)


def fingerprint(text: str) -> str:
    """Normalized question hash: case/punctuation/whitespace insensitive."""
    norm = re.sub(r"[^a-z0-9 ]", "", (text or "").lower())
    norm = re.sub(r"\s+", " ", norm).strip()
    return hashlib.sha1(norm.encode("utf-8")).hexdigest()[:16]


def load_adaptive_state(session: Dict[str, Any], max_turns: Optional[int] = None) -> AdaptiveState:
    """Read session["adaptive"], tolerating missing/corrupt data."""
    raw = (session or {}).get("adaptive")
    try:
        if isinstance(raw, dict):
            state = AdaptiveState.model_validate(raw)
        else:
            state = AdaptiveState()
    except Exception:
        state = AdaptiveState()
    if max_turns is None:
        try:
            max_turns = int(os.getenv("RAG_MAX_TURNS", str(MAX_TURNS_DEFAULT)))
        except ValueError:
            max_turns = MAX_TURNS_DEFAULT
    state.max_turns = max(1, max_turns)
    return state


# ------------------------------------------------------------ assessment ----
_ASSESS_INSTRUCTIONS = (
    "You judge a mock-interview answer. Respond with ONLY a JSON object with keys: "
    'correctness ("correct"/"partial"/"incorrect"/"unknown"), '
    'understanding ("basic"/"intermediate"/"advanced"/"unknown"), '
    "needs_followup (true/false), "
    'recommended_action ("advance"/"deepen"/"followup"/"simplify"/"change_topic"), '
    "rationale (one short factual sentence). "
    "Judge only what is written. Never infer confidence, personality, or traits."
)


def _heuristic_assess(answer: str) -> AnswerAssessment:
    """Transparent fallback when the judge call fails. Coarse but safe:
    unknown correctness by default; explicit uncertainty counts against."""
    text = (answer or "").strip()
    if _UNCERTAIN_RE.search(text.lower()):
        return AnswerAssessment(correctness="incorrect", understanding="basic",
                                needs_followup=True, recommended_action="simplify",
                                rationale="Candidate explicitly stated uncertainty (heuristic).")
    if len(text) < 30:
        return AnswerAssessment(correctness="partial", understanding="basic",
                                needs_followup=True, recommended_action="followup",
                                rationale="Answer too brief to judge (heuristic).")
    return AnswerAssessment(correctness="unknown", understanding="unknown",
                            needs_followup=False, recommended_action="advance",
                            rationale="No judgment available (heuristic).")


def assess_answer(
    question: str,
    answer: str,
    topic: str = "",
    evidence_snippet: str = "",
    mode: str = "llm",
) -> AnswerAssessment:
    """Judge one answer. LLM-as-judge with strict validation; any failure
    (no key, bad JSON, schema violation) falls back to the heuristic."""
    if not (answer or "").strip():
        return AnswerAssessment(correctness="unknown", understanding="unknown",
                                needs_followup=True, recommended_action="followup",
                                rationale="Empty answer.")
    mode = (mode or "llm").strip().lower()
    if mode != "llm":
        return _heuristic_assess(answer)
    prompt = (
        f"{_ASSESS_INSTRUCTIONS}\n\n"
        f"TOPIC: {topic or '(general)'}\n"
        f"QUESTION ASKED: {(question or '(unknown)')[:800]}\n"
        f"CANDIDATE ANSWER: {answer[:2000]}\n"
        + (f"CANDIDATE EVIDENCE (facts only): {evidence_snippet[:800]}\n" if evidence_snippet else "")
        + "JSON assessment:"
    )
    try:
        from generate import generate_text  # lazy: keeps RAG import offline-safe

        raw = (generate_text(prompt) or "").strip()
        if raw.startswith("```"):
            raw = raw.strip("`")
            if raw.lower().startswith("json"):
                raw = raw[4:]
            raw = raw.strip()
        return AnswerAssessment.model_validate(json.loads(raw))
    except Exception:
        return _heuristic_assess(answer)


# -------------------------------------------------------------- transitions --
def _step(difficulty: Difficulty, direction: int) -> Difficulty:
    i = DIFFICULTY_ORDER.index(difficulty)
    return DIFFICULTY_ORDER[min(2, max(0, i + direction))]


def update_after_answer(
    state: AdaptiveState,
    assessment: AnswerAssessment,
    topic: str = "",
) -> AdaptiveState:
    """Advance the state machine one turn. Single-step difficulty moves;
    topics marked covered only after sufficient, non-struggling depth."""
    topic = (topic or "").strip()
    state.technical_turns += 1
    state.last_assessment = assessment
    try:
        state.assessment_history.append({
            "topic": topic,
            "correctness": assessment.correctness,
            "needs_followup": bool(assessment.needs_followup),
        })
        del state.assessment_history[:-20]
    except Exception:
        pass

    if topic:
        if topic == state.current_topic:
            state.depth += 1
        else:
            state.current_topic = topic
            state.depth = 0

    correctness = assessment.correctness
    if correctness == "correct":
        state.current_difficulty = _step(state.current_difficulty, +1)
        if topic and topic not in state.strong_topics:
            state.strong_topics.append(topic)
        if topic and state.depth >= 1 and topic not in state.topics_covered:
            state.topics_covered.append(topic)
    elif correctness == "incorrect":
        state.current_difficulty = _step(state.current_difficulty, -1)
        if topic and topic not in state.weak_topics:
            state.weak_topics.append(topic)
        # stay on the topic: do NOT mark covered
    elif correctness == "partial":
        pass  # difficulty unchanged; follow-up handled via needs_followup
    else:  # unknown: avoid stalls, but only after real depth
        if topic and state.depth >= MAX_DEPTH_PER_TOPIC and topic not in state.topics_covered:
            state.topics_covered.append(topic)

    for lst in (state.topics_covered, state.weak_topics, state.strong_topics):
        del lst[:-30]
    if state.technical_turns >= state.max_turns:
        state.should_wrap_up = True
    return state


def record_asked(state: AdaptiveState, question_text: str) -> AdaptiveState:
    """Fingerprint the just-asked question for repetition prevention."""
    fp = fingerprint(question_text or "")
    if fp and fp not in state.asked_fingerprints:
        state.asked_fingerprints.append(fp)
        del state.asked_fingerprints[:-50]
    text = (question_text or "").strip()
    if text:
        state.questions_asked.append(text[:300])
        del state.questions_asked[:-5]
    return state


def closing_plan() -> QuestionPlan:
    """Forced end-of-budget plan: reflective close, no new topic."""
    return QuestionPlan(
        intent="behavioral",
        topic="wrap-up",
        source=None,
        difficulty="basic",
        question_style="behavioral",
        retrieval_query="closing the interview reflection",
        reason="Interview turn budget reached; wrap up gracefully.",
    )


def to_planner_state(state: AdaptiveState) -> InterviewState:
    """Bridge adaptive state -> Phase 4 planner input."""
    signal = {"correct": "correct", "incorrect": "struggling"}.get(
        (state.last_assessment.correctness if state.last_assessment else "unknown"), "unknown")
    return InterviewState(
        phase="technical",
        covered_topics=list(state.topics_covered),
        previous_difficulty=state.current_difficulty,
        last_signal=signal,  # type: ignore[arg-type]
        turns_taken=state.technical_turns,
    )


def adaptive_prompt_block(state: AdaptiveState, assessment: Optional[AnswerAssessment]) -> str:
    """[ADAPTIVE STATE] prompt section: assessment + do-not-repeat list."""
    lines = ["[ADAPTIVE STATE]"]
    if assessment is not None:
        lines.append(
            f"Last answer: {assessment.correctness} ({assessment.understanding}); "
            f"follow-up needed: {assessment.needs_followup}; "
            f"recommended: {assessment.recommended_action}."
        )
    lines.append(f"Current level: {state.current_difficulty} (topic: {state.current_topic or 'new'}).")
    if state.questions_asked:
        lines.append("Already asked — do NOT repeat or paraphrase these:")
        lines.extend(f"- {q}" for q in state.questions_asked[-5:])
    if state.should_wrap_up:
        lines.append("Turn budget reached: wrap up the interview within the next 1-2 turns.")
    lines.append("[/ADAPTIVE STATE]")
    return "\n".join(lines)


__all__ = [
    "AnswerAssessment",
    "AdaptiveState",
    "assess_answer",
    "closing_plan",
    "fingerprint",
    "load_adaptive_state",
    "record_asked",
    "to_planner_state",
    "adaptive_prompt_block",
    "update_after_answer",
]

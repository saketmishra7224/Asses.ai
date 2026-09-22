"""Structured final interview evaluation (Phase 7).

Inputs: transcript, questions asked, answers, topics covered, difficulty
progression, per-turn answer assessments, and candidate profile evidence.
Output: a validated EvaluationReport (structured JSON internally) rendered
to the existing frontend-compatible feedback string.

Anti-invention rules (enforced by prompt + renderer, tested):
* Interview-knowledge claims ("demonstrated X") require transcript evidence
  and are labeled [Interview].
* Resume/profile facts are labeled [Profile] and never presented as
  demonstrated skill.
* With insufficient evidence, scores are null ("n/a") and the report says
  so instead of guessing.
* Any malformed LLM output falls back to a factual heuristic report.

Importing this module performs no network I/O.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator

DIMENSIONS = (
    "communication",
    "cs_fundamentals",
    "problem_solving",
    "technical_depth",
    "project_understanding",
    "answer_quality",
)
DIMENSION_LABELS = {
    "communication": "Communication",
    "cs_fundamentals": "CS Fundamentals",
    "problem_solving": "Problem Solving",
    "technical_depth": "Technical Depth",
    "project_understanding": "Project Understanding",
    "answer_quality": "Answer Quality",
}

EvidenceSource = Literal["interview", "profile"]


class EvidenceItem(BaseModel):
    claim: str = ""
    source: EvidenceSource = "interview"
    detail: str = ""  # answer quote/topic for interview; resume fact for profile


class DimensionScore(BaseModel):
    score: Optional[float] = Field(default=None, ge=0, le=10)
    strengths: List[str] = []
    gaps: List[str] = []
    evidence: List[EvidenceItem] = []

    @field_validator("strengths", "gaps", mode="before")
    @classmethod
    def _str_list(cls, v: Any) -> Any:
        if isinstance(v, list):
            return [str(x).strip() for x in v if str(x).strip()][:6]
        return []


class EvaluationReport(BaseModel):
    communication: DimensionScore = DimensionScore()
    cs_fundamentals: DimensionScore = DimensionScore()
    problem_solving: DimensionScore = DimensionScore()
    technical_depth: DimensionScore = DimensionScore()
    project_understanding: DimensionScore = DimensionScore()
    answer_quality: DimensionScore = DimensionScore()
    strengths: List[str] = []
    weaknesses: List[str] = []
    topics_to_improve: List[str] = []
    recommended_practice: List[str] = []
    overall_score: Optional[float] = Field(default=None, ge=0, le=10)
    summary: str = ""

    def dimensions(self) -> Dict[str, DimensionScore]:
        return {name: getattr(self, name) for name in DIMENSIONS}


def build_evaluation_inputs(session: Dict[str, Any], history: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collect everything the evaluator may use. Pure function, never raises
    on odd shapes (missing parts become explicit empty lists)."""
    session = session if isinstance(session, dict) else {}
    history = [m for m in (history or []) if isinstance(m, dict)]
    profile = session.get("profile") or {}
    adaptive = session.get("adaptive") or {}
    plans = [p for p in (session.get("question_plans") or []) if isinstance(p, dict)]

    questions = [str(m.get("content", "")) for m in history if m.get("role") == "ai"]
    answers = [str(m.get("content", "")) for m in history if m.get("role") == "user"]
    topics = list(dict.fromkeys(
        [str(t) for t in (adaptive.get("topics_covered") or []) if str(t).strip()]
        + [str(p.get("topic", "")) for p in plans if p.get("topic")]
        + [str(p.get("project_name", "")) for p in plans if p.get("project_name")]
    ))
    difficulties = [str(p.get("difficulty", "")) for p in plans
                    if p.get("difficulty") in ("basic", "intermediate", "advanced")]
    assessments = [a for a in (adaptive.get("assessment_history") or []) if isinstance(a, dict)]

    projects = []
    for repo in (profile.get("github") or []):
        if not isinstance(repo, dict):
            continue
        url, name = str(repo.get("url") or ""), ""
        if "github.com" in url:
            parts = url.strip().strip("/").split("/")
            if len(parts) >= 5:
                name = parts[4]
        projects.append({"name": name, "repository": url,
                         "language": str(repo.get("language") or ""),
                         "description": str(repo.get("description") or "")[:200]})

    leet = []
    stats = profile.get("leetcode_stats") or {}
    counts = (stats.get("tagProblemCounts") if isinstance(stats, dict) else None) or {}
    if not counts and isinstance(profile.get("leetcode_raw"), str) and profile["leetcode_raw"].strip().startswith("{"):
        try:
            counts = json.loads(profile["leetcode_raw"]).get("tagProblemCounts", {})
        except Exception:
            counts = {}
    for bucket in ("fundamental", "intermediate", "advanced"):
        for tag in counts.get(bucket) or []:
            if isinstance(tag, dict) and tag.get("tagSlug"):
                leet.append({"topic": str(tag["tagSlug"]),
                             "solved": int(tag.get("problemsSolved") or 0),
                             "category": bucket})

    return {
        "transcript": "\n".join(f"{m.get('role')}: {m.get('content')}" for m in history[-60:]),
        "questions": questions[-20:],
        "answers": answers[-20:],
        "topics_covered": topics,
        "difficulty_progression": difficulties[-20:],
        "assessments": assessments[-20:],
        "weak_topics": [str(t) for t in (adaptive.get("weak_topics") or [])],
        "strong_topics": [str(t) for t in (adaptive.get("strong_topics") or [])],
        "projects": projects,
        "leetcode": leet,
        "turns": len(history),
        "technical_turns": int(adaptive.get("technical_turns", 0) or 0),
    }


_EVAL_INSTRUCTIONS = (
    "You are a senior hiring manager writing a structured mock-interview evaluation. "
    "Respond with ONLY a JSON object with keys: communication, cs_fundamentals, "
    "problem_solving, technical_depth, project_understanding, answer_quality "
    "(each: {score 0-10 or null, strengths[], gaps[], evidence[]}), strengths[], "
    "weaknesses[], topics_to_improve[], recommended_practice[], overall_score (0-10 or null), summary. "
    "EVIDENCE RULES (strict): each evidence item has {claim, source, detail} where "
    'source is "interview" (something the candidate SAID/did in the transcript, '
    'quote or paraphrase it in detail) or "profile" (a resume/GitHub/LeetCode fact '
    "from the profile section). NEVER label a profile fact as interview evidence. "
    "NEVER claim demonstrated knowledge without transcript evidence. "
    "If a dimension has no evidence, use score null with an empty evidence list. "
    "Keep every list to at most 5 short items."
)


def _parse_report(raw: str) -> EvaluationReport:
    text = (raw or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("evaluator did not return a JSON object")
    return EvaluationReport.model_validate(data)


def heuristic_report(inputs: Dict[str, Any]) -> EvaluationReport:
    """Factual fallback: no scores without evidence, only observed coverage."""
    topics = inputs.get("topics_covered") or []
    weak = inputs.get("weak_topics") or []
    strong = inputs.get("strong_topics") or []
    turns = inputs.get("technical_turns", 0)
    if not topics and turns == 0:
        summary = ("Insufficient interview evidence for scoring: no technical "
                   "questions were answered. Scores are marked n/a rather than guessed.")
    else:
        summary = (f"Partial evidence ({turns} technical turn(s), "
                   f"{len(topics)} topic(s) touched). Automated evaluation unavailable; "
                   "scores withheld where evidence is thin.")
    return EvaluationReport(
        topics_to_improve=list(dict.fromkeys([str(t) for t in weak]))[:5],
        strengths=[f"Discussed: {t}" for t in strong[:3]],
        weaknesses=[f"Needs work: {t}" for t in weak[:3]] or (
            ["No completed technical assessment"] if turns == 0 else []),
        recommended_practice=[f"Practice {t} fundamentals" for t in weak[:3]],
        summary=summary,
    )


def evaluate_interview(inputs: Dict[str, Any]) -> tuple:
    """Run the LLM evaluator with strict validation. Returns (report, used_llm).

    used_llm=False means the heuristic fallback produced the report (the
    caller maps this to mock=True, preserving the old mock semantics).
    """
    prompt = (
        f"{_EVAL_INSTRUCTIONS}\n\nTRANSCRIPT (interview evidence):\n"
        f"{inputs.get('transcript') or '(no conversation yet)'}\n\n"
        f"TOPICS COVERED: {inputs.get('topics_covered') or []}\n"
        f"DIFFICULTY PROGRESSION: {inputs.get('difficulty_progression') or []}\n"
        f"ANSWER ASSESSMENTS: {inputs.get('assessments') or []}\n"
        f"PROFILE EVIDENCE (resume/GitHub/LeetCode facts only): "
        f"projects={inputs.get('projects') or []} leetcode={inputs.get('leetcode') or []}\n\n"
        "JSON evaluation:"
    )
    try:
        from generate import generate_text  # lazy: keeps RAG import offline-safe

        return _parse_report(generate_text(prompt)), True
    except Exception:
        return heuristic_report(inputs), False


def _fmt_score(score: Optional[float]) -> str:
    return f"{score:.1f}/10" if score is not None else "n/a (insufficient evidence)"


def render_feedback(report: EvaluationReport) -> str:
    """Render the report as the frontend-compatible feedback string."""
    lines = ["Interview Feedback"]
    if report.overall_score is not None:
        lines.append(f"Overall: {_fmt_score(report.overall_score)}")
    for name in DIMENSIONS:
        dim = getattr(report, name)
        lines.append(f"{DIMENSION_LABELS[name]}: {_fmt_score(dim.score)}")
        for s in dim.strengths[:2]:
            lines.append(f"  + {s}")
        for g in dim.gaps[:2]:
            lines.append(f"  - {g}")
        for ev in dim.evidence[:2]:
            tag = "Interview" if ev.source == "interview" else "Profile"
            lines.append(f"  [{tag}] {ev.claim}" + (f" — {ev.detail}" if ev.detail else ""))
    if report.strengths:
        lines.append("Strengths: " + "; ".join(report.strengths[:4]))
    if report.weaknesses:
        lines.append("Weaknesses: " + "; ".join(report.weaknesses[:4]))
    if report.topics_to_improve:
        lines.append("Topics to improve: " + ", ".join(report.topics_to_improve[:5]))
    if report.recommended_practice:
        lines.append("Recommended practice: " + "; ".join(report.recommended_practice[:4]))
    if report.summary:
        lines.append(report.summary)
    return "\n".join(lines)


__all__ = [
    "DIMENSIONS",
    "EvidenceItem",
    "DimensionScore",
    "EvaluationReport",
    "build_evaluation_inputs",
    "evaluate_interview",
    "heuristic_report",
    "render_feedback",
]

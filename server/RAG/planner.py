"""Question planner (Phase 4): structured next-question plans.

The planner does NOT write natural-language questions. It produces a
validated :class:`QuestionPlan` (intent, topic, difficulty, retrieval query,
...) that the Phase 3 retriever + Gemini interviewer consume in a later phase.

Two strategies:
  * HeuristicPlanner — deterministic rules over candidate evidence +
    interview state. No network, no key. Default; also the safety net.
  * LLMPlanner — asks Gemini for a plan, validates it strictly, and falls
    back to the heuristic on ANY failure (bad JSON, schema violation, or a
    hallucinated project/skill). The interview can therefore never stall on
    a malformed model response, nor ask about invented projects.

Importing this module performs no network I/O (the Gemini call is lazily
imported inside LLMPlanner).
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

INTENTS = (
    "resume_followup",
    "project_followup",
    "github_project",
    "github_implementation",
    "leetcode_topic",
    "cs_fundamentals",
    "coding_problem",
    "optimization",
    "clarification",
    "behavioral",
)
Intent = Literal[
    "resume_followup", "project_followup", "github_project",
    "github_implementation", "leetcode_topic", "cs_fundamentals",
    "coding_problem", "optimization", "clarification", "behavioral",
]
Difficulty = Literal["basic", "intermediate", "advanced"]
QuestionStyle = Literal["conceptual", "coding", "scenario", "behavioral", "clarification"]
Signal = Literal["correct", "struggling", "unknown"]
Phase = Literal["intro", "technical"]

SOURCELESS_INTENTS = ("cs_fundamentals", "clarification", "behavioral")
DIFFICULTY_ORDER = ("basic", "intermediate", "advanced")
CS_TOPIC_BANK = ("OOPs", "DBMS", "Operating Systems", "Computer Networks", "SQL")

# Chunk-type hints consumed by retrieve_candidate_context (Phase 3).
INTENT_CHUNK_TYPE = {
    "github_project": "project",
    "github_implementation": "readme",
    "project_followup": None,
    "resume_followup": "section",
    "leetcode_topic": "topic",
}

INTENT_STYLE = {
    "coding_problem": "coding",
    "leetcode_topic": "coding",
    "optimization": "coding",
    "behavioral": "behavioral",
    "clarification": "clarification",
    "github_implementation": "scenario",
}


class PlannerError(Exception):
    """Unrecoverable planner failure (bad inputs, not model output)."""


class QuestionPlan(BaseModel):
    """Structured next-question plan. Invariants are enforced so the
    interviewer can never be pointed at an invented project/skill:

    * github_*/project_followup intents require a real project_name,
    * source=None is only allowed for sourceless intents,
    * retrieval_query is always non-empty (the retriever needs it).
    """

    intent: Intent
    topic: str = Field(min_length=1, max_length=200)
    source: Optional[Literal["resume", "github", "leetcode"]] = None
    project_name: str = ""
    difficulty: Difficulty = "basic"
    question_style: QuestionStyle = "conceptual"
    retrieval_query: str = Field(min_length=1, max_length=500)
    reason: str = ""

    @field_validator("topic", "retrieval_query", "project_name", "reason", mode="before")
    @classmethod
    def _strip(cls, v: Any, info) -> Any:
        if isinstance(v, str):
            s = v.strip()
            if info.field_name == "retrieval_query" and len(s) > 500:
                s = s[:500]
            elif info.field_name == "topic" and len(s) > 200:
                s = s[:200]
            elif info.field_name == "project_name" and len(s) > 100:
                s = s[:100]
            return s
        return v

    @model_validator(mode="after")
    def _check_grounding(self) -> "QuestionPlan":
        if self.intent in ("github_project", "github_implementation", "project_followup"):
            if not self.project_name:
                raise ValueError(f"intent {self.intent!r} requires project_name")
            if self.source is None:
                self.source = "github"
            if self.source != "github":
                raise ValueError(f"intent {self.intent!r} requires source 'github'")
        if self.source is None and self.intent not in SOURCELESS_INTENTS:
            raise ValueError(f"intent {self.intent!r} requires a source (resume/github/leetcode)")
        return self

    def retrieval_args(self) -> Dict[str, Any]:
        """kwargs for Phase 3 retrieve_candidate_context (besides query/k)."""
        return {
            "source": self.source,
            "chunk_type": INTENT_CHUNK_TYPE.get(self.intent),
            "project_name": self.project_name or None,
            "topic": self.topic if self.intent in ("leetcode_topic",) else None,
        }


class ProjectEvidence(BaseModel):
    name: str = ""
    repository: str = ""
    language: str = ""
    description: str = ""


class TopicEvidence(BaseModel):
    topic: str = ""
    problems_solved: int = 0
    category: str = ""


class CandidateContext(BaseModel):
    """Everything the planner may reference. Empty == unknown candidate."""

    projects: List[ProjectEvidence] = []
    leetcode_topics: List[TopicEvidence] = []
    resume_sections: List[str] = []
    resume_skills: List[str] = []
    summary: str = ""

    def project_names(self) -> List[str]:
        return [p.name for p in self.projects if p.name]

    def is_empty(self) -> bool:
        return not self.projects and not self.leetcode_topics and not self.resume_skills


class InterviewState(BaseModel):
    """Where the interview stands. `covered_topics` are lowercase-normalized.

    Convention for the caller (Phase 5 wiring): after asking a project
    follow-up, record ``"<project-name>:followup"`` in covered_topics so the
    planner advances to fresh topics instead of repeating the follow-up.
    """

    phase: Phase = "technical"
    covered_topics: List[str] = []
    previous_difficulty: Difficulty = "basic"
    last_signal: Signal = "unknown"
    turns_taken: int = 0

    @field_validator("covered_topics", mode="before")
    @classmethod
    def _normalize_covered(cls, v: Any) -> Any:
        if not isinstance(v, list):
            return []
        return [str(t).strip().lower() for t in v if str(t).strip()]


def context_from_profile(profile: Optional[Dict[str, Any]]) -> CandidateContext:
    """Build planner context from a session profile dict (as stored by upload).

    Tolerates legacy shapes: github items with or without url/language,
    leetcode_stats as {tagProblemCounts}, {matchedUser: {...}}, or raw JSON.
    Never raises on odd input — worst case returns an empty context, which
    routes to the CS-fundamentals fallback instead of inventing evidence.
    """
    ctx = CandidateContext()
    if not isinstance(profile, dict):
        return ctx
    for i, repo in enumerate(profile.get("github") or []):
        if not isinstance(repo, dict):
            continue
        url = str(repo.get("url") or "")
        slug = ""
        if "github.com" in url:
            parts = url.strip().strip("/").split("/")
            if len(parts) >= 5:
                slug = f"{parts[3]}/{parts[4]}"
        name = slug.split("/")[-1] if slug else f"project-{i + 1}"
        lang = str(repo.get("language") or "").strip()
        if len(lang) > 40 or "\n" in lang or lang.startswith("#"):
            lang = ""
        if repo.get("description") or repo.get("readme") or url:
            ctx.projects.append(ProjectEvidence(
                name=name[:100],
                repository=slug[:100],
                language=lang,
                description=str(repo.get("description") or "")[:300],
            ))
    stats = profile.get("leetcode_stats")
    if stats is None and profile.get("leetcode_raw"):
        try:
            stats = json.loads(profile["leetcode_raw"]) if str(profile["leetcode_raw"]).strip().startswith("{") else None
        except Exception:
            stats = None
    counts = {}
    if isinstance(stats, dict):
        counts = stats.get("tagProblemCounts") or stats.get("matchedUser", {}).get("tagProblemCounts") or {}
    for bucket in ("fundamental", "intermediate", "advanced"):
        for tag in counts.get(bucket) or []:
            if not isinstance(tag, dict) or not tag.get("tagSlug"):
                continue
            ctx.leetcode_topics.append(TopicEvidence(
                topic=str(tag["tagSlug"]),
                problems_solved=int(tag.get("problemsSolved") or 0),
                category=bucket,
            ))
    ctx.resume_skills = [str(s) for s in (profile.get("resume_skills") or []) if str(s).strip()]
    resume_text = str(profile.get("resume_text") or "")
    if resume_text.strip():
        ctx.summary = " ".join(resume_text.split())[:500]
    return ctx


def _escalate(difficulty: Difficulty, signal: Signal) -> Difficulty:
    i = DIFFICULTY_ORDER.index(difficulty)
    if signal == "correct":
        return DIFFICULTY_ORDER[min(i + 1, 2)]
    if signal == "struggling":
        return DIFFICULTY_ORDER[max(i - 1, 0)]
    return difficulty


class HeuristicPlanner:
    """Deterministic planner: evidence + state in, validated plan out."""

    def plan(self, state: InterviewState, context: CandidateContext) -> QuestionPlan:
        if state.phase == "intro":
            return QuestionPlan(
                intent="behavioral",
                topic="self-introduction",
                source=None,
                difficulty="basic",
                question_style="behavioral",
                retrieval_query="candidate background and self-introduction",
                reason="Introduction phase: warm-up before technical rounds.",
            )
        covered = set(state.covered_topics)

        # 1. Struggling candidates get a clarification lifeline, not a new topic.
        if state.last_signal == "struggling" and covered:
            last = state.covered_topics[-1].removesuffix(":followup")
            return QuestionPlan(
                intent="clarification",
                topic=last,
                source=None,
                difficulty="basic",
                question_style="clarification",
                retrieval_query=f"basics of {last}",
                reason="Candidate struggled; clarify the current topic before moving on.",
            )

        # 2. Fresh projects first (breadth before depth).
        for proj in context.projects:
            if proj.name.lower() not in covered and proj.repository.lower() not in covered:
                difficulty = _escalate("basic", "unknown")
                if state.turns_taken >= 4:
                    difficulty = "intermediate"
                return QuestionPlan(
                    intent="github_project",
                    topic=proj.name[:200],
                    source="github",
                    project_name=proj.name[:100],
                    difficulty=difficulty,
                    question_style="conceptual",
                    retrieval_query=f"{proj.name} {proj.language} project overview and implementation".strip()[:490],
                    reason="Project not yet explored; start with its goals and design.",
                )

        # 3. Revisit a covered project at depth (why/how, not what again) —
        # once per project: callers record "<name>:followup" afterwards.
        for proj in context.projects:
            name = proj.name.lower()
            if name in covered and f"{name}:followup" not in covered:
                deep = state.turns_taken >= 6
                return QuestionPlan(
                    intent="github_implementation" if deep else "project_followup",
                    topic=proj.name[:200],
                    source="github",
                    project_name=proj.name[:100],
                    difficulty=_escalate(state.previous_difficulty, state.last_signal),
                    question_style="scenario" if deep else "conceptual",
                    retrieval_query=f"why {proj.name} design decisions {proj.language} trade-offs".strip()[:490],
                    reason="Project already introduced; probe decisions and trade-offs instead of repeating basics.",
                )
        return self._leetcode_or_cs(state, context, covered)

    def _leetcode_or_cs(
        self, state: InterviewState, context: CandidateContext, covered: set
    ) -> QuestionPlan:
        # 4. Strongest uncovered LeetCode topic (meet them where they're strong).
        candidates = [t for t in context.leetcode_topics
                      if t.topic and t.problems_solved > 0 and t.topic.lower() not in covered]
        if candidates:
            candidates.sort(key=lambda t: t.problems_solved, reverse=True)
            top = candidates[0]
            difficulty = _escalate(state.previous_difficulty, state.last_signal)
            # Evidence matters but never dictates alone: high solve counts lift
            # the floor, the answer signal still drives escalation.
            if top.problems_solved >= 50 and state.last_signal != "struggling":
                if DIFFICULTY_ORDER.index(difficulty) < 1:
                    difficulty = "intermediate"
            return QuestionPlan(
                intent="leetcode_topic",
                topic=top.topic,
                source="leetcode",
                difficulty=difficulty,
                question_style="coding",
                retrieval_query=f"candidate experience with {top.topic} algorithms on LeetCode",
                reason=f"Strongest unexplored topic ({top.problems_solved} solved); probe with a coding question.",
            )
        # 5. Resume skills never discussed.
        for skill in context.resume_skills:
            if skill.lower() not in covered:
                return QuestionPlan(
                    intent="resume_followup",
                    topic=skill,
                    source="resume",
                    difficulty=_escalate(state.previous_difficulty, state.last_signal),
                    question_style="conceptual",
                    retrieval_query=f"candidate experience with {skill}",
                    reason="Resume skill not yet discussed.",
                )
        # 6. General CS fallback: source=None, nothing invented.
        for topic in CS_TOPIC_BANK:
            if topic.lower() not in covered:
                return QuestionPlan(
                    intent="cs_fundamentals",
                    topic=topic,
                    source=None,
                    difficulty=_escalate(state.previous_difficulty, state.last_signal),
                    question_style="conceptual",
                    retrieval_query=f"core computer science fundamentals: {topic}",
                    reason="No unexplored candidate evidence left; general CS question.",
                )
        return QuestionPlan(
            intent="behavioral",
            topic="project challenges",
            source=None,
            difficulty="intermediate",
            question_style="behavioral",
            retrieval_query="candidate project challenges and teamwork experience",
            reason="All tracked topics covered; close with behavioral reflection.",
        )


class LLMPlanner:
    """Gemini-backed planner with strict validation + heuristic fallback.

    Any failure — no API key, bad JSON, schema violation, or a plan that
    references evidence absent from the context — falls back to
    HeuristicPlanner, so the interview never stalls and never hallucinates.
    """

    SYSTEM_INSTRUCTIONS = (
        "You are an interview question planner. Respond with ONLY a JSON object "
        'with keys: intent, topic, source ("resume"/"github"/"leetcode"/null), '
        "project_name, difficulty (basic/intermediate/advanced), "
        'question_style (conceptual/coding/scenario/behavioral/clarification), '
        "retrieval_query, reason. "
        f"intent must be one of: {', '.join(INTENTS)}. "
        "NEVER invent projects, skills, or topics absent from the candidate "
        "evidence below. If evidence is thin, use intent cs_fundamentals with "
        "source null."
    )

    def __init__(self, fallback: Optional[HeuristicPlanner] = None) -> None:
        self.fallback = fallback or HeuristicPlanner()

    def _prompt(self, state: InterviewState, context: CandidateContext) -> str:
        return (
            f"{self.SYSTEM_INSTRUCTIONS}\n\n"
            f"CANDIDATE EVIDENCE:\n{context.model_dump_json(indent=1)}\n\n"
            f"INTERVIEW STATE:\n{state.model_dump_json(indent=1)}\n\n"
            "JSON plan:"
        )

    def _parse(self, raw: str) -> QuestionPlan:
        text = (raw or "").strip()
        if text.startswith("```"):
            text = text.strip("`")
            if text.lower().startswith("json"):
                text = text[4:]
            text = text.strip()
        return QuestionPlan.model_validate(json.loads(text))

    def _grounded(self, plan: QuestionPlan, context: CandidateContext) -> bool:
        """Reject plans pointing at evidence the candidate doesn't have."""
        names = {p.name.lower() for p in context.projects}
        repos = {p.repository.lower() for p in context.projects if p.repository}
        topics = {t.topic.lower() for t in context.leetcode_topics}
        skills = {s.lower() for s in context.resume_skills}
        if plan.intent in ("github_project", "github_implementation", "project_followup"):
            return plan.project_name.lower() in names or plan.project_name.lower() in repos
        if plan.intent == "leetcode_topic":
            return plan.topic.lower() in topics
        if plan.intent == "resume_followup":
            return (plan.topic.lower() in skills or plan.topic.lower() in topics or
                    any(plan.topic.lower() in p.lower() for p in names))
        return True  # cs_fundamentals/clarification/behavioral/coding/optimization need no grounding

    def plan(self, state: InterviewState, context: CandidateContext) -> QuestionPlan:
        try:
            from generate import generate_text  # lazy: keeps RAG import offline-safe

            plan = self._parse(generate_text(self._prompt(state, context)))
            if not self._grounded(plan, context):
                raise ValueError("plan references evidence absent from candidate context")
            return plan
        except Exception:
            fallback = self.fallback.plan(state, context)
            if not fallback.reason:
                fallback.reason = "LLM plan unavailable or invalid; heuristic fallback."
            elif "fallback" not in fallback.reason.lower():
                fallback.reason += " (LLM plan unavailable or invalid; heuristic fallback.)"
            return fallback


def plan_next_question(
    state: InterviewState,
    context: CandidateContext,
    strategy: str = "heuristic",
) -> QuestionPlan:
    """Entry point. strategy: 'heuristic' (deterministic) or 'llm' (validated)."""
    if not isinstance(state, InterviewState) or not isinstance(context, CandidateContext):
        raise PlannerError("plan_next_question requires InterviewState and CandidateContext.")
    strategy = (strategy or "heuristic").strip().lower()
    if strategy == "llm":
        return LLMPlanner().plan(state, context)
    if strategy == "heuristic":
        return HeuristicPlanner().plan(state, context)
    raise PlannerError(f"Unknown planner strategy {strategy!r}. Expected 'heuristic' or 'llm'.")

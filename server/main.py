"""asses.ai — FastAPI backend.

Exposes the CLI-only interview logic (Agents/intro.py, Agents/concept.py,
Scrapper/scrap.py, Models/*) as HTTP endpoints so the React frontend can:

  POST /api/session            -> create an interview session
  POST /api/resume/upload      -> upload PDF/DOCX, extract GitHub + LeetCode
  POST /api/interview/chat     -> send a candidate message, get AI reply
  GET  /api/interview/history/{session_id}
  POST /api/interview/end      -> AI feedback summary + close session
  POST /api/transcribe         -> AssemblyAI transcription (optional)
  GET  /api/health, GET /

Session state lives in Redis (falls back to in-memory store when Redis is
unreachable so the UI still works for local development).
"""

import json
import asyncio
import os
import tempfile
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

try:
    from redis_global import redis_client
    try:
        redis_client.ping()
        _REDIS_OK = type(redis_client).__name__ != "_MemoryRedis"
    except Exception:
        _REDIS_OK = False
        redis_client = None
except Exception as e:  # redis not installed / docker down
    print(f"[warn] Redis unavailable ({e}), using in-memory store.")
    redis_client = None
    _REDIS_OK = False

from generate import (
    generate_text_safe,
    get_active_provider,
    is_ai_configured,
    is_gemini_configured,
    is_groq_configured,
    save_key_to_env,
    set_google_api_key,
    set_groq_api_key,
    test_connection,
)

try:
    from Scrapper.scrap import (
        extract_links_from_pdf,
        extract_links_from_text,
        extract_resume_text,
        extract_text_from_docx,
        get_github_details,
        get_github_entries,
        get_leetcode_details,
        leetcode_details,
        repository_details,
    )
except ImportError:  # allow running as `uvicorn main:app` and `python main.py`
    from Scrapper.scrap import (  # type: ignore
        extract_links_from_pdf,
        extract_links_from_text,
        extract_resume_text,
        extract_text_from_docx,
        get_github_details,
        get_github_entries,
        get_leetcode_details,
        leetcode_details,
        repository_details,
    )

# ---------------------------------------------------------------- store ----
_mem_store: dict = {}  # fallback when Redis is down


def _store_get(key: str):
    if _REDIS_OK and redis_client is not None:
        val = redis_client.get(key)
        return val
    return _mem_store.get(key)


def _store_set(key: str, value: str, ex: int | None = None):
    if _REDIS_OK and redis_client is not None:
        redis_client.set(key, value, ex=ex)
    else:
        _mem_store[key] = value


def _store_incr(key: str) -> int:
    if _REDIS_OK and redis_client is not None:
        return int(redis_client.incr(key))
    _mem_store[key] = int(_mem_store.get(key, 0)) + 1
    return int(_mem_store[key])


def _session_key(session_id: str) -> str:
    return f"session:{session_id}"


def _history_key(session_id: str) -> str:
    return f"interview:history:{session_id}"


def load_session(session_id: str) -> dict:
    raw = _store_get(_session_key(session_id))
    if not raw:
        raise HTTPException(status_code=404, detail="Unknown session_id. Create one via POST /api/session.")
    return json.loads(raw)


def save_session(session_id: str, data: dict):
    _store_set(_session_key(session_id), json.dumps(data))


def load_history(session_id: str) -> list:
    raw = _store_get(_history_key(session_id))
    return json.loads(raw) if raw else []


def save_history(session_id: str, history: list):
    _store_set(_history_key(session_id), json.dumps(history))


# ------------------------------------------------------------------ app ----
app = FastAPI(title="asses.ai API", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # dev-friendly; tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------- models ----
class CreateSessionResponse(BaseModel):
    session_id: str
    created_at: float


class ChatRequest(BaseModel):
    session_id: str
    message: str = Field(min_length=1, max_length=8000)
    phase: str = Field(default="intro", pattern="^(intro|technical)$")


class ChatResponse(BaseModel):
    reply: str
    done: bool = False
    phase: str
    mock: bool = False  # True when GOOGLE_API_KEY missing and fallback used
    turn: int


class EndRequest(BaseModel):
    session_id: str


class ApiKeyRequest(BaseModel):
    groq_api_key: str | None = Field(default=None, max_length=200)
    google_api_key: str | None = Field(default=None, max_length=200)
    assemblyai_api_key: str | None = Field(default=None, max_length=200)
    validate_key: bool = True  # make one tiny live call to verify the key


from RAG.planner import (
    HeuristicPlanner,
    context_from_profile,
    plan_next_question,
)
from RAG.prompts import (
    build_intro_prompt,
    build_technical_prompt,
    derive_interview_state,
    format_evidence,
)
from RAG.retrieval import retrieve_candidate_context
from RAG.index import connect_redis as _connect_redis, has_vector_search as _has_vector_search
from RAG.adaptive import (
    adaptive_prompt_block,
    assess_answer,
    closing_plan,
    load_adaptive_state,
    record_asked,
    to_planner_state,
    update_after_answer,
)


# -------------------------------------------------------------- prompts ----
# Prompt text lives in RAG/prompts.py (single-purpose builders). The legacy
# _build_prompt was replaced by the planner -> retrieval -> context pipeline
# below; intro wording is preserved verbatim in prompts.INTRO_SYSTEM.


def _technical_turn(session_id: str, session: dict, history: list, user_msg: str):
    """One adaptive technical interview turn.

    Candidate message -> assess previous answer -> update adaptive state
    (difficulty/coverage/fingerprints/budget, persisted in session["adaptive"])
    -> question plan -> candidate-aware retrieval -> evidence prompt ->
    Gemini -> reply. Every adaptive/RAG step degrades gracefully, so the
    turn always completes with the response contract intact.
    Returns (reply, was_mock).
    """
    profile = session.get("profile") or {}
    context = context_from_profile(profile)
    plans = session.get("question_plans") or []
    adaptive = load_adaptive_state(session)

    # 1. Assess the just-given answer against the last asked question.
    # First technical turn has no prior question -> no assessment yet.
    last_ai = next((m.get("content", "") for m in reversed(history)
                    if isinstance(m, dict) and m.get("role") == "ai"), "")
    last_topic = plans[-1].get("topic", "") if plans and isinstance(plans[-1], dict) else ""
    assessment = None
    if last_ai.strip():
        assess_mode = (os.getenv("RAG_ASSESS_MODE", "llm") or "llm").strip().lower()
        try:
            assessment = assess_answer(question=last_ai, answer=user_msg,
                                       topic=last_topic, mode=assess_mode)
        except Exception as e:
            print(f"[rag] assessment failed ({e}); continuing without it.")
            assessment = None
        try:
            adaptive = update_after_answer(adaptive, assessment, topic=last_topic) \
                if assessment is not None else adaptive
        except Exception as e:
            print(f"[rag] state update failed ({e}).")
    else:
        adaptive.technical_turns += 1
        if adaptive.technical_turns >= adaptive.max_turns:
            adaptive.should_wrap_up = True

    # 2. Plan: forced close at budget, otherwise planner from adaptive state.
    state = to_planner_state(adaptive)
    if adaptive.should_wrap_up:
        plan = closing_plan()
    else:
        strategy = (os.getenv("RAG_PLANNER_STRATEGY", "heuristic") or "heuristic").strip().lower()
        try:
            plan = plan_next_question(state, context, strategy=strategy)
        except Exception as e:
            print(f"[rag] planner failed ({e}); heuristic fallback.")
            try:
                plan = HeuristicPlanner().plan(state, context)
            except Exception as e2:
                print(f"[rag] heuristic planner also failed ({e2}); emergency closing/cs fallback.")
                from RAG.planner import QuestionPlan
                plan = closing_plan() if state.turns_taken >= 1 else QuestionPlan(
                    intent="cs_fundamentals",
                    topic="OOPs",
                    source=None,
                    difficulty="basic",
                    question_style="conceptual",
                    retrieval_query="object oriented programming fundamentals",
                    reason="Emergency fallback plan.",
                )

    evidence = []
    try:
        args = plan.retrieval_args()
        evidence = retrieve_candidate_context(
            session_id, plan.retrieval_query,
            source=args["source"], chunk_type=args["chunk_type"],
            project_name=args["project_name"], topic=args["topic"],
        )
    except Exception as e:
        # Vector DB down / no key / empty index -> general questions, no crash.
        print(f"[rag] retrieval degraded, continuing without evidence: {e}")
        evidence = []

    covered = [str(t) for t in state.covered_topics]
    extra = adaptive_prompt_block(adaptive, assessment)
    prompt = build_technical_prompt(plan, format_evidence(evidence), history, user_msg, covered, extra)
    reply, was_mock = generate_text_safe(prompt)

    # 3. Persist: plan (coverage), question fingerprint (repetition guard),
    # current topic/project, and the adaptive state itself.
    plans.append({
        "intent": plan.intent,
        "topic": plan.topic,
        "project_name": plan.project_name,
        "difficulty": plan.difficulty,
        "t": time.time(),
    })
    session["question_plans"] = plans[-50:]
    if plan.intent in ("project_followup", "github_implementation") and plan.project_name:
        marker = f"{plan.project_name.strip().lower()}:followup"
        if marker not in adaptive.topics_covered:
            adaptive.topics_covered.append(marker)
    adaptive.current_topic = plan.topic
    adaptive.current_project = plan.project_name or ""
    try:
        record_asked(adaptive, reply)
    except Exception as e:
        print(f"[rag] fingerprint failed ({e}).")
    try:
        session["adaptive"] = adaptive.model_dump(mode="json")
    except Exception as e:
        print(f"[rag] adaptive persist failed ({e}).")
    save_session(session_id, session)
    return reply, was_mock


# --------------------------------------------------------------- routes ----
@app.get("/")
def root():
    return {
        "name": "asses.ai API",
        "version": "0.2.0",
        "docs": "/docs",
        "health": "/api/health",
        "endpoints": [
            "POST /api/session",
            "POST /api/resume/upload",
            "POST /api/interview/chat",
            "GET /api/interview/history/{session_id}",
            "POST /api/interview/end",
            "POST /api/transcribe",
            "POST /api/settings/key",
        ],
    }


@app.get("/api/health")
def health():
    return {
        "ok": True,
        "redis": "up" if _REDIS_OK else "memory-fallback",
        "vector_search": _vector_status(),
        "groq_configured": is_groq_configured(),
        "gemini_configured": is_gemini_configured(),
        "ai_configured": is_ai_configured(),
        "ai_provider": get_active_provider(),
        "assemblyai_configured": bool(os.getenv("ASSEMBLYAI_API_KEY")),
        "time": time.time(),
    }


_VECTOR_STATUS_CACHE = {"at": 0.0, "value": None}
_VECTOR_STATUS_TTL = 60.0


def _vector_status():
    """Cheap cached vector-search status for /api/health. Never raises."""
    now = time.time()
    cached = _VECTOR_STATUS_CACHE["value"]
    if cached is not None and now - _VECTOR_STATUS_CACHE["at"] < _VECTOR_STATUS_TTL:
        return cached
    try:
        store_mode = os.getenv("RAG_STORE_MODE", "").strip().lower()
        if store_mode == "memory":
            status = {"status": "ready", "backend": "memory"}
        else:
            from RAG.index import RagConfig

            cfg = RagConfig.from_env()
            client = _connect_redis(cfg)
            if client is None:
                status = {"status": "unavailable", "reason": "Redis unreachable."}
            elif _has_vector_search(client, cfg.redis_host, cfg.redis_port):
                status = {"status": "ready"}
            else:
                status = {"status": "unsupported",
                          "reason": "Redis has no RediSearch module (plain Redis?). "
                                    "Use the redis/redis-stack image — see docker-compose.yml."}
    except Exception as e:
        status = {"status": "unavailable", "reason": f"{type(e).__name__}: {e}"}
    _VECTOR_STATUS_CACHE["at"] = now
    _VECTOR_STATUS_CACHE["value"] = status
    return status


@app.post("/api/settings/key")
def connect_api_key(req: ApiKeyRequest):
    """Connect API keys at runtime (no restart needed) and persist to server/.env.

    - groq_api_key: verified with one tiny live call when validate=True.
    - google_api_key: verified with one tiny live call when validate=True.
    - assemblyai_api_key: stored for /api/transcribe (optional).
    """
    if not req.groq_api_key and not req.google_api_key and not req.assemblyai_api_key:
        raise HTTPException(
            status_code=400,
            detail="Provide groq_api_key, google_api_key, and/or assemblyai_api_key.",
        )
    result: dict = {"ok": True}

    if req.assemblyai_api_key:
        akey = req.assemblyai_api_key.strip()
        if not akey:
            raise HTTPException(status_code=400, detail="Empty AssemblyAI key.")
        os.environ["ASSEMBLYAI_API_KEY"] = akey
        save_key_to_env("ASSEMBLYAI_API_KEY", akey)
        result["assemblyai_configured"] = True

    if req.groq_api_key:
        try:
            info = set_groq_api_key(req.groq_api_key, persist=True)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        if req.validate_key:
            try:
                probe = test_connection()
                result["key_valid"] = True
                result["probe"] = (probe or "")[:50]
            except Exception as e:
                # Key saved but rejected by Groq — tell the UI plainly.
                raise HTTPException(status_code=400, detail=f"Key saved but Groq rejected it: {e}")
        else:
            result["key_valid"] = None
        result["groq_configured"] = True
        result["ai_configured"] = True
        result["provider"] = "groq"
        result["model"] = info.get("model")

    if req.google_api_key:
        try:
            info = set_google_api_key(req.google_api_key, persist=True)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        if req.validate_key:
            try:
                probe = test_connection()
                result["key_valid"] = True
                result["probe"] = (probe or "")[:50]
            except Exception as e:
                # Key saved but rejected by Google — tell the UI plainly.
                raise HTTPException(status_code=400, detail=f"Key saved but Google rejected it: {e}")
        else:
            result["key_valid"] = None
        result["gemini_configured"] = True
        result["ai_configured"] = True
        result["model"] = info.get("model")

    return result


@app.post("/api/session", response_model=CreateSessionResponse)
def create_session():
    session_id = uuid.uuid4().hex[:12]
    save_session(session_id, {"created_at": time.time(), "profile": {}, "status": "created"})
    save_history(session_id, [])
    return {"session_id": session_id, "created_at": time.time()}


@app.get("/api/profile/{session_id}")
def get_profile(session_id: str):
    session = load_session(session_id)
    return {"session_id": session_id, "profile": session.get("profile", {}), "status": session.get("status")}


def _enrich_profile(tmp_path: str, github_url: str | None, leetcode_username: str | None):
    """Fetch GitHub/LeetCode enrichment for a resume file.

    Blocking (uses requests) — always called via asyncio.to_thread.
    Never raises: every failure is captured in `warnings` so the upload
    endpoint returns 200 with partial data instead of 500.
    """
    warnings: list[str] = []
    github_entries: list[dict] = []  # [{"url": ..., "details": [desc, readme, language]}]
    leetcode_raw: str = ""
    extra_links: list[str] = []
    links_found: list[str] = []
    resume_text: str = ""

    # 0. Full resume text (fast, local) — feeds RAG ingestion later.
    try:
        resume_text = extract_resume_text(tmp_path)
    except Exception as e:
        warnings.append(f"Could not extract resume text: {e}")

    # 1. Local link extraction (fast, no network) — always reported.
    try:
        _, ext = os.path.splitext(tmp_path)
        if ext.lower() == ".pdf":
            links_found = extract_links_from_pdf(tmp_path)
        else:
            links_found = extract_links_from_text(extract_text_from_docx(tmp_path))
    except Exception as e:
        warnings.append(f"Could not extract links from resume: {e}")

    # 2. GitHub enrichment — per-repo failures are isolated in scrap.py.
    try:
        github_entries = get_github_entries(tmp_path) or []
        failed = sum(1 for e in github_entries if not e.get("details"))
        if failed:
            warnings.append(
                f"{failed} GitHub repo(s) unreachable (private, deleted, or network issue) — skipped."
            )
        if not github_entries and any("github.com" in (l or "") for l in links_found):
            warnings.append(
                "GitHub API unreachable right now (timeout/rate limit). "
                "Resume links were saved; retry analysis later or add a GITHUB_TOKEN in server/.env."
            )
    except Exception as e:
        warnings.append(f"GitHub enrichment failed: {e}")

    if github_url:
        extra_links.append(github_url)
        try:
            github_entries = github_entries + [{"url": github_url, "details": repository_details(github_url)}]
        except Exception as e:
            warnings.append(f"Could not fetch {github_url}: {e}")

    # 3. LeetCode enrichment.
    try:
        leetcode_raw = get_leetcode_details(tmp_path) or ""
    except Exception as e:
        leetcode_raw = f"Error: {e}"
    if leetcode_username:
        try:
            leetcode_raw = leetcode_details(leetcode_username)
        except Exception as e:
            leetcode_raw = f"Error: {e}"
    if isinstance(leetcode_raw, str) and leetcode_raw.startswith("Error:"):
        warnings.append(f"LeetCode lookup failed ({leetcode_raw[:150]}).")

    return github_entries, leetcode_raw, extra_links, warnings, links_found, resume_text


def _rag_prerequisites() -> tuple[bool, str]:
    """Check RAG ingestion prerequisites quickly: AI key + reachable Redis.

    Never raises. Used to decide whether to schedule background ingestion.
    """
    try:
        if not is_ai_configured():
            return False, "No AI key set (set GROQ_API_KEY or GOOGLE_API_KEY)"
        store_mode = os.getenv("RAG_STORE_MODE", "").strip().lower()
        if store_mode == "memory":
            return True, ""
        from RAG.index import connect_redis

        client = connect_redis()
        if client is None:
            return False, "Redis unreachable"
        try:
            client.close()
        except Exception:
            pass
        return True, ""
    except Exception as e:
        return False, f"RAG unavailable: {e}"


async def _run_ingestion(session_id: str, profile: dict, *, store=None, provider=None, config=None):
    """Background RAG ingestion for one uploaded profile. Never raises.

    On completion writes profile["rag"] status into the session so a later
    GET /api/profile shows ready/partial/failed with counts and warnings.
    """
    try:
        from RAG.ingestion import ingest_candidate_profile

        stats = await asyncio.to_thread(
            ingest_candidate_profile, session_id, profile,
            store=store, provider=provider, config=config, replace=True,
        )
    except Exception as e:
        stats = {
            "status": "failed", "documents_created": 0, "documents_updated": 0,
            "sources": {"resume": 0, "github": 0, "leetcode": 0},
            "warnings": [f"Ingestion crashed: {e}"],
        }
    try:
        session = load_session(session_id)
        prof = session.get("profile") or {}
        total = int(stats.get("documents_created", 0)) + int(stats.get("documents_updated", 0))
        prof["rag"] = {
            "status": stats.get("status", "failed"),
            "chunks": total,
            "sources": stats.get("sources", {}),
            "warnings": (stats.get("warnings") or [])[:5],
        }
        session["profile"] = prof
        save_session(session_id, session)
    except Exception as e:
        print(f"[rag] could not persist ingestion status for {session_id}: {e}")


@app.post("/api/resume/upload")
async def upload_resume(
    file: UploadFile = File(...),
    session_id: str | None = Form(default=None),
    leetcode_username: str | None = Form(default=None),
    github_url: str | None = Form(default=None),
):
    """Accept PDF/DOCX resume, extract GitHub repos + LeetCode stats (RAG context)."""
    name = (file.filename or "").lower()
    if not (name.endswith(".pdf") or name.endswith(".docx")):
        raise HTTPException(status_code=400, detail="Only .pdf and .docx resumes are supported.")

    sid = session_id or uuid.uuid4().hex[:12]
    try:
        load_session(sid)
    except HTTPException:
        save_session(sid, {"created_at": time.time(), "profile": {}, "status": "created"})
        save_history(sid, [])

    suffix = ".pdf" if name.endswith(".pdf") else ".docx"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        if len(content) > 10 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="File too large (max 10MB).")
        tmp.write(content)
        tmp_path = tmp.name

    try:
        # Blocking network enrichment runs in a thread (keeps event loop responsive).
        # _enrich_profile never raises — failures come back as warnings.
        github_entries, leetcode_raw, extra_links, warnings, links_found, resume_text = await asyncio.to_thread(
            _enrich_profile, tmp_path, github_url, leetcode_username
        )
        # Normalise entries: [{url, details:[description, readme, language]}].
        # url/language are additive — the frontend only reads description/readme.
        repos = []
        for entry in github_entries:
            item = (entry or {}).get("details") or []
            if isinstance(item, (list, tuple)) and len(item) >= 1:
                desc = (item[0] or "")[:500]
                p1 = str(item[1] if len(item) > 1 else "")
                p2 = str(item[2] if len(item) > 2 else "")
                # If item[2] is long or contains markdown/newlines, it's the readme and item[1] is language
                if len(item) > 2 and (len(p2) > 100 or "\n" in p2 or p2.startswith("#")):
                    readme = p2[:4000]
                    language = p1[:50]
                else:
                    readme = p1[:4000]
                    language = p2[:50]
                repos.append({
                    "url": (entry or {}).get("url") or "",
                    "description": desc,
                    "readme": readme,
                    "language": language,
                })
            elif isinstance(item, dict):
                repos.append(item)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    # Try to parse leetcode JSON string into structured stats for the UI
    leetcode_stats = None
    try:
        parsed = json.loads(leetcode_raw) if isinstance(leetcode_raw, str) and leetcode_raw.strip().startswith("{") else None
        leetcode_stats = parsed
    except Exception:
        leetcode_stats = None

    profile = {
        "filename": file.filename,
        "github": repos,
        "leetcode_raw": leetcode_raw if isinstance(leetcode_raw, str) else json.dumps(leetcode_raw),
        "leetcode_stats": leetcode_stats,
        "extra_links": extra_links,
        "links_found": links_found,
        "resume_text": resume_text,
        "uploaded_at": time.time(),
    }

    # RAG ingestion runs in the background: the upload response must never
    # wait on embeddings, and must never fail because of them. The frontend
    # ignores the additive "rag" field (no UI change required).
    rag_ok, rag_reason = _rag_prerequisites()
    if rag_ok:
        profile["rag"] = {"status": "pending", "chunks": 0}
    else:
        profile["rag"] = {"status": "degraded", "chunks": 0, "reason": rag_reason}
        warnings.append(f"RAG ingestion skipped: {rag_reason}.")

    session = load_session(sid)
    session["profile"] = profile
    session["status"] = "profile_ready"
    save_session(sid, session)

    if rag_ok:
        asyncio.create_task(_run_ingestion(sid, dict(profile)))

    return {"session_id": sid, "profile": profile, "warnings": warnings}


@app.post("/api/interview/chat", response_model=ChatResponse)
def interview_chat(req: ChatRequest):
    phase = req.phase if req.phase in ("intro", "technical") else "intro"
    session = load_session(req.session_id)
    history = load_history(req.session_id)

    history.append({"role": "user", "content": req.message, "phase": phase, "t": time.time()})
    if phase == "intro":
        # Intro is deliberately RAG-free: pure conversation, unchanged behavior.
        prompt = build_intro_prompt(history, req.message)
        reply, was_mock = generate_text_safe(prompt)
    else:
        reply, was_mock = _technical_turn(req.session_id, session, history, req.message)

    done = False
    if phase == "intro" and "[INTRO_DONE]" in reply:
        done = True
        reply = reply.replace("[INTRO_DONE]", "").strip()
    if phase == "technical" and "[INTERVIEW_DONE]" in reply:
        done = True
        reply = reply.replace("[INTERVIEW_DONE]", "").strip()

    history.append({"role": "ai", "content": reply, "phase": phase, "t": time.time()})
    save_history(req.session_id, history)

    # Mirror legacy Redis key layout so Agents/*.py scripts keep working
    try:
        turn = _store_incr(f"interview:turn:{req.session_id}")
        _store_set(f"technical_interview:chat:{req.session_id}:{turn}", json.dumps({"user": "AI", "message": reply}))
    except Exception:
        turn = len(history)

    if session.get("status") != "finished":
        session["status"] = f"in_{phase}"
        save_session(req.session_id, session)

    return {"reply": reply, "done": done, "phase": phase, "mock": was_mock, "turn": turn if isinstance(turn, int) else len(history)}


@app.get("/api/interview/history/{session_id}")
def interview_history(session_id: str):
    load_session(session_id)  # 404 if unknown
    return {"session_id": session_id, "history": load_history(session_id)}


@app.post("/api/interview/end")
def end_interview(req: EndRequest):
    session = load_session(req.session_id)
    history = load_history(req.session_id)
    try:
        # Structured evaluation: transcript + plans + assessments + profile
        # evidence -> validated report -> frontend-compatible feedback string.
        from RAG.evaluation import (
            build_evaluation_inputs,
            evaluate_interview,
            render_feedback,
        )

        report, used_llm = evaluate_interview(build_evaluation_inputs(session, history))
        feedback = render_feedback(report)
        report_dict = report.model_dump(mode="json")
        was_mock = not used_llm
    except Exception as e:
        # Evaluation layer itself broken: legacy plain-text path, never a 500.
        print(f"[eval] structured evaluation failed ({e}); legacy fallback.")
        transcript = "\n".join(f"{m['role']}: {m['content']}" for m in history[-60:]) or "(no conversation yet)"
        prompt = (
            "You are a senior hiring manager. Given this mock interview transcript, "
            "write concise feedback: 1) strengths 2) gaps 3) score /10 for communication, "
            "CS fundamentals, problem-solving 4) one next step. Keep under 250 words.\n\n"
            + transcript
        )
        feedback, was_mock = generate_text_safe(prompt, fallback_prefix="Feedback")
        report_dict = {}
    session["status"] = "finished"
    session["feedback"] = feedback
    session["report"] = report_dict  # additive: frontend reads feedback only
    save_session(req.session_id, session)
    return {"session_id": req.session_id, "feedback": feedback, "mock": was_mock,
            "turns": len(history), "report": report_dict}


@app.post("/api/transcribe")
async def transcribe(file: UploadFile = File(...)):
    """Proxy audio uploads to AssemblyAI (kept server-side so the key stays secret)."""
    from dotenv import load_dotenv

    load_dotenv()
    api_key = os.getenv("ASSEMBLYAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=503, detail="ASSEMBLYAI_API_KEY not configured. Use browser mic instead.")

    import requests as _requests

    suffix = Path(file.filename or "audio.webm").suffix or ".webm"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    try:
        # Lazy import avoids hard dependency at startup
        from Models.speech_to_text import transcribe_audio

        text = transcribe_audio(tmp_path)
        return {"text": text}
    except Exception as e:
        # Fallback: direct upload path if Models helper fails on webm
        raise HTTPException(status_code=502, detail=f"Transcription failed: {e}")
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

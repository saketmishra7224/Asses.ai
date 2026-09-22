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
    is_gemini_configured,
    save_key_to_env,
    set_google_api_key,
    test_connection,
)

try:
    from Scrapper.scrap import (
        extract_links_from_pdf,
        extract_links_from_text,
        extract_text_from_docx,
        get_github_details,
        get_leetcode_details,
        leetcode_details,
        repository_details,
    )
except ImportError:  # allow running as `uvicorn main:app` and `python main.py`
    from Scrapper.scrap import (  # type: ignore
        extract_links_from_pdf,
        extract_links_from_text,
        extract_text_from_docx,
        get_github_details,
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
    google_api_key: str | None = Field(default=None, max_length=200)
    assemblyai_api_key: str | None = Field(default=None, max_length=200)
    validate_key: bool = True  # make one tiny live call to verify the Google key


# -------------------------------------------------------------- prompts ----
INTRO_SYSTEM = """You are an AI Interviewer conducting a mock technical interview.
Goal: simulate a real-world interview, starting with a professional introduction
and short conversation (at least ~2 minutes) before technical questions.
Structure: 1) Introduce yourself (30-45s) 2) Ask for self-introduction
3) Conversational follow-up (1 min+).
Reply with ONE natural sentence/paragraph per turn based on context.
When the intro has reached a logical stopping point, include the exact token
[INTRO_DONE] at the end of your reply."""

TECHNICAL_SYSTEM = """You are an AI interviewer running the technical round.
Flow: 1) 2-3 core CS questions (OOPs, DBMS), conceptual not definitional.
2) Algorithmic coding questions tailored to the candidate profile below.
3) If brute-force given, push for optimisation ("time complexity?", "better DS?").
4) If optimal, acknowledge and close positively.
Be engaging, professional, slightly challenging. One question/message per turn.
When the interview should end, include the exact token [INTERVIEW_DONE].
Candidate context (intro transcript + RAG profile):"""


def _build_prompt(phase: str, session: dict, history: list, user_msg: str) -> str:
    profile = session.get("profile") or {}
    intro_transcript = "\n".join(
        f"{m['role']}: {m['content']}" for m in history[-30:]
    )
    if phase == "intro":
        return (
            INTRO_SYSTEM
            + f"\n\nConversation so far:\n{intro_transcript}\nUser: {user_msg}\nAI:"
        )
    coding_topics = profile.get("leetcode_raw", "no leetcode data")
    github = profile.get("github", [])
    return (
        TECHNICAL_SYSTEM
        + f"\nLeetCode stats: {coding_topics}\nGitHub projects: {json.dumps(github)[:4000]}"
        + f"\n\nConversation so far:\n{intro_transcript}\nUser: {user_msg}\nAI:"
    )


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
        "gemini_configured": is_gemini_configured(),
        "assemblyai_configured": bool(os.getenv("ASSEMBLYAI_API_KEY")),
        "time": time.time(),
    }


@app.post("/api/settings/key")
def connect_api_key(req: ApiKeyRequest):
    """Connect API keys at runtime (no restart needed) and persist to server/.env.

    - google_api_key: verified with one tiny live call when validate=True.
    - assemblyai_api_key: stored for /api/transcribe (optional).
    """
    if not req.google_api_key and not req.assemblyai_api_key:
        raise HTTPException(status_code=400, detail="Provide google_api_key and/or assemblyai_api_key.")
    result: dict = {"ok": True}

    if req.assemblyai_api_key:
        akey = req.assemblyai_api_key.strip()
        if not akey:
            raise HTTPException(status_code=400, detail="Empty AssemblyAI key.")
        os.environ["ASSEMBLYAI_API_KEY"] = akey
        save_key_to_env("ASSEMBLYAI_API_KEY", akey)
        result["assemblyai_configured"] = True

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
    github_data: list = []
    leetcode_raw: str = ""
    extra_links: list[str] = []
    links_found: list[str] = []

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
        github_data = get_github_details(tmp_path) or []
        failed = sum(1 for item in github_data if not item)
        if failed:
            warnings.append(
                f"{failed} GitHub repo(s) unreachable (private, deleted, or network issue) — skipped."
            )
        if not github_data and any("github.com" in (l or "") for l in links_found):
            warnings.append(
                "GitHub API unreachable right now (timeout/rate limit). "
                "Resume links were saved; retry analysis later or add a GITHUB_TOKEN in server/.env."
            )
    except Exception as e:
        warnings.append(f"GitHub enrichment failed: {e}")

    if github_url:
        extra_links.append(github_url)
        try:
            github_data = github_data + [repository_details(github_url)]
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

    return github_data, leetcode_raw, extra_links, warnings, links_found


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
        github_data, leetcode_raw, extra_links, warnings, links_found = await asyncio.to_thread(
            _enrich_profile, tmp_path, github_url, leetcode_username
        )
        # Normalise github_data: scrap.py returns list of lists
        repos = []
        for item in github_data:
            if isinstance(item, (list, tuple)) and len(item) >= 1:
                repos.append({
                    "description": (item[0] or "")[:500],
                    "readme": (item[1] if len(item) > 1 else "")[:4000],
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
        "uploaded_at": time.time(),
    }
    session = load_session(sid)
    session["profile"] = profile
    session["status"] = "profile_ready"
    save_session(sid, session)

    return {"session_id": sid, "profile": profile, "warnings": warnings}


@app.post("/api/interview/chat", response_model=ChatResponse)
def interview_chat(req: ChatRequest):
    phase = req.phase if req.phase in ("intro", "technical") else "intro"
    session = load_session(req.session_id)
    history = load_history(req.session_id)

    history.append({"role": "user", "content": req.message, "phase": phase, "t": time.time()})
    prompt = _build_prompt(phase, session, history, req.message)
    reply, was_mock = generate_text_safe(prompt)

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
    transcript = "\n".join(f"{m['role']}: {m['content']}" for m in history[-60:]) or "(no conversation yet)"
    prompt = (
        "You are a senior hiring manager. Given this mock interview transcript, "
        "write concise feedback: 1) strengths 2) gaps 3) score /10 for communication, "
        "CS fundamentals, problem-solving 4) one next step. Keep under 250 words.\n\n"
        + transcript
    )
    feedback, was_mock = generate_text_safe(prompt, fallback_prefix="Feedback")
    session["status"] = "finished"
    session["feedback"] = feedback
    save_session(req.session_id, session)
    return {"session_id": req.session_id, "feedback": feedback, "mock": was_mock, "turns": len(history)}


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

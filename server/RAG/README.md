# Candidate RAG — vector storage layer (Phase 1)

Candidate-isolated semantic storage for interview context, built on the
**existing Redis Stack** deployment (RediSearch + HASH storage). No new
database, no new vendor: embeddings come from the Gemini embedding API using
the existing `GOOGLE_API_KEY`.

> Phase 1 covers the **storage layer only**: data models, embedding
> abstraction, the Redis vector index, and tests. Document ingestion
> (resume/GitHub/LeetCode → chunks) and interview wiring land in later phases.
> Nothing here changes interview behavior or the frontend.

## How it works

```
RagDocument (candidate_id, source, text, metadata)
        │  embed via EmbeddingProvider (768-d float vector)
        ▼
Redis HASH  rag:{candidate_id}:{document_id}
  {candidate_id, source, chunk_type, project_name, repository,
   topic, language, text, document_id, created_at, updated_at,
   embedding: <FLOAT32 blob>}
        │  indexed by idx:candidate_chunks (HNSW, cosine)
        ▼
retrieve(session→candidate_id, query, k, filters)  →  top-k chunks, this candidate only
```

* **Write path** (later phases): chunk → `embed_texts()` → `store.upsert(doc, vec)` (pipelined `HSET` + `EXPIRE`).
* **Read path**: embed the query → `FT.SEARCH` KNN with a **mandatory**
  `@candidate_id:{id}` TAG filter → results re-checked in Python
  (`check_candidates_match`). No code outside `RAG/index.py` may issue a
  vector query.
* **Full text**: `text` is also a `TEXT` field, so `SearchFilters(text=...)`
  does full-text pre-filtering combined with KNN.

## Environment variables

| Var | Default | Meaning |
|---|---|---|
| `RAG_INDEX_NAME` | `idx:candidate_chunks` | RediSearch index name |
| `RAG_KEY_PREFIX` | `rag:` | Key namespace prefix |
| `RAG_EMBEDDING_PROVIDER` | `local` | Provider selector (`local` deterministic hash, or `gemini`) |
| `RAG_EMBEDDING_MODEL` | `models/gemini-embedding-001` | Embedding model (when using `gemini` provider) |
| `RAG_EMBEDDING_DIM` | `768` | Vector dim. Default: 768 |
| `RAG_TOP_K` | `4` | Default retrieval depth |
| `RAG_CHUNK_CHARS` | `1200` | Max chars per chunk window |
| `RAG_CHUNK_OVERLAP` | `150` | Overlap between consecutive windows |
| `RAG_TTL_SECONDS` | `604800` (7 d) | Chunk key expiry. `0` disables |
| `RAG_HNSW_M` / `RAG_HNSW_EF_CONSTRUCTION` / `RAG_HNSW_EF_RUNTIME` | `16` / `200` / `10` | HNSW tuning |
| `REDIS_HOST` / `REDIS_PORT` | `localhost` / `6379` | Reused from existing config |
| `GROQ_API_KEY` | — | Primary LLM key for Groq |
| `GOOGLE_API_KEY` | — | Optional key for Gemini LLM / embeddings |

## Initialization

Idempotent — safe on every startup; an existing index is verified, never
blindly recreated:

```python
from RAG import ensure_index

status = ensure_index()  # uses env config + Gemini provider dim
# {'ok': True, 'status': 'created'|'exists', 'index': ..., 'dimension': 768}
# {'ok': False, 'status': 'unavailable'|'unsupported'|'dimension_mismatch', 'reason': ...}
```

If Redis is down you get `status: 'unavailable'` (session/history keep
working). If an index exists with a different dim you get
`status: 'dimension_mismatch'` with recovery instructions — it is **never**
auto-dropped. To rebuild deliberately:

```
FT.DROPINDEX idx:candidate_chunks DD
```

then call `ensure_index()` again. Inspect data in Redis Insight
(`docker compose up redis` → `localhost:8001`).

## Redis schema

Index `idx:candidate_chunks`, `ON HASH`, `PREFIX 1 rag:`:

| Field | Type | Purpose |
|---|---|---|
| `candidate_id` | TAG | **isolation filter (mandatory)** |
| `source` | TAG | `resume` \| `github` \| `leetcode` \| `summary` |
| `chunk_type` | TAG | `section` \| `project` \| `profile` \| `summary` |
| `project_name` | TAG | repo/project scoping |
| `repository` | TAG | `owner/repo` scoping |
| `topic` | TAG | skill topic (e.g. `two-pointers`) |
| `language` | TAG | programming language |
| `section` | TAG | resume section (`education`/`skills`/…) or README heading |
| `skill` | TAG | single-skill label where applicable (e.g. LeetCode topic slug) |
| `category` | TAG | difficulty bucket (`fundamental`/`intermediate`/`advanced`) |
| `document_id` | TAG | stable doc id within a candidate |
| `text` | TEXT | full-text search + prompt evidence |
| `created_at` / `updated_at` | NUMERIC | recency / debugging |
| `problems_solved` | NUMERIC | LeetCode topic solve count |
| `embedding` | VECTOR HNSW FLOAT32 dim + COSINE | similarity search |

Keys: `rag:{candidate_id}:{document_id}` (+ advisory meta key
`rag:_meta:{index}` recording the build dim for mismatch detection).
Untouched legacy keys: `session:*`, `interview:history:*`,
`interview:turn:*`, `technical_interview:*`.

## Ingestion (Phase 2)

`RAG/ingestion.py` turns an upload profile into chunks — pure functions, no
network (fetching still lives in `Scrapper/scrap.py`):

* **Resume** (`chunk_resume_text`): heading detection (known keywords, ALL-CAPS,
  markdown `#`) → sections (education/skills/experience/projects/achievements/
  certifications/summary/other) → sliding windows (`RAG_CHUNK_CHARS` /
  `RAG_CHUNK_OVERLAP`) on long sections. IDs `resume|<section>|<n>`.
* **GitHub** (`chunk_github_repo`): per repo one `project` overview doc, one
  `readme` doc per README heading window, one `technology` doc when the
  primary language is known. Empty/unreachable repos yield nothing. Bounded:
  README/project level only, never full code indexing.
* **LeetCode** (`chunk_leetcode_stats`): one `topic` statement per solved tag
  (`"Candidate has solved 53 DFS problems…"`, zero-count tags skipped) with
  `topic`/`category`/`problems_solved` metadata, plus one `profile` totals
  summary. The original structured stats stay in the session profile for the UI.

`ingest_candidate_profile(candidate_id, profile, replace=False)` orchestrates
chunk → batch-embed (per-document fallback) → upsert and returns
`{status, documents_created, documents_updated, sources, warnings}`.
Deterministic IDs make re-ingestion update instead of duplicate; `replace=True`
purges the candidate's old chunks first (used on fresh re-upload).

**Upload wiring** (`POST /api/resume/upload`): after the profile is saved,
ingestion runs as a background task and writes `profile["rag"] =
{status: pending|ready|partial|degraded, chunks, sources, warnings}` back into
the session. The response shape is unchanged apart from the additive `rag`
field; if prerequisites (API key, Redis) are missing the upload still returns
200 with `rag.status: "degraded"` and a warning. Interview/chat/history
behavior is untouched.

## Retrieval (Phase 3)

`RAG/retrieval.py` — `retrieve_candidate_context(candidate_id, query,
top_k=8, source=None, chunk_type=None, project_name=None, topic=None,
mode="hybrid", diversity=True)`:

* **Vector path**: query embedded with the same provider used at ingestion →
  KNN with the mandatory `@candidate_id` TAG filter (+ metadata pre-filters).
* **Keyword path**: question tokenized (`extract_terms`: keeps `Next.js`/`C++`,
  drops stopwords) → lexical `FT.SEARCH` on the `text` field, equally
  candidate-scoped at the query level. Never global-search-then-filter.
* **Hybrid (default)**: Reciprocal Rank Fusion (`RRF_K=60`) over both ranked
  lists, deduped by chunk key — transparent and debuggable, no reranker.
* **Diversity**: greedy per-source cap (`max_per_source=3`) so 8 README windows
  can't crowd out resume/LeetCode evidence; rank 1 always kept.
* **Empty**: unknown candidate or zero matches → `[]`; `format_context([])`
  renders an explicit no-evidence marker so the interviewer falls back to
  general CS questions instead of fabricating candidate facts.
* **Score convention**: `SearchResult.score` is always a distance
  (lower = more relevant) across `vector` / `keyword` / `hybrid` origins;
  `origin` records the producing path. Provider/store outages raise
  (`EmbeddingError` / `RagUnavailableError`) rather than returning fake empties.

## Planner (Phase 4)

`RAG/planner.py` decides *what to ask next* as a validated `QuestionPlan`
(intent, topic, source, project, difficulty, question_style,
`retrieval_query`, reason) — never natural-language questions.

* **Intents** (10): resume/project follow-ups, github project & implementation,
  leetcode topic, cs_fundamentals, coding_problem, optimization,
  clarification, behavioral.
* **Difficulty** (basic/intermediate/advanced) blends previous difficulty,
  answer signal (correct/struggling/unknown), evidence depth (high solve
  counts lift the floor, never dictate alone), and phase/turns.
* **No repetition**: covered topics are tracked in `InterviewState`; repeats
  become why/how follow-ups, and callers record `"<name>:followup"` so the
  planner advances. Struggling candidates get a clarification lifeline.
* **Grounding invariants** (Pydantic): github intents require a real
  `project_name`; `source=None` only for cs_fundamentals/clarification/
  behavioral; `retrieval_query` always non-empty for the Phase 3 retriever.
* **Two strategies**: `HeuristicPlanner` (deterministic, offline) is the
  default and the safety net; `LLMPlanner` asks Gemini for a plan, then
  rejects malformed JSON, schema violations, and hallucinated projects —
  falling back to the heuristic every time. `plan.retrieval_args()` maps a
  plan straight onto `retrieve_candidate_context` filters.
* `context_from_profile()` bridges the session-profile shape (both legacy
  and current) into planner evidence; empty/odd input yields an empty
  context → cs_fundamentals fallback instead of invented facts.

## Interview integration (Phase 5)

Each technical turn in `POST /api/interview/chat` runs:
message → session/history → `derive_interview_state` (coverage from stored
`question_plans`, struggling signal from explicit phrases only) →
`plan_next_question` (`RAG_PLANNER_STRATEGY`, default heuristic) →
`retrieve_candidate_context(candidate_id=session_id, plan.retrieval_query,
plan filters)` → `build_technical_prompt` → Gemini → reply stored, plan
appended to `session["question_plans"]` (additive key, history untouched).

* Intro turns are RAG-free (wording preserved verbatim).
* Evidence is injected as `[CANDIDATE EVIDENCE]` with `Source:`/`Project:`/
  `Topic:`/`Language:` labels plus grounding rules (never invent, never
  assume cross-project tech, topic-level LeetCode only) and a Level 1→5
  depth ladder. Empty retrieval yields an explicit marker → general CS.
* Any RAG failure (no key, Redis down, empty index) degrades to evidence-free
  prompting; the response contract `{reply, done, phase, mock, turn}` and the
  `[INTRO_DONE]`/`[INTERVIEW_DONE]` tokens are unchanged.

## Adaptive questioning (Phase 6)

`RAG/adaptive.py` runs a lightweight state machine around every technical
turn, persisted in the existing session as `session["adaptive"]` (no second
database):

assess previous answer -> update state -> plan from state -> ask -> fingerprint

* **Assessment** (`assess_answer`): LLM-as-judge returning strict
  `AnswerAssessment{correctness, understanding, needs_followup,
  recommended_action, rationale}` — factual fields only, no psychological
  claims. Malformed output, missing key, or `RAG_ASSESS_MODE=heuristic`
  falls back to a transparent heuristic. First turn has no prior question,
  so nothing is assessed.
* **Transitions**: difficulty moves exactly one step (correct up / incorrect
  down / partial holds); topics are marked covered after sufficient
  non-struggling depth (unknown advances only to avoid stalls); struggling
  stays on topic.
* **Repetition**: asked questions stored as normalized SHA fingerprints
  (+ recent raw text injected into the prompt as do-not-repeat); follow-up
  plans record `<name>:followup` markers so the planner advances.
* **Limits**: `RAG_MAX_TURNS` (default 20) technical turns; at budget the
  planner is bypassed with a behavioral closing plan and the prompt is told
  to wrap up — follow-up loops cannot run forever.
* Every step degrades independently; the turn always completes.

## Structured evaluation (Phase 7)

`RAG/evaluation.py` upgrades `POST /api/interview/end` from free-text
feedback to a validated `EvaluationReport` rendered into the same
frontend-compatible `feedback` string (plus an additive `report` object;
the old response keys are unchanged).

* **Inputs** (`build_evaluation_inputs`): bounded transcript, questions,
  answers, topics covered, difficulty progression, per-turn answer
  assessments (`session["adaptive"].assessment_history`), weak/strong
  topics, and profile evidence (projects, LeetCode topics).
* **Output**: six dimensions (communication, CS fundamentals, problem
  solving, technical depth, project understanding, answer quality), each
  `{score 0-10|null, strengths, gaps, evidence[]}`, plus strengths,
  weaknesses, topics_to_improve, recommended_practice, overall_score.
* **Evidence rule**: `[Interview]` items must quote/paraphrase the transcript;
  `[Profile]` items are resume/GitHub/LeetCode facts — never presented as
  demonstrated skill. Dimensions without evidence score null ("n/a").
* **Robustness**: malformed JSON, schema violations, or missing key fall
  back to `heuristic_report` (factual coverage only, no guessed scores);
  if the evaluation module itself fails, the endpoint keeps a legacy
  plain-text path. The endpoint never 500s on evaluation errors.

## Security model

* **Untrusted input**: resume text, READMEs, and candidate answers are
  attacker-influenced data. They are sanitized at prompt-build time
  (`sanitize_untrusted`: control chars stripped, structural markers like
  `[INTERVIEW_DONE]`/`[CANDIDATE EVIDENCE]` neutralized to `(…)`), while
  stored data stays raw for audit. The system prompt explicitly labels the
  evidence block as untrusted data that can never issue instructions.
* **Isolation**: see below; enforced at the RediSearch query level, not in
  Python post-filtering.
* **Secrets**: API keys live only in `server/.env` (git-ignored and
  docker-ignored, never baked into images, never returned by any endpoint —
  `/api/settings/key` returns status flags only). Logs (`print` to stdout,
  captured by Docker/uvicorn) never include key material or full profiles.
* **No auth layer**: anyone holding a session ID can read that session.
  Session IDs are 48-bit random hex; do not expose them in URLs/logs beyond
  the owning browser. Multi-tenant hardening (auth, per-user scoping) is
  out of scope — see Known limitations.

## Candidate isolation

1. Every chunk stores `candidate_id` (TAG) and lives under a
   `rag:{candidate_id}:*` key.
2. `VectorStore.search()` **requires** `candidate_id` (raises otherwise) and
   always injects `@candidate_id:{id}` into the RediSearch query.
3. Results pass `check_candidates_match()` before return (defense in depth).
4. `server/tests/test_rag_isolation.py` seeds two candidates and asserts zero
   cross-contamination, including adversarial near-duplicate text.

## Graceful degradation

* No Redis / no RediSearch module → status dicts with `ok: False`, or
  `RagUnavailableError` at call time. Session creation, resume upload
  parsing, chat, and history are unaffected (they don't import this path yet).
* No API key → `EmbeddingError` only when embedding is attempted; imports and
  index-metadata operations never need a key.
* `InMemoryVectorStore` mirrors filtering semantics for unit tests and offline
  dev (select via `get_vector_store(mode="memory")`; production uses `"redis"`).

## Tests

```powershell
cd server
python -m pytest tests/ -v
```

* `test_rag_embeddings.py` — interface compliance, validation, config (no network).
* `test_rag_index.py` — init/idempotency, insert, filters, isolation, empty
  index on the in-memory backend (always runs).
* `test_rag_isolation.py` — cross-candidate leakage suite (in-memory always;
  same suite against real Redis Stack when reachable, else skipped).
* `test_rag_redis.py` — live Redis integration (index init, insert, search,
  count, delete); **skipped** automatically without a reachable Redis Stack.

To run the live suite: `docker compose up redis`, then
`RAG_TEST_REDIS=1 python -m pytest tests/ -v`.

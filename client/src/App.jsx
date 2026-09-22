/* eslint-disable react/prop-types */
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { api } from './api.js'

const PHASES = [
  { id: 'intro', label: 'Introduction', hint: 'Self-introduction & warm-up (~2 min)' },
  { id: 'technical', label: 'Technical', hint: 'CS fundamentals + coding (RAG on your profile)' },
]

const STARTERS = {
  intro: 'Hi! I am excited for this mock interview. Let me introduce myself briefly.',
  technical: 'I am ready for the technical round. Please ask your first question.',
}

function Badge({ tone, children }) {
  const tones = {
    green: 'bg-emerald-500/15 text-emerald-300 border-emerald-500/30',
    red: 'bg-red-500/15 text-red-300 border-red-500/30',
    amber: 'bg-amber-500/15 text-amber-300 border-amber-500/30',
    slate: 'bg-slate-500/15 text-slate-300 border-slate-500/30',
    violet: 'bg-violet-500/15 text-violet-300 border-violet-500/30',
  }
  return (
    <span className={`inline-flex items-center gap-1.5 rounded-full border px-2.5 py-0.5 text-xs font-medium ${tones[tone] || tones.slate}`}>
      {children}
    </span>
  )
}

function StepDot({ n, active, done, label }) {
  return (
    <div className="flex items-center gap-2">
      <div
        className={`flex h-7 w-7 items-center justify-center rounded-full text-xs font-bold ${
          done ? 'bg-emerald-500 text-white' : active ? 'bg-violet-500 text-white' : 'bg-slate-700 text-slate-300'
        }`}
      >
        {done ? '✓' : n}
      </div>
      <span className={`text-sm ${active ? 'text-white font-semibold' : 'text-slate-400'}`}>{label}</span>
    </div>
  )
}

function fmtTime(sec) {
  const m = Math.floor(sec / 60)
  const s = sec % 60
  return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`
}

function speak(text, enabled) {
  if (!enabled || !('speechSynthesis' in window)) return
  try {
    window.speechSynthesis.cancel()
    const u = new SpeechSynthesisUtterance(text.slice(0, 600))
    u.rate = 1
    u.lang = 'en-US'
    window.speechSynthesis.speak(u)
  } catch {
    /* ignore */
  }
}

export default function App() {
  const [health, setHealth] = useState(null)
  const [healthErr, setHealthErr] = useState('')
  const [sessionId, setSessionId] = useState('')
  const [step, setStep] = useState(1)
  const [phase, setPhase] = useState('intro')

  const [file, setFile] = useState(null)
  const [leetcodeUsername, setLeetcodeUsername] = useState('')
  const [githubUrl, setGithubUrl] = useState('')
  const [uploading, setUploading] = useState(false)
  const [uploadErr, setUploadErr] = useState('')
  const [uploadWarnings, setUploadWarnings] = useState([])
  const [profile, setProfile] = useState(null)

  const [history, setHistory] = useState([]) // {role, content, phase}
  const [input, setInput] = useState('')
  const [sending, setSending] = useState(false)
  const [chatErr, setChatErr] = useState('')
  const [phaseDone, setPhaseDone] = useState(false)
  const [voiceOn, setVoiceOn] = useState(true)
  const [listening, setListening] = useState(false)
  const [startedAt, setStartedAt] = useState(null)
  const [elapsed, setElapsed] = useState(0)

  const [feedback, setFeedback] = useState('')
  const [ending, setEnding] = useState(false)
  const [mockMode, setMockMode] = useState(false)

  const [keyOpen, setKeyOpen] = useState(false)
  const [apiKey, setApiKey] = useState('')
  const [assemblyKey, setAssemblyKey] = useState('')
  const [keySaving, setKeySaving] = useState(false)
  const [keyMsg, setKeyMsg] = useState('')
  const [keyErr, setKeyErr] = useState('')

  const chatRef = useRef(null)
  const recogRef = useRef(null)

  // ---- boot: health + session ----
  useEffect(() => {
    api.health().then(setHealth).catch((e) => setHealthErr(String(e.message || e)))
    api.createSession().then((s) => setSessionId(s.session_id)).catch((e) => setChatErr(`Could not create session: ${e.message}`))
  }, [])

  useEffect(() => {
    if (!health) return
    if (!health.gemini_configured) {
      setMockMode(true)
      setKeyOpen(true) // prompt for the key when backend has none
    }
  }, [health])

  // ---- timer ----
  useEffect(() => {
    if (!startedAt) return
    const t = setInterval(() => setElapsed(Math.floor((Date.now() - startedAt) / 1000)), 1000)
    return () => clearInterval(t)
  }, [startedAt])

  // ---- autoscroll ----
  useEffect(() => {
    const el = chatRef.current
    if (el) el.scrollTop = el.scrollHeight
  }, [history, sending])

  const leetcodeGroups = useMemo(() => {
    const stats = profile?.leetcode_stats
    const counts = stats?.tagProblemCounts || stats?.matchedUser?.tagProblemCounts || null
    if (!counts) return null
    return ['fundamental', 'intermediate', 'advanced'].map((k) => ({ key: k, items: counts[k] || [] })).filter((g) => g.items.length)
  }, [profile])

  const upload = useCallback(async () => {
    if (!file) {
      setUploadErr('Choose a PDF or DOCX resume first.')
      return
    }
    setUploading(true)
    setUploadErr('')
    setUploadWarnings([])
    try {
      const res = await api.uploadResume({ file, sessionId, leetcodeUsername: leetcodeUsername.trim(), githubUrl: githubUrl.trim() })
      if (res.session_id) setSessionId(res.session_id)
      setProfile(res.profile)
      if (res.warnings?.length) setUploadWarnings(res.warnings)
    } catch (e) {
      setUploadErr(String(e.message || e))
    } finally {
      setUploading(false)
    }
  }, [file, sessionId, leetcodeUsername, githubUrl])

  const send = useCallback(async (text) => {
    const msg = (text ?? input).trim()
    if (!msg || sending) return
    if (!sessionId) {
      setChatErr('No session yet — wait a moment and retry.')
      return
    }
    setChatErr('')
    setPhaseDone(false)
    setSending(true)
    setHistory((h) => [...h, { role: 'user', content: msg, phase }])
    setInput('')
    if (!startedAt) setStartedAt(Date.now())
    try {
      const res = await api.chat({ sessionId, message: msg, phase })
      if (res.mock) setMockMode(true)
      setHistory((h) => [...h, { role: 'ai', content: res.reply, phase }])
      speak(res.reply, voiceOn)
      if (res.done) setPhaseDone(true)
    } catch (e) {
      setChatErr(String(e.message || e))
    } finally {
      setSending(false)
    }
  }, [input, sending, sessionId, phase, startedAt, voiceOn])

  const toggleMic = useCallback(() => {
    const SR = window.SpeechRecognition || window.webkitSpeechRecognition
    if (!SR) {
      setChatErr('Browser speech recognition is not available here — type your answer instead. (Chrome + HTTPS works best.)')
      return
    }
    if (listening) {
      try { recogRef.current?.stop() } catch { /* ignore */ }
      setListening(false)
      return
    }
    try {
      const rec = new SR()
      recogRef.current = rec
      rec.lang = 'en-US'
      rec.interimResults = false
      rec.onresult = (e) => {
        const text = Array.from(e.results).map((r) => r[0].transcript).join(' ')
        setInput((prev) => (prev ? prev + ' ' + text : text))
      }
      rec.onerror = (e) => {
        setChatErr(`Mic error: ${e.error || 'unknown'}. Check microphone permission / HTTPS.`)
        setListening(false)
      }
      rec.onend = () => setListening(false)
      rec.start()
      setListening(true)
      setChatErr('')
    } catch (e) {
      setChatErr(`Could not start mic: ${e.message}`)
    }
  }, [listening])

  const endInterview = useCallback(async () => {
    if (!sessionId) return
    setEnding(true)
    try {
      const res = await api.end(sessionId)
      if (res.mock) setMockMode(true)
      setFeedback(res.feedback)
      const h = await api.history(sessionId).catch(() => null)
      if (h?.history) setHistory(h.history)
      setStep(3)
    } catch (e) {
      setChatErr(String(e.message || e))
    } finally {
      setEnding(false)
    }
  }, [sessionId])

  const resetAll = useCallback(async () => {
    try {
      const s = await api.createSession()
      setSessionId(s.session_id)
    } catch { /* keep old */ }
    setProfile(null)
    setFile(null)
    setUploadWarnings([])
    setHistory([])
    setFeedback('')
    setPhase('intro')
    setStep(1)
    setStartedAt(null)
    setElapsed(0)
    setInput('')
    setPhaseDone(false)
  }, [])

  const connectKey = useCallback(async () => {
    if (!apiKey.trim() && !assemblyKey.trim()) {
      setKeyErr('Paste your Google API key first (find it at Google AI Studio → Get API key).')
      return
    }
    setKeySaving(true)
    setKeyErr('')
    setKeyMsg('')
    try {
      const res = await api.saveApiKey({ googleApiKey: apiKey.trim(), assemblyaiApiKey: assemblyKey.trim(), validateKey: true })
      const h = await api.health().catch(() => null)
      if (h) setHealth(h)
      if (res.gemini_configured) {
        setMockMode(false)
        setApiKey('')
        setKeyMsg('Gemini connected and verified — live AI answers from now on.')
      } else if (res.assemblyai_configured) {
        setAssemblyKey('')
        setKeyMsg('AssemblyAI key saved for voice transcription.')
      }
    } catch (e) {
      setKeyErr(String(e.message || e))
    } finally {
      setKeySaving(false)
    }
  }, [apiKey, assemblyKey])

  const downloadTranscript = useCallback(() => {    const blob = new Blob([JSON.stringify({ session_id: sessionId, phase, profile, history, feedback }, null, 2)], { type: 'application/json' })
    const a = document.createElement('a')
    a.href = URL.createObjectURL(blob)
    a.download = `asses-ai-${sessionId || 'session'}.json`
    a.click()
    URL.revokeObjectURL(a.href)
  }, [sessionId, phase, profile, history, feedback])

  const backendUp = !!health && !healthErr
  const turns = history.length

  return (
    <div className="min-h-screen bg-gradient-to-b from-slate-950 via-slate-900 to-slate-950 text-slate-100">
      {/* ---------- header ---------- */}
      <header className="border-b border-white/10 bg-slate-950/70 backdrop-blur sticky top-0 z-10">
        <div className="mx-auto flex max-w-6xl flex-wrap items-center gap-3 px-4 py-3">
          <div className="flex items-center gap-2">
            <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-violet-600 text-lg font-black">a.</div>
            <div>
              <div className="text-lg font-bold leading-tight">asses.ai</div>
              <div className="text-xs text-slate-400 leading-tight">AI mock interview platform</div>
            </div>
          </div>
          <div className="ml-auto flex flex-wrap items-center gap-2">
            <Badge tone={backendUp ? 'green' : 'red'}>
              <span className={`h-2 w-2 rounded-full ${backendUp ? 'bg-emerald-400' : 'bg-red-400'}`} />
              {backendUp ? 'Backend live' : 'Backend unreachable'}
            </Badge>
            {health && <Badge tone="slate">store: {health.redis}</Badge>}
            {health && (
              <button onClick={() => setKeyOpen((v) => !v)} title="Connect / change API keys">
                <Badge tone={health.gemini_configured ? 'green' : 'amber'}>
                  {health.gemini_configured ? 'Gemini live' : mockMode ? 'Mock AI (no key)' : 'AI'} 🔑
                </Badge>
              </button>
            )}
            {sessionId && <Badge tone="violet">session {sessionId.slice(0, 8)}</Badge>}
          </div>
        </div>
        {healthErr && (
          <div className="mx-auto max-w-6xl px-4 pb-3 text-xs text-red-300">
            API health check failed ({healthErr}). Start the backend: <code className="rounded bg-white/10 px-1">cd server; python main.py</code> or <code className="rounded bg-white/10 px-1">docker compose up</code>.
          </div>
        )}
      </header>

      <main className="mx-auto max-w-6xl px-4 py-6">
        {/* ---------- stepper ---------- */}
        <div className="mb-6 flex flex-wrap items-center gap-x-6 gap-y-2">
          <StepDot n={1} label="Profile & resume" active={step === 1} done={step > 1 || !!profile} />
          <StepDot n={2} label="Live interview" active={step === 2} done={step > 2} />
          <StepDot n={3} label="Feedback & review" active={step === 3} done={false} />
          <div className="ml-auto flex gap-2">
            {[1, 2, 3].map((n) => (
              <button
                key={n}
                onClick={() => setStep(n)}
                className={`rounded-lg px-3 py-1.5 text-xs font-semibold ${step === n ? 'bg-violet-600 text-white' : 'bg-white/5 text-slate-300 hover:bg-white/10'}`}
              >
                Step {n}
              </button>
            ))}
          </div>
        </div>

        {mockMode && (
          <div className="mb-4 rounded-xl border border-amber-500/30 bg-amber-500/10 px-4 py-2.5 text-xs text-amber-200">
            Running in <b>mock-AI mode</b> (no <code>GOOGLE_API_KEY</code> on the backend). The full UI flow works.{' '}
            <button className="underline font-bold" onClick={() => setKeyOpen(true)}>Connect your Gemini key →</button>
          </div>
        )}

        {keyOpen && (
          <div className="mb-4 rounded-2xl border border-violet-500/30 bg-violet-500/5 p-4">
            <div className="flex items-center gap-2">
              <div className="text-sm font-bold">🔑 Connect Google AI key</div>
              {health?.gemini_configured && <Badge tone="green">connected</Badge>}
              <button onClick={() => setKeyOpen(false)} className="ml-auto text-xs text-slate-400 hover:text-white">✕ close</button>
            </div>
            <p className="mt-1 text-xs text-slate-400">
              Get a key at <code className="rounded bg-white/10 px-1">aistudio.google.com → Get API key</code>, paste it below and hit Connect.
              The backend verifies it with one tiny live call and saves it to <code>server/.env</code> (git-ignored) — no restart needed.
            </p>
            <div className="mt-3 grid gap-2 sm:grid-cols-2">
              <div>
                <label className="text-xs font-semibold text-slate-300">Google API key (Gemini)</label>
                <input
                  type="password"
                  value={apiKey}
                  onChange={(e) => setApiKey(e.target.value)}
                  onKeyDown={(e) => { if (e.key === 'Enter') connectKey() }}
                  placeholder="AIza…"
                  autoComplete="off"
                  className="mt-1 w-full rounded-lg border border-white/10 bg-slate-900 px-3 py-2 text-sm font-mono outline-none focus:border-violet-400"
                />
              </div>
              <div>
                <label className="text-xs font-semibold text-slate-300">AssemblyAI key <span className="font-normal text-slate-500">(optional, voice transcription)</span></label>
                <input
                  type="password"
                  value={assemblyKey}
                  onChange={(e) => setAssemblyKey(e.target.value)}
                  onKeyDown={(e) => { if (e.key === 'Enter') connectKey() }}
                  placeholder="(optional)"
                  autoComplete="off"
                  className="mt-1 w-full rounded-lg border border-white/10 bg-slate-900 px-3 py-2 text-sm font-mono outline-none focus:border-violet-400"
                />
              </div>
            </div>
            {keyErr && <div className="mt-2 rounded-lg bg-red-500/10 border border-red-500/30 px-3 py-2 text-xs text-red-300">{keyErr}</div>}
            {keyMsg && <div className="mt-2 rounded-lg bg-emerald-500/10 border border-emerald-500/30 px-3 py-2 text-xs text-emerald-300">{keyMsg}</div>}
            <div className="mt-3 flex gap-2">
              <button onClick={connectKey} disabled={keySaving} className="rounded-xl bg-violet-600 px-4 py-2 text-xs font-bold hover:bg-violet-500 disabled:opacity-50">
                {keySaving ? 'Verifying with Google…' : 'Connect & verify'}
              </button>
              <span className="self-center text-[11px] text-slate-500">Prefer files? Paste the key into <code>server/.env</code> after <code>GOOGLE_API_KEY=</code> and restart the backend.</span>
            </div>
          </div>
        )}

        {/* ================= STEP 1 ================= */}
        {step === 1 && (
          <div className="grid gap-4 lg:grid-cols-5">
            <section className="rounded-2xl border border-white/10 bg-white/5 p-5 lg:col-span-2">
              <h2 className="text-base font-bold">1 · Upload resume</h2>
              <p className="mt-1 text-sm text-slate-400">
                PDF/DOCX is parsed by <code>Scrapper/scrap.py</code>: GitHub repos + README, LeetCode tag stats — the RAG context for your interview.
              </p>
              <label className="mt-4 block rounded-xl border border-dashed border-white/20 bg-slate-900/60 p-4 text-sm hover:border-violet-400 cursor-pointer">
                <input
                  type="file"
                  accept=".pdf,.docx"
                  className="hidden"
                  onChange={(e) => setFile(e.target.files?.[0] || null)}
                />
                <div className="font-semibold">{file ? file.name : 'Click to choose resume.pdf / .docx'}</div>
                <div className="text-xs text-slate-400">{file ? `${(file.size / 1024).toFixed(0)} KB — ready to analyze` : 'Max 10 MB. Links inside are auto-extracted.'}</div>
              </label>
              <div className="mt-3 grid gap-3">
                <div>
                  <label className="text-xs font-semibold text-slate-300">LeetCode username (optional fallback)</label>
                  <input value={leetcodeUsername} onChange={(e) => setLeetcodeUsername(e.target.value)} placeholder="e.g. neetcode" className="mt-1 w-full rounded-lg border border-white/10 bg-slate-900 px-3 py-2 text-sm outline-none focus:border-violet-400" />
                </div>
                <div>
                  <label className="text-xs font-semibold text-slate-300">GitHub repo URL (optional fallback)</label>
                  <input value={githubUrl} onChange={(e) => setGithubUrl(e.target.value)} placeholder="https://github.com/owner/repo" className="mt-1 w-full rounded-lg border border-white/10 bg-slate-900 px-3 py-2 text-sm outline-none focus:border-violet-400" />
                </div>
              </div>
              {uploadErr && <div className="mt-3 rounded-lg bg-red-500/10 border border-red-500/30 px-3 py-2 text-xs text-red-300">{uploadErr}</div>}
              {uploadWarnings.length > 0 && (
                <div className="mt-3 rounded-lg bg-amber-500/10 border border-amber-500/30 px-3 py-2 text-xs text-amber-200 space-y-1">
                  {uploadWarnings.map((w, i) => <div key={i}>⚠ {w}</div>)}
                </div>
              )}
              <button onClick={upload} disabled={uploading} className="mt-4 w-full rounded-xl bg-violet-600 px-4 py-2.5 text-sm font-bold hover:bg-violet-500 disabled:opacity-50">
                {uploading ? 'Analyzing…' : 'Analyze resume'}
              </button>
              <button onClick={() => setStep(2)} className="mt-2 w-full rounded-xl bg-white/5 px-4 py-2.5 text-sm font-semibold text-slate-200 hover:bg-white/10">
                Skip — start interview without resume →
              </button>
            </section>

            <section className="rounded-2xl border border-white/10 bg-white/5 p-5 lg:col-span-3">
              <h2 className="text-base font-bold">Candidate context (RAG)</h2>
              {!profile ? (
                <div className="mt-2 text-sm text-slate-400">
                  No profile yet. Upload a resume to see extracted GitHub projects and LeetCode strengths here — the technical agent uses exactly this data to personalize questions.
                  <ul className="mt-3 list-disc pl-5 space-y-1 text-xs text-slate-500">
                    <li>GitHub: repo description + README via public API</li>
                    <li>LeetCode: tag-wise solved counts via GraphQL</li>
                    <li>Stored per-session in Redis (<code>session:{'{id}'}</code>)</li>
                  </ul>
                </div>
              ) : (
                <div className="mt-3 space-y-4">
                  <div>
                    <div className="text-sm font-bold">GitHub projects ({profile.github?.length || 0})</div>
                    {(profile.github?.length || 0) === 0 && <div className="text-xs text-slate-400">No repo links found in resume. Paste one on the left and re-analyze.</div>}
                    <div className="mt-2 grid gap-2">
                      {(profile.github || []).map((r, i) => (
                        <div key={i} className="rounded-xl border border-white/10 bg-slate-900/60 p-3">
                          <div className="text-sm text-slate-200">{r.description || 'No description'}</div>
                          {r.readme && <div className="mt-1 line-clamp-3 whitespace-pre-wrap text-xs text-slate-400">{r.readme.slice(0, 400)}</div>}
                        </div>
                      ))}
                    </div>
                  </div>
                  <div>
                    <div className="text-sm font-bold">LeetCode strengths</div>
                    {!leetcodeGroups ? (
                      <div className="mt-1 text-xs text-slate-400">No LeetCode data — add a profile link/username. Raw: <code className="text-slate-500">{String(profile.leetcode_raw || '').slice(0, 120)}</code></div>
                    ) : (
                      leetcodeGroups.map((g) => (
                        <div key={g.key} className="mt-2">
                          <div className="text-xs uppercase tracking-wide text-slate-400">{g.key}</div>
                          <div className="mt-1 flex flex-wrap gap-1.5">
                            {g.items.slice(0, 14).map((t) => (
                              <span key={t.tagSlug} className="rounded-full bg-white/5 border border-white/10 px-2.5 py-1 text-xs">
                                {t.tagName} <b className="text-violet-300">{t.problemsSolved}</b>
                              </span>
                            ))}
                          </div>
                        </div>
                      ))
                    )}
                  </div>
                  <button onClick={() => setStep(2)} className="w-full rounded-xl bg-emerald-600 px-4 py-2.5 text-sm font-bold hover:bg-emerald-500">
                    Continue to live interview →
                  </button>
                </div>
              )}
            </section>
          </div>
        )}

        {/* ================= STEP 2 ================= */}
        {step === 2 && (
          <div className="grid gap-4 lg:grid-cols-3">
            <section className="flex flex-col rounded-2xl border border-white/10 bg-white/5 lg:col-span-2 overflow-hidden">
              <div className="flex flex-wrap items-center gap-2 border-b border-white/10 p-3">
                {PHASES.map((p) => (
                  <button
                    key={p.id}
                    onClick={() => setPhase(p.id)}
                    title={p.hint}
                    className={`rounded-lg px-3 py-1.5 text-xs font-bold ${phase === p.id ? 'bg-violet-600 text-white' : 'bg-white/5 text-slate-300 hover:bg-white/10'}`}
                  >
                    {p.label}
                  </button>
                ))}
                <span className="text-xs text-slate-400 hidden sm:inline">{PHASES.find((p) => p.id === phase)?.hint}</span>
                <div className="ml-auto flex items-center gap-2 text-xs">
                  <span className="rounded-lg bg-black/40 px-2 py-1 font-mono text-slate-300">⏱ {fmtTime(elapsed)}</span>
                  <span className="rounded-lg bg-black/40 px-2 py-1 font-mono text-slate-300">{turns} msgs</span>
                </div>
              </div>

              {phaseDone && (
                <div className="border-b border-emerald-500/30 bg-emerald-500/10 px-4 py-2 text-xs text-emerald-200">
                  {phase === 'intro'
                    ? <>Intro round complete. <button className="underline font-bold" onClick={() => setPhase('technical')}>Switch to Technical →</button></>
                    : <>The AI signaled the interview is complete. <button className="underline font-bold" onClick={endInterview}>Get feedback →</button></>}
                </div>
              )}

              <div ref={chatRef} className="h-[46vh] overflow-y-auto p-4 space-y-3 chat-scroll">
                {history.length === 0 && (
                  <div className="rounded-xl border border-white/10 bg-slate-900/60 p-4 text-sm text-slate-300">
                    <div className="font-bold text-white">Welcome to your mock interview.</div>
                    <div className="mt-1 text-slate-400">Flow mirrors <code>Agents/intro.py → concept.py</code>: intro chat, then OOPs/DBMS, then coding tuned to your LeetCode/GitHub profile.</div>
                    <div className="mt-3 flex flex-wrap gap-2">
                      <button onClick={() => send(STARTERS[phase])} className="rounded-lg bg-violet-600 px-3 py-1.5 text-xs font-bold hover:bg-violet-500">Send starter message</button>
                      <button onClick={() => setStep(1)} className="rounded-lg bg-white/5 px-3 py-1.5 text-xs hover:bg-white/10">Add resume context first</button>
                    </div>
                  </div>
                )}
                {history.map((m, i) => (
                  <div key={i} className={`flex ${m.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                    <div className={`max-w-[85%] rounded-2xl px-3.5 py-2.5 text-sm leading-relaxed ${m.role === 'user' ? 'bg-violet-600 text-white rounded-br-md' : 'bg-slate-800 text-slate-100 border border-white/10 rounded-bl-md'}`}>
                      <div className="mb-0.5 text-[10px] uppercase tracking-wide opacity-60">{m.role === 'user' ? 'You' : 'AI interviewer'} · {m.phase}</div>
                      <div className="whitespace-pre-wrap">{m.content}</div>
                    </div>
                  </div>
                ))}
                {sending && (
                  <div className="flex justify-start">
                    <div className="rounded-2xl rounded-bl-md border border-white/10 bg-slate-800 px-4 py-3 text-sm text-slate-300">
                      <span className="typing"><span /><span /><span /></span> thinking…
                    </div>
                  </div>
                )}
              </div>

              {chatErr && <div className="border-t border-red-500/30 bg-red-500/10 px-4 py-2 text-xs text-red-300">{chatErr}</div>}

              <div className="border-t border-white/10 p-3">
                <div className="flex gap-2">
                  <button onClick={toggleMic} title="Dictate with browser mic" className={`rounded-xl px-3 py-2.5 text-sm font-bold ${listening ? 'bg-red-600 animate-pulse' : 'bg-white/5 hover:bg-white/10'}`}>
                    {listening ? '● Stop' : '🎤 Mic'}
                  </button>
                  <button onClick={() => { setVoiceOn((v) => { if (v && 'speechSynthesis' in window) window.speechSynthesis.cancel(); return !v }) }} title="AI voice replies" className={`rounded-xl px-3 py-2.5 text-sm ${voiceOn ? 'bg-white/10' : 'bg-white/5 text-slate-400'}`}>
                    {voiceOn ? '🔊' : '🔇'}
                  </button>
                  <input
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    onKeyDown={(e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send() } }}
                    placeholder={phase === 'intro' ? 'Introduce yourself…' : 'Explain your approach…'}
                    className="flex-1 rounded-xl border border-white/10 bg-slate-900 px-3 py-2.5 text-sm outline-none focus:border-violet-400"
                  />
                  <button onClick={() => send()} disabled={sending || !input.trim()} className="rounded-xl bg-violet-600 px-4 py-2.5 text-sm font-bold hover:bg-violet-500 disabled:opacity-40">
                    Send
                  </button>
                </div>
                <div className="mt-2 flex flex-wrap gap-2">
                  <button onClick={endInterview} disabled={ending} className="rounded-lg bg-emerald-600 px-3 py-1.5 text-xs font-bold hover:bg-emerald-500 disabled:opacity-50">
                    {ending ? 'Generating feedback…' : 'End interview & get feedback'}
                  </button>
                  <span className="text-[11px] text-slate-500 self-center">Enter to send · Mic uses Web Speech API (no server key needed) · 🔊 reads AI replies</span>
                </div>
              </div>
            </section>

            <aside className="space-y-4">
              <div className="rounded-2xl border border-white/10 bg-white/5 p-4">
                <div className="text-sm font-bold">Session</div>
                <div className="mt-1 break-all font-mono text-xs text-slate-400">{sessionId || 'creating…'}</div>
                <div className="mt-2 text-xs text-slate-400">History persists in Redis (<code>interview:history:{'{id}'}</code>). Same keys the CLI agents use.</div>
              </div>
              <div className="rounded-2xl border border-white/10 bg-white/5 p-4">
                <div className="text-sm font-bold">Candidate context</div>
                {!profile ? (
                  <div className="mt-1 text-xs text-slate-400">No resume analyzed. <button className="underline" onClick={() => setStep(1)}>Add one</button> to personalize technical questions.</div>
                ) : (
                  <div className="mt-1 text-xs text-slate-300 space-y-1">
                    <div>📄 {profile.filename}</div>
                    <div>💻 {profile.github?.length || 0} GitHub project(s)</div>
                    <div>🧩 LeetCode: {leetcodeGroups ? leetcodeGroups.map((g) => `${g.key}(${g.items.length})`).join(', ') : 'none found'}</div>
                  </div>
                )}
              </div>
              <div className="rounded-2xl border border-white/10 bg-white/5 p-4 text-xs text-slate-400">
                <div className="font-bold text-slate-200 text-sm">How it maps to the backend</div>
                <ul className="mt-1 space-y-1 list-disc pl-4">
                  <li>Chat → <code>POST /api/interview/chat</code> → <code>generate.py</code> (Gemini 3.6 Flash)</li>
                  <li>Intro prompt from <code>Agents/intro.py</code>, technical from <code>Agents/concept.py</code></li>
                  <li>Voice in-browser; server <code>/api/transcribe</code> (AssemblyAI) optional</li>
                </ul>
              </div>
            </aside>
          </div>
        )}

        {/* ================= STEP 3 ================= */}
        {step === 3 && (
          <div className="grid gap-4 lg:grid-cols-2">
            <section className="rounded-2xl border border-white/10 bg-white/5 p-5">
              <h2 className="text-base font-bold">Feedback</h2>
              {!feedback ? (
                <div className="mt-2 text-sm text-slate-400">No feedback yet — run the interview (Step 2) then click “End interview & get feedback”.</div>
              ) : (
                <div className="mt-2 whitespace-pre-wrap rounded-xl border border-white/10 bg-slate-900/60 p-4 text-sm leading-relaxed">{feedback}</div>
              )}
              <div className="mt-3 flex flex-wrap gap-2">
                <button onClick={downloadTranscript} className="rounded-xl bg-white/5 px-4 py-2 text-xs font-bold hover:bg-white/10">⬇ Download transcript (JSON)</button>
                <button onClick={resetAll} className="rounded-xl bg-violet-600 px-4 py-2 text-xs font-bold hover:bg-violet-500">↺ New session</button>
                <button onClick={() => setStep(2)} className="rounded-xl bg-white/5 px-4 py-2 text-xs hover:bg-white/10">Back to interview</button>
              </div>
            </section>
            <section className="rounded-2xl border border-white/10 bg-white/5 p-5">
              <h2 className="text-base font-bold">Transcript <span className="text-xs font-normal text-slate-400">({turns} messages · {fmtTime(elapsed)})</span></h2>
              <div className="mt-2 max-h-[50vh] overflow-y-auto space-y-2 chat-scroll">
                {history.length === 0 && <div className="text-sm text-slate-400">Empty.</div>}
                {history.map((m, i) => (
                  <div key={i} className="rounded-lg border border-white/10 bg-slate-900/60 px-3 py-2 text-xs">
                    <span className={`font-bold ${m.role === 'user' ? 'text-violet-300' : 'text-emerald-300'}`}>{m.role === 'user' ? 'You' : 'AI'}</span>
                    <span className="text-slate-500"> · {m.phase}</span>
                    <div className="mt-0.5 whitespace-pre-wrap text-slate-200 text-[13px]">{m.content}</div>
                  </div>
                ))}
              </div>
            </section>
          </div>
        )}

        {/* ---------- footer ---------- */}
        <footer className="mt-8 border-t border-white/10 pt-4 text-xs text-slate-500">
          <div className="flex flex-wrap gap-x-6 gap-y-1">
            <span>Backend: <code>POST /api/session · /api/resume/upload · /api/interview/chat · /api/interview/end</code></span>
            <span>Docs: <code>{api.base}/docs</code> when server runs</span>
            <span>Needs: <code>GOOGLE_API_KEY</code> (Gemini) · <code>ASSEMBLYAI_API_KEY</code> (optional STT) · Redis via <code>docker compose up</code></span>
          </div>
        </footer>
      </main>
    </div>
  )
}

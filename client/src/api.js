/**
 * Minimal API client for the asses.ai FastAPI backend.
 * Base URL resolution:
 *  - `VITE_API_URL` env (e.g. http://localhost:8000 for direct calls), or
 *  - '' (same origin) — works with `vite dev` proxy and with nginx `/api` proxy in Docker.
 */
const BASE = (import.meta.env.VITE_API_URL || '').replace(/\/$/, '');

async function handle(res) {
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`;
    try {
      const body = await res.json();
      detail = body.detail || JSON.stringify(body);
    } catch {
      try {
        detail = await res.text();
      } catch { /* ignore */ }
    }
    throw new Error(detail);
  }
  return res.json();
}

export const api = {
  base: BASE || '(same-origin)',
  health() {
    return fetch(`${BASE}/api/health`).then(handle);
  },
  createSession() {
    return fetch(`${BASE}/api/session`, { method: 'POST' }).then(handle);
  },
  getProfile(sessionId) {
    return fetch(`${BASE}/api/profile/${sessionId}`).then(handle);
  },
  uploadResume({ file, sessionId, leetcodeUsername, githubUrl }) {
    const form = new FormData();
    form.append('file', file);
    if (sessionId) form.append('session_id', sessionId);
    if (leetcodeUsername) form.append('leetcode_username', leetcodeUsername);
    if (githubUrl) form.append('github_url', githubUrl);
    return fetch(`${BASE}/api/resume/upload`, { method: 'POST', body: form }).then(handle);
  },
  chat({ sessionId, message, phase }) {
    return fetch(`${BASE}/api/interview/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id: sessionId, message, phase }),
    }).then(handle);
  },
  history(sessionId) {
    return fetch(`${BASE}/api/interview/history/${sessionId}`).then(handle);
  },
  end(sessionId) {
    return fetch(`${BASE}/api/interview/end`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id: sessionId }),
    }).then(handle);
  },
  /** Connect API keys at runtime: verifies the Google key live and persists to server/.env. */
  saveApiKey({ googleApiKey, assemblyaiApiKey, validateKey = true }) {
    return fetch(`${BASE}/api/settings/key`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        google_api_key: googleApiKey || null,
        assemblyai_api_key: assemblyaiApiKey || null,
        validate_key: validateKey,
      }),
    }).then(handle);
  },
};

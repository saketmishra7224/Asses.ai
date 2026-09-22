import os
from pathlib import Path
import requests
from dotenv import load_dotenv

try:
    import google.generativeai as genai
except ImportError:
    genai = None

try:
    from groq import Groq
except ImportError:
    Groq = None

# Always load server/.env (next to this file) so `python main.py` and
# `uvicorn main:app` both pick up keys regardless of current directory.
ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=ENV_PATH)

GROQ_DEFAULT_MODEL = "openai/gpt-oss-120b"
GEMINI_DEFAULT_MODEL = "gemini-3.6-flash"

PREFERRED_GROQ_MODELS = [
    "openai/gpt-oss-120b",
    "qwen/qwen3.8-27b",
    "openai/gpt-oss-20b",
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
]

MODEL_NAME = os.getenv("GEMINI_MODEL", GEMINI_DEFAULT_MODEL)


def get_groq_model() -> str:
    return os.getenv("GROQ_MODEL", GROQ_DEFAULT_MODEL)


def get_gemini_model() -> str:
    return os.getenv("GEMINI_MODEL", GEMINI_DEFAULT_MODEL)


def discover_available_groq_model(key: str) -> str | None:
    """Query Groq API models endpoint and pick the best available chat model."""
    try:
        r = requests.get(
            "https://api.groq.com/openai/v1/models",
            headers={"Authorization": f"Bearer {key}"},
            timeout=10,
        )
        if not r.ok:
            return None
        available_ids = {m["id"] for m in r.json().get("data", [])}
        for pref in PREFERRED_GROQ_MODELS:
            if pref in available_ids:
                return pref
        # Fallback: any model not containing whisper, guard, embedding
        for m_id in available_ids:
            if not any(skip in m_id.lower() for skip in ["whisper", "guard", "embedding"]):
                return m_id
    except Exception as e:
        print(f"[warn] Failed to discover Groq models: {e}")
    return None


# Groq client state
_groq_client = None


def _init_groq_client():
    global _groq_client
    key = (os.getenv("GROQ_API_KEY") or "").strip()
    if key and Groq is not None:
        try:
            _groq_client = Groq(api_key=key, timeout=30)
        except Exception:
            _groq_client = None
    else:
        _groq_client = None


# Gemini client state
model = None


def _init_gemini_model():
    global model
    key = (os.getenv("GOOGLE_API_KEY") or "").strip()
    if key and genai is not None:
        try:
            genai.configure(api_key=key)
            model = genai.GenerativeModel(get_gemini_model())
        except Exception:
            model = None
    else:
        model = None


# Initial configure on module import
_init_groq_client()
_init_gemini_model()


def is_groq_configured() -> bool:
    """True when a Groq API key is present in env or server/.env."""
    return bool((os.getenv("GROQ_API_KEY") or "").strip())


def is_gemini_configured() -> bool:
    """True when a Google API key is present in env or server/.env."""
    return bool((os.getenv("GOOGLE_API_KEY") or "").strip())


def is_ai_configured() -> bool:
    """True when either Groq or Gemini is configured."""
    return is_groq_configured() or is_gemini_configured()


def get_active_provider() -> str:
    """Returns 'groq' if configured, else 'gemini' if configured, else 'none'."""
    if is_groq_configured():
        return "groq"
    if is_gemini_configured():
        return "gemini"
    return "none"


def set_groq_api_key(api_key: str, persist: bool = True, model_name: str | None = None) -> dict:
    """Connect a new Groq key at runtime (no server restart needed).

    Reconfigures the Groq client immediately and, by default, saves the
    key into server/.env so it survives restarts.
    """
    key = (api_key or "").strip()
    if not key:
        raise ValueError("Empty Groq API key.")
    os.environ["GROQ_API_KEY"] = key
    if model_name:
        os.environ["GROQ_MODEL"] = model_name.strip()
    _init_groq_client()
    if persist:
        save_key_to_env("GROQ_API_KEY", key)
        if model_name:
            save_key_to_env("GROQ_MODEL", model_name.strip())
    return {
        "configured": True,
        "persisted": persist,
        "provider": "groq",
        "model": get_groq_model(),
    }


def set_google_api_key(api_key: str, persist: bool = True) -> dict:
    """Connect a new Gemini key at runtime (no server restart needed).

    Reconfigures the Gemini client immediately and, by default, saves the
    key into server/.env so it survives restarts.
    """
    key = (api_key or "").strip()
    if not key:
        raise ValueError("Empty Google API key.")
    os.environ["GOOGLE_API_KEY"] = key
    _init_gemini_model()
    if persist:
        save_key_to_env("GOOGLE_API_KEY", key)
    return {
        "configured": True,
        "persisted": persist,
        "provider": "gemini",
        "model": get_gemini_model(),
    }


def save_key_to_env(name: str, value: str):
    """Upsert a KEY=value line in server/.env, preserving comments/order."""
    lines = []
    if ENV_PATH.exists():
        lines = ENV_PATH.read_text(encoding="utf-8").splitlines()
    found = False
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith(f"{name}=") or stripped.startswith(f"{name} ="):
            lines[i] = f"{name}={value}"
            found = True
    if not found:
        if lines and lines[-1].strip() != "":
            lines.append("")
        lines.append(f"{name}={value}")
    ENV_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


# Error markers meaning "the host itself is unreachable" (DNS failure,
# refused, no route). Retrying via the REST fallback hits the same dead host,
# so we fail fast instead of doubling the delay.
_UNREACHABLE_MARKERS = (
    "Failed to resolve",
    "getaddrinfo failed",
    "Name or service not known",
    "nodename nor servname",
    "Connection refused",
    "No route to host",
    "Network is unreachable",
)
# Error markers meaning "transient" — the REST fallback serves as the one retry.
_TIMEOUT_MARKERS = (
    "timed out",
    "timeout",
    "ReadTimeout",
    "ConnectTimeout",
)
_UNREACHABLE_TYPES = (
    "APIConnectionError",
    "ConnectError",
    "ConnectionError",
    "NameResolutionError",
    "NewConnectionError",
    "MaxRetryError",
)


def _is_unreachable_error(e: Exception) -> bool:
    """True when retrying the same host is pointless (DNS / refused / no route)."""
    msg = str(e)
    if any(m in msg for m in _TIMEOUT_MARKERS):
        return False
    if type(e).__name__ in _UNREACHABLE_TYPES:
        return True
    return any(m in msg for m in _UNREACHABLE_MARKERS)


def _generate_groq(prompt: str, retry_on_not_found: bool = True) -> str:
    key = (os.getenv("GROQ_API_KEY") or "").strip()
    if not key:
        raise RuntimeError("GROQ_API_KEY is not set.")
    model_name = get_groq_model()

    # 1. Try official Groq SDK client
    global _groq_client
    if _groq_client is None and Groq is not None:
        _init_groq_client()

    if _groq_client is not None:
        try:
            resp = _groq_client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            sdk_error = e
            if _is_unreachable_error(e):
                raise RuntimeError(
                    f"Groq API unreachable ({e}). Check network connectivity / DNS "
                    f"for api.groq.com."
                )
            print(f"[warn] Groq SDK call failed ({e}), attempting REST fallback...")

    # 2. HTTP REST fallback to Groq endpoint
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
    }
    r = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers=headers,
        json=payload,
        timeout=30,
    )
    if not r.ok:
        err_text = r.text
        # Check if error is model_not_found / does not exist
        if retry_on_not_found and ("model_not_found" in err_text or "does not exist" in err_text or r.status_code == 404):
            print(f"[info] Groq model '{model_name}' not available on this key. Discovering alternatives...")
            fallback_model = discover_available_groq_model(key)
            if fallback_model and fallback_model != model_name:
                print(f"[info] Auto-switching Groq model to '{fallback_model}'...")
                os.environ["GROQ_MODEL"] = fallback_model
                save_key_to_env("GROQ_MODEL", fallback_model)
                _init_groq_client()
                return _generate_groq(prompt, retry_on_not_found=False)
        raise RuntimeError(f"Groq API call failed (HTTP {r.status_code}): {err_text}")
    data = r.json()
    return data["choices"][0]["message"]["content"]


def _generate_gemini(prompt: str) -> str:
    global model
    if model is None:
        _init_gemini_model()
    if model is None:
        raise RuntimeError("Gemini model is not configured.")
    response = model.generate_content(prompt)
    return response.candidates[0].content.parts[0].text


def test_connection() -> str:
    """Minimal live call to verify the active key works. Raises on failure."""
    provider = get_active_provider()
    if provider == "groq":
        return _generate_groq("Reply with exactly: ok")
    if provider == "gemini":
        return _generate_gemini("Reply with exactly: ok")
    raise RuntimeError("No AI key configured (neither GROQ_API_KEY nor GOOGLE_API_KEY).")


def generate_text(prompt: str) -> str:
    provider = get_active_provider()
    if provider == "groq":
        return _generate_groq(prompt)
    if provider == "gemini":
        return _generate_gemini(prompt)
    raise RuntimeError(
        "GROQ_API_KEY is not set. Paste it into server/.env "
        "or connect it from the app (Connect API Key button)."
    )


def generate_text_safe(prompt: str, fallback_prefix: str = "Mock interviewer"):
    """Wrapper that never crashes the API: falls back to a mock reply."""
    try:
        return generate_text(prompt), False
    except Exception as e:
        print(f"[warn] generate_text failed, using mock fallback: {e}")
        provider = get_active_provider()
        if provider == "groq" and is_groq_configured():
            hint = ("GROQ_API_KEY is set but api.groq.com is unreachable — "
                    "check network connectivity")
        elif provider == "gemini" and is_gemini_configured():
            hint = ("GOOGLE_API_KEY is set but the Gemini API is unreachable — "
                    "check network connectivity")
        else:
            provider_hint = "GROQ_API_KEY" if not is_gemini_configured() else "GOOGLE_API_KEY"
            hint = f"set {provider_hint} for live AI"
        fallback = (
            f"[{fallback_prefix} - mock reply, {hint}]\n"
            "Thanks for sharing that. Could you elaborate a bit more with a concrete "
            "example from your experience? (Prompt received "
            f"{len(prompt)} chars.)"
        )
        return fallback, True

import google.generativeai as genai
from dotenv import load_dotenv
import os
from pathlib import Path

# Always load server/.env (next to this file) so `python main.py` and
# `uvicorn main:app` both pick up keys regardless of current directory.
ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=ENV_PATH)

MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")

# Configure the API key (may be empty on first boot -> mock mode until connected)
genai.configure(api_key=os.getenv("GOOGLE_API_KEY") or "missing")

# Initialize the model
model = genai.GenerativeModel(MODEL_NAME)


def is_gemini_configured():
    """True when a Google API key is present in env or server/.env."""
    return bool(os.getenv("GOOGLE_API_KEY"))


def set_google_api_key(api_key, persist=True):
    """Connect a new key at runtime (no server restart needed).

    Reconfigures the Gemini client immediately and, by default, saves the
    key into server/.env so it survives restarts. Returns the key source.
    """
    key = (api_key or "").strip()
    if not key:
        raise ValueError("Empty API key.")
    os.environ["GOOGLE_API_KEY"] = key
    genai.configure(api_key=key)
    global model
    model = genai.GenerativeModel(MODEL_NAME)
    if persist:
        save_key_to_env("GOOGLE_API_KEY", key)
    return {"configured": True, "persisted": persist, "model": MODEL_NAME}


def save_key_to_env(name, value):
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


def test_connection():
    """Minimal live call to verify the key actually works. Raises on failure."""
    resp = model.generate_content("Reply with exactly: ok")
    return resp.candidates[0].content.parts[0].text


# Function to generate text
def generate_text(prompt):
    if not is_gemini_configured():
        raise RuntimeError(
            "GOOGLE_API_KEY is not set. Paste it into server/.env "
            "or connect it from the app (Connect Gemini button)."
        )
    # Generate text
    response = model.generate_content(prompt)

    return response.candidates[0].content.parts[0].text


def generate_text_safe(prompt, fallback_prefix="Mock interviewer"):
    """Wrapper that never crashes the API: falls back to a mock reply."""
    try:
        return generate_text(prompt), False
    except Exception as e:
        print(f"[warn] generate_text failed, using mock fallback: {e}")
        fallback = (
            f"[{fallback_prefix} - mock reply, set GOOGLE_API_KEY for live AI]\n"
            "Thanks for sharing that. Could you elaborate a bit more with a concrete "
            "example from your experience? (Prompt received "
            f"{len(prompt)} chars.)"
        )
        return fallback, True

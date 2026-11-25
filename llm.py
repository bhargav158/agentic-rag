"""All Gemini calls live here so the rest of the code never touches the SDK."""
import contextvars
import json
import re
import time

import config

_counter = contextvars.ContextVar("llm_counter", default=None)
_client = None


def start_counting():
    """Start counting generate() calls in this context. Returns a 1-item list."""
    c = [0]
    _counter.set(c)
    return c


def _get_client():
    global _client
    if _client is None:
        if not config.GEMINI_API_KEY:
            raise RuntimeError("GEMINI_API_KEY is missing. Copy .env.example to .env and add your key.")
        from google import genai
        from google.genai import types
        # v1beta is required for JSON mode (response_mime_type)
        _client = genai.Client(
            api_key=config.GEMINI_API_KEY,
            http_options=types.HttpOptions(api_version="v1beta"),
        )
    return _client


def _retryable(err):
    """Only retry rate limits and temporary server errors. A 400 will never fix itself."""
    code = getattr(err, "code", None)
    if code in (429, 500, 502, 503, 504):
        return True
    text = str(err)
    return "RESOURCE_EXHAUSTED" in text or "UNAVAILABLE" in text or "429" in text[:40]


def _retry_wait(err, attempt):
    """Seconds to wait after an error. Obeys Google's 'retry in 27s' hint on 429s."""
    m = re.search(r"retry in ([\d.]+)s", str(err)) or re.search(r"'retryDelay': '(\d+)s'", str(err))
    if m:
        return min(float(m.group(1)) + 2, 70)
    return min(5 * (attempt + 1), 30)


def _raw_generate(prompt, json_mode=False):
    cfg = {"response_mime_type": "application/json"} if json_mode else None
    for attempt in range(6):
        try:
            r = _get_client().models.generate_content(
                model=config.GEN_MODEL, contents=prompt, config=cfg
            )
            return r.text or ""
        except Exception as e:
            if attempt == 5 or not _retryable(e):
                raise
            wait = _retry_wait(e, attempt)
            print(f"  [rate limit] waiting {wait:.0f}s then retrying...")
            time.sleep(wait)


def generate(prompt, json_mode=False):
    c = _counter.get()
    if c is not None:
        c[0] += 1
    return _raw_generate(prompt, json_mode)


def _raw_stream(prompt):
    for piece in _get_client().models.generate_content_stream(model=config.GEN_MODEL, contents=prompt):
        if piece.text:
            yield piece.text


def generate_stream(prompt, on_token=None):
    """Like generate(), but calls on_token(text) as the answer is written."""
    if on_token is None:
        return generate(prompt)
    c = _counter.get()
    if c is not None:
        c[0] += 1
    for attempt in range(6):
        parts = []
        try:
            for piece in _raw_stream(prompt):
                parts.append(piece)
                on_token(piece)
            return "".join(parts)
        except Exception as e:
            if parts or attempt == 5 or not _retryable(e):  # never restart a half-shown answer
                raise
            wait = _retry_wait(e, attempt)
            print(f"  [rate limit] waiting {wait:.0f}s then retrying...")
            time.sleep(wait)


def generate_json(prompt, default=None):
    text = generate(prompt, json_mode=True)
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text or "", re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
    return default


def embed(texts, task):
    """task: RETRIEVAL_DOCUMENT for chunks, RETRIEVAL_QUERY for questions.
    Free tier allows ~30,000 embedding tokens per minute, so big jobs are paced."""
    out = []
    size = 10
    for i in range(0, len(texts), size):
        batch = texts[i:i + size]
        for attempt in range(6):
            try:
                r = _get_client().models.embed_content(
                    model=config.EMBED_MODEL, contents=batch, config={"task_type": task}
                )
                out += [e.values for e in r.embeddings]
                break
            except Exception as e:
                if attempt == 5 or not _retryable(e):
                    raise
                wait = _retry_wait(e, attempt)
                print(f"  [rate limit] waiting {wait:.0f}s then retrying...")
                time.sleep(wait)
        if i + size < len(texts):  # pace: stay near 25k tokens/minute
            est_tokens = sum(len(t) for t in batch) / 3.5
            time.sleep(min(est_tokens / 25000 * 60, 20))
    return out

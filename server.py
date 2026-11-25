import asyncio
import base64
import os
import secrets
import time

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from pydantic import BaseModel

import config
import repo_manager
from agent import run_agent
from formatting import parse_answer

app = FastAPI(title="Agentic RAG Code Assistant")
HERE = os.path.dirname(os.path.abspath(__file__))


# ---------- optional password (set APP_PASSWORD) ----------
class PasswordGate:
    """Pure ASGI middleware so it protects both pages and the WebSocket.
    The browser shows its normal login box; the username can be anything."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        protected = scope["type"] in ("http", "websocket") and config.APP_PASSWORD
        if not protected or scope.get("path") == "/health":
            return await self.app(scope, receive, send)
        auth = dict(scope["headers"]).get(b"authorization", b"").decode()
        ok = False
        if auth.lower().startswith("basic "):
            try:
                decoded = base64.b64decode(auth[6:]).decode()
                password = decoded.split(":", 1)[1] if ":" in decoded else ""
                ok = secrets.compare_digest(password.encode(), config.APP_PASSWORD.encode())
            except Exception:
                ok = False
        if ok:
            return await self.app(scope, receive, send)
        if scope["type"] == "http":
            await send({"type": "http.response.start", "status": 401, "headers": [
                (b"www-authenticate", b'Basic realm="Agentic RAG"'), (b"content-type", b"text/plain")]})
            await send({"type": "http.response.body", "body": b"Password required"})
        else:
            await receive()
            await send({"type": "websocket.close", "code": 1008})


app.add_middleware(PasswordGate)

# ---------- optional rate limit (set RATE_LIMIT_PER_MIN) ----------
_hits = {}


def rate_limited(key):
    limit = config.RATE_LIMIT_PER_MIN
    if limit <= 0:
        return False
    now = time.time()
    recent = [t for t in _hits.get(key, []) if now - t < 60]
    if len(recent) >= limit:
        _hits[key] = recent
        return True
    recent.append(now)
    _hits[key] = recent
    return False


def client_key(conn):
    forwarded = conn.headers.get("x-forwarded-for", "").split(",")[0].strip()
    return forwarded or (conn.client.host if conn.client else "unknown")


# ---------- models and helpers ----------
class Ask(BaseModel):
    question: str
    repo: str = ""                 # empty = search every indexed repo
    history: list[dict] = []       # earlier {"q": ..., "a": ...} turns, for follow-up questions


class IndexRequest(BaseModel):
    url: str


def _clean_history(history):
    out = []
    for t in (history or [])[-3:]:
        if isinstance(t, dict) and isinstance(t.get("q"), str) and isinstance(t.get("a"), str):
            out.append({"q": t["q"][:500], "a": t["a"][:500]})
    return out


def _run(question, emit=None, repo=None, history=None):
    res = run_agent(question, emit, repo or None, _clean_history(history))
    res.pop("contexts", None)
    res["parts"] = parse_answer(res["answer"])  # Answer / Explanation / Where, or None
    return res


# ---------- routes ----------
@app.get("/")
def home():
    return FileResponse(os.path.join(HERE, "static", "index.html"))


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/repos")
def repos():
    return repo_manager.list_repos()


@app.post("/index")
def start_index(req: IndexRequest, request: Request):
    if rate_limited(client_key(request)):
        raise HTTPException(429, "Too many requests. Please wait a minute.")
    try:
        return {"name": repo_manager.start_job(req.url)}
    except ValueError as e:
        raise HTTPException(400, str(e))
    except RuntimeError as e:
        raise HTTPException(409, str(e))


@app.get("/index/status/{name}")
def index_status(name: str):
    return repo_manager.JOBS.get(name, {"status": "unknown"})


@app.post("/ask")
def ask(body: Ask, request: Request):
    if rate_limited(client_key(request)):
        raise HTTPException(429, "Too many requests. Please wait a minute.")
    return _run(body.question, None, body.repo, body.history)


@app.websocket("/ws")
async def ws(websocket: WebSocket):
    """Streams every agent step, and the answer as it is written, to the browser."""
    await websocket.accept()
    try:
        while True:
            msg = await websocket.receive_json()
            if rate_limited(client_key(websocket)):
                await websocket.send_json({"step": "error", "message": "Too many requests. Please wait a minute."})
                continue
            question, repo, history = msg["question"], msg.get("repo", ""), msg.get("history", [])
            loop = asyncio.get_running_loop()
            queue = asyncio.Queue()

            def emit(ev):
                loop.call_soon_threadsafe(queue.put_nowait, ev)

            task = asyncio.create_task(asyncio.to_thread(_run, question, emit, repo, history))
            while not (task.done() and queue.empty()):
                try:
                    ev = await asyncio.wait_for(queue.get(), timeout=0.1)
                    await websocket.send_json(ev)
                except asyncio.TimeoutError:
                    continue
            try:
                res = await task
                res.pop("trace", None)
                await websocket.send_json({"step": "final", **res})
            except Exception as e:
                await websocket.send_json({"step": "error", "message": str(e)})
    except WebSocketDisconnect:
        pass

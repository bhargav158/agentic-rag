import asyncio
import os

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from pydantic import BaseModel

import repo_manager
from agent import run_agent
from formatting import parse_answer

app = FastAPI(title="Agentic RAG Code Assistant")
HERE = os.path.dirname(os.path.abspath(__file__))


class Ask(BaseModel):
    question: str
    repo: str = ""         # empty = search every indexed repo


class IndexRequest(BaseModel):
    url: str


def _run(question, emit=None, repo=None):
    res = run_agent(question, emit, repo or None)
    res.pop("contexts", None)
    res["parts"] = parse_answer(res["answer"])  # Answer / Explanation / Where, or None
    return res


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
def start_index(req: IndexRequest):
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
def ask(body: Ask):
    return _run(body.question, None, body.repo)


@app.websocket("/ws")
async def ws(websocket: WebSocket):
    """Streams every agent step to the browser as it happens."""
    await websocket.accept()
    try:
        while True:
            msg = await websocket.receive_json()
            question, repo = msg["question"], msg.get("repo", "")
            loop = asyncio.get_running_loop()
            queue = asyncio.Queue()

            def emit(ev):
                loop.call_soon_threadsafe(queue.put_nowait, ev)

            task = asyncio.create_task(asyncio.to_thread(_run, question, emit, repo))
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

import asyncio
import os

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from pydantic import BaseModel

from agent import run_agent
from naive import run_naive

app = FastAPI(title="Agentic RAG Code Assistant")
HERE = os.path.dirname(os.path.abspath(__file__))


class Ask(BaseModel):
    question: str
    mode: str = "agentic"  # "agentic" or "naive"


def _run(question, mode, emit=None):
    fn = run_naive if mode == "naive" else run_agent
    res = fn(question, emit)
    res.pop("contexts", None)
    return res


@app.get("/")
def index():
    return FileResponse(os.path.join(HERE, "static", "index.html"))


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask")
def ask(body: Ask):
    return _run(body.question, body.mode)


@app.websocket("/ws")
async def ws(websocket: WebSocket):
    """Streams every agent step to the browser as it happens."""
    await websocket.accept()
    try:
        while True:
            msg = await websocket.receive_json()
            question, mode = msg["question"], msg.get("mode", "agentic")
            loop = asyncio.get_running_loop()
            queue = asyncio.Queue()

            def emit(ev):
                loop.call_soon_threadsafe(queue.put_nowait, ev)

            task = asyncio.create_task(asyncio.to_thread(_run, question, mode, emit))
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

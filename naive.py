"""Baseline: retrieve once, answer once. Used to benchmark against the agent."""
import time

import llm
import prompts
from config import TOP_K
from retrieval import retrieve


def run_naive(question, emit=None):
    emit = emit or (lambda e: None)
    counter = llm.start_counting()
    t0 = time.time()
    trace = []

    def log(step, **data):
        ev = {"step": step, "t": round(time.time() - t0, 2), **data}
        trace.append(ev)
        emit(ev)

    chunks = retrieve(question, "vector", TOP_K)
    log("retrieve", attempt=1, strategy="vector", query=question, files=[c["file"] for c in chunks])
    context = "\n\n".join(c["text"] for c in chunks)
    answer = llm.generate(prompts.ANSWER.format(feedback="", context=context, question=question))
    log("generate", chunks=len(chunks))

    return {
        "answer": answer, "verified": None, "attempts": 1,
        "sources": [{"file": c["file"], "start": c["start"], "end": c["end"]} for c in chunks],
        "llm_calls": counter[0], "latency_s": round(time.time() - t0, 2),
        "trace": trace, "contexts": [c["text"] for c in chunks],
    }

"""Offline test: runs the whole pipeline with a FAKE LLM and fake embeddings.
No API key or internet needed. It checks the wiring (loop, rewrite, verify, server, benchmark),
NOT answer quality. Run:  python tests/test_offline.py
"""
import json
import os
import re
import sys
import tempfile
import zlib

os.environ["CHROMA_PATH"] = tempfile.mkdtemp()
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import llm  # noqa: E402

state = {"grade_n": 0, "verify_n": 0}


def fake_generate(prompt, json_mode=False):
    task = prompt.split("\n", 1)[0].replace("TASK:", "").strip()
    if task == "ROUTE":
        state["grade_n"] = 0
        state["verify_n"] = 0
        return json.dumps({"strategy": "hybrid", "reason": "test"})
    if task == "GRADE":
        state["grade_n"] += 1
        return json.dumps({"relevant": [] if state["grade_n"] == 1 else [1, 2]})  # force one rewrite
    if task == "REWRITE":
        return "run_agent retrieve grade loop"
    if task == "ANSWER":
        return "The loop lives in agent.py. [x:1-2]"
    if task == "VERIFY":
        state["verify_n"] += 1
        return json.dumps({"supported": state["verify_n"] > 1, "unsupported_claims": ["made-up claim"]})
    if task == "JUDGE":
        return json.dumps({"correct": True})
    if task == "GENQ":
        return json.dumps({"question": "What does run_agent do?", "answer": "It runs the loop."})
    return "ok"


def fake_embed(texts, task):
    out = []
    for t in texts:
        v = [0.0] * 128
        for tok in re.findall(r"\w+", t.lower()):
            v[zlib.crc32(tok.encode()) % 128] += 1.0
        n = sum(x * x for x in v) ** 0.5 or 1.0
        out.append([x / n for x in v])
    return out


llm._raw_generate = fake_generate
llm.embed = fake_embed

import agent, benchmark, gen_questions, index_repo, naive, retrieval, server  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

n = index_repo.index_repo(root, name="selftest", reset=True)
assert n > 5, "indexing produced too few chunks"
print("index ok:", n, "chunks")

assert retrieval.keyword_search("run_agent", 3), "keyword search returned nothing"
assert retrieval.vector_search("retrieve grade rewrite loop", 3), "vector search returned nothing"
assert retrieval.hybrid_search("run_agent loop", 3), "hybrid search returned nothing"
print("retrieval ok")

res = agent.run_agent("How does the agent loop work?")
steps = [e["step"] for e in res["trace"]]
assert "rewrite" in steps, steps          # first grade was empty -> must rewrite
assert "regenerate" in steps, steps       # first verify failed -> must regenerate
assert res["verified"] is True and res["sources"] and res["llm_calls"] >= 6
print("agent ok:", steps)

nres = naive.run_naive("How does the agent loop work?")
assert nres["llm_calls"] == 1 and nres["answer"]
print("naive ok")

qs = gen_questions.main(n=3, out=os.path.join(os.environ["CHROMA_PATH"], "q.json"))
assert qs
rows = benchmark.run(os.path.join(os.environ["CHROMA_PATH"], "q.json"), limit=2, sleep=0,
                     out_dir=os.path.join(os.environ["CHROMA_PATH"], "results"))
assert len(rows) == 4 and all("error" not in r for r in rows), rows
print("benchmark ok")

client = TestClient(server.app)
assert client.get("/health").json() == {"status": "ok"}
assert client.get("/").status_code == 200
r = client.post("/ask", json={"question": "What does the router do?", "mode": "agentic"})
assert r.status_code == 200 and r.json()["answer"]
with client.websocket_connect("/ws") as sock:
    sock.send_json({"question": "How does retrieval work?", "mode": "agentic"})
    seen = []
    while True:
        ev = sock.receive_json()
        seen.append(ev["step"])
        if ev["step"] in ("final", "error"):
            break
    assert seen[-1] == "final" and "route" in seen and "retrieve" in seen, seen
print("server + websocket ok:", seen)

print("\nALL OFFLINE TESTS PASSED")

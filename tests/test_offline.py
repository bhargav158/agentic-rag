"""Offline test: runs the whole pipeline with a FAKE LLM and fake embeddings.
No API key or internet needed. It checks the wiring (loop, rewrite, verify, repo links,
paste-a-URL jobs, server, benchmark), NOT answer quality. Run:  python tests/test_offline.py
"""
import json
import os
import re
import sys
import tempfile
import time
import zlib

os.environ["CHROMA_PATH"] = tempfile.mkdtemp()
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import llm  # noqa: E402

state = {"grade_n": 0, "verify_n": 0}

STRUCTURED = ("ANSWER: The loop lives in agent.py.\n"
              "EXPLANATION: It retries (tries again) when search is poor.\n"
              "WHERE:\n- agent.py:1-40 - the loop")
OVERVIEW = ("WHAT IT DOES: A code assistant.\nTECH STACK: Python, FastAPI\n"
            "KEY FILES:\n- agent.py - the loop\nKEY TERMS:\n- RAG - search then answer")


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
        return STRUCTURED
    if task == "VERIFY":
        state["verify_n"] += 1
        return json.dumps({"supported": state["verify_n"] > 1, "unsupported_claims": ["made-up claim"]})
    if task == "JUDGE":
        return json.dumps({"correct": True})
    if task == "GENQ":
        return json.dumps({"question": "What does run_agent do?", "answer": "It runs the loop."})
    if task == "OVERVIEW":
        return OVERVIEW
    if task == "CONDENSE":
        return "What calls the retrieve function?"
    return "ok"


def fake_stream(prompt):
    for i in range(0, len(STRUCTURED), 12):  # the answer arrives in small pieces
        yield STRUCTURED[i:i + 12]


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
llm._raw_stream = fake_stream
llm.embed = fake_embed

import agent, benchmark, chunking, config, formatting, gen_questions, index_repo, repo_manager, retrieval, server, store  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

import shutil  # noqa: E402

real_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
root = tempfile.mkdtemp()  # index a clean copy, so stray folders (repos/, venv/) cannot affect the test
for f in os.listdir(real_root):
    if f.endswith((".py", ".md", ".html")):
        shutil.copy(os.path.join(real_root, f), root)
shutil.copytree(os.path.join(real_root, "tests"), os.path.join(root, "tests"), ignore=shutil.ignore_patterns("__pycache__"))
shutil.copytree(os.path.join(real_root, "static"), os.path.join(root, "static"))

# ---- formatting ----
p = formatting.parse_answer("**ANSWER:** Yes.\nEXPLANATION: Because.\nWHERE:\n- a.py:1-2 - x\n- b.py:3-4 - y")
assert p["answer"] == "Yes." and p["explanation"] == "Because." and len(p["where"]) == 2, p
assert formatting.parse_answer("I don't know based on the codebase.") is None
ov = formatting.parse_overview(OVERVIEW)
assert ov["what"] and ov["files"] and ov["terms"], ov
link = formatting.source_dict({"file": "r/a.py", "path": "a.py", "start": 1, "end": 5, "github": "o/r"})["link"]
assert link == "https://github.com/o/r/blob/HEAD/a.py#L1-L5", link
assert formatting.source_dict({"file": "r/a.py", "path": "a.py", "start": 1, "end": 5, "github": ""})["link"] == ""
print("formatting ok")

# ---- function-level chunking ----
PY = ("import os\n\nX = 1\n\n\ndef alpha():\n    return 1\n\n\n# helper for beta\ndef beta():\n    return 2\n\n\n"
      "class Big:\n" + "".join(f"    def m{i}(self):\n" + "".join(f"        x{j} = {j}\n" for j in range(12)) + "\n"
                                  for i in range(10)))
n_lines = len(PY.splitlines())
ch = chunking.chunk_text("a.py", PY)
covered = set()
for s, e, body, syms in ch:
    covered.update(range(s, e + 1))
    assert e - s + 1 <= chunking.MAX_UNIT, (s, e)
assert covered == set(range(1, n_lines + 1)), "some lines were dropped"
all_syms = [x for c in ch for x in c[3]]
assert "alpha" in all_syms and "beta" in all_syms and "Big.m0" in all_syms, all_syms
beta = next(c for c in ch if "beta" in c[3])
assert "# helper for beta" in beta[2], "comment above a function must stay with it"
assert chunking.chunk_text("bad.py", "def broken(:\n  pass\n")  # syntax error falls back to line windows

JS = "import x from 'x';\n\nexport function foo() {\n  return 1;\n}\n\nconst bar = async (a) => {\n  return a;\n};\n" + "// pad\n" * 60
jchunks = chunking.chunk_text("a.js", JS)
assert {"foo", "bar"} <= {s for c in jchunks for s in c[3]}, jchunks
MD = "# Title\nintro\n\n```\n# not a heading\n```\n\n## Setup\n" + "line\n" * 60 + "## Usage\ntext\n"
mchunks = chunking.chunk_text("README.md", MD)
msyms = {s for c in mchunks for s in c[3]}
assert "Setup" in msyms and "Usage" in msyms and "not a heading" not in msyms, msyms
assert len(chunking.chunk_text("x.txt", "\n".join(str(i) for i in range(100)))) >= 3
print("chunking ok")

# ---- url parsing (also rejects non-GitHub links) ----
assert repo_manager.parse_github_url("https://github.com/bhargav158/ai-agent-app") == ("bhargav158", "ai-agent-app")
assert repo_manager.parse_github_url("github.com/a/b.git") == ("a", "b")
assert repo_manager.parse_github_url("https://github.com/a/b/tree/main/src") == ("a", "b")
for bad in ["https://evil.com/a/b", "https://github.com/onlyowner", "", "rm -rf /"]:
    try:
        repo_manager.parse_github_url(bad)
        raise AssertionError("accepted bad url: " + bad)
    except ValueError:
        pass
print("url parsing ok")

# ---- indexing, resume, dedupe, cap ----
n = index_repo.index_repo(root, name="selftest", reset=True)
assert n > 5, "indexing produced too few chunks"
calls = {"n": 0}
_orig_embed = llm.embed
llm.embed = lambda texts, task: (calls.__setitem__("n", calls["n"] + 1), _orig_embed(texts, task))[1]
index_repo.index_repo(root, name="selftest")  # second run: nothing left to embed
assert calls["n"] == 0, "resume re-embedded stored chunks"
llm.embed = _orig_embed

d = tempfile.mkdtemp()
os.makedirs(os.path.join(d, "copy"))
same = "def hello():\n    return 'hi'\n" * 3
for sub in ("a.py", os.path.join("copy", "a.py")):
    open(os.path.join(d, sub), "w").write(same)
open(os.path.join(d, "b.py"), "w").write("def other():\n    return 2\n")
index_repo.index_repo(d, name="dup")
paths = {m["path"] for m in store.collection().get(where={"repo": "dup"}, include=["metadatas"])["metadatas"]}
assert len(paths) == 2, f"duplicate file was not skipped: {paths}"

index_repo.MAX_CHUNKS = 3
assert index_repo.index_repo(root, name="capped") <= 3
index_repo.MAX_CHUNKS = 400
print("index ok:", n, "chunks; resume, dedupe, cap ok")

# ---- retrieval and repo filter ----
assert retrieval.keyword_search("run_agent", 3)
assert retrieval.vector_search("retrieve grade rewrite loop", 3)
assert retrieval.hybrid_search("run_agent loop", 3)
for fn in (retrieval.vector_search, retrieval.keyword_search, retrieval.hybrid_search):
    res = fn("hello other def", 4, "dup")
    assert res and all(c["file"].startswith("dup/") for c in res), (fn.__name__, [c["file"] for c in res])
print("retrieval + repo filter ok")

# ---- agent ----
res = agent.run_agent("How does the agent loop work?")
steps = [e["step"] for e in res["trace"]]
assert "rewrite" in steps and "regenerate" in steps, steps
assert res["verified"] is True and res["sources"] and res["llm_calls"] >= 6
assert "link" in res["sources"][0]
print("agent ok:", steps)
rres = agent.run_agent("hello", None, "dup")
assert all(s["file"].startswith("dup/") for s in rres["sources"]), rres["sources"]
print("scoped agent ok")

# follow-up question is condensed using the conversation
hist = [{"q": "What does run_agent do?", "a": "It runs the loop."}]
fres = agent.run_agent("and where is it called?", None, None, hist)
fsteps = [e["step"] for e in fres["trace"]]
assert fsteps[0] == "condense" and fres["standalone"] == "What calls the retrieve function?", fsteps
assert "condense" not in [e["step"] for e in agent.run_agent("What does run_agent do?")["trace"]]
print("follow-up ok")

# a question that names a file reads the whole file instead of searching
rf = agent.run_agent("Explain agent.py", None, "selftest")
rsteps = [e["step"] for e in rf["trace"]]
assert "read_file" in rsteps and "route" not in rsteps and "retrieve" not in rsteps, rsteps
assert rf["sources"] and all(s["file"] == "selftest/agent.py" for s in rf["sources"]), rf["sources"]
assert retrieval.files_mentioned("what is in nothing_here.py?", "selftest") == []
print("read-file tool ok")

# the answer streams to the UI token by token
events = []
sres = agent.run_agent("How does the agent loop work?", events.append)
toks = [e["text"] for e in events if e["step"] == "token"]
assert len(toks) > 3 and "".join(toks).startswith(STRUCTURED), toks
assert sres["answer"] == STRUCTURED
assert "token" not in [e["step"] for e in sres["trace"]], "tokens must not bloat the trace"
print("streaming ok:", len(toks), "tokens")


# ---- benchmark ----
qs = gen_questions.main(n=3, out=os.path.join(os.environ["CHROMA_PATH"], "q.json"), repo="dup")
assert qs is not None
json.dump([{"question": "q1", "answer": "a1"}, {"question": "q2", "answer": "a2"}],
          open(os.path.join(os.environ["CHROMA_PATH"], "q.json"), "w"))
rows = benchmark.run(os.path.join(os.environ["CHROMA_PATH"], "q.json"), limit=2, sleep=0,
                     out_dir=os.path.join(os.environ["CHROMA_PATH"], "results"))
assert len(rows) == 2 and all("error" not in r for r in rows), rows
print("benchmark ok")

# ---- server: paste-a-link job, repos list, scoped ask, websocket ----
repo_manager.clone = lambda owner, repo: root  # no network: pretend we cloned this project
client = TestClient(server.app)
assert client.get("/health").json() == {"status": "ok"}
assert client.get("/").status_code == 200
assert client.post("/index", json={"url": "https://evil.com/x/y"}).status_code == 400

r = client.post("/index", json={"url": "https://github.com/test/selfrepo"})
assert r.status_code == 200 and r.json()["name"] == "selfrepo", r.text
for _ in range(100):
    st = client.get("/index/status/selfrepo").json()
    if st["status"] in ("done", "error"):
        break
    time.sleep(0.1)
assert st["status"] == "done", st
assert st["overview"]["what"] and st["chunks"] > 5
listed = {x["name"]: x for x in client.get("/repos").json()}
assert listed["selfrepo"]["github"] == "test/selfrepo" and listed["selfrepo"]["overview"], listed
print("paste-a-link job ok:", st["message"])

r = client.post("/ask", json={"question": "What does the router do?", "repo": "selfrepo"})
j = r.json()
assert r.status_code == 200 and j["parts"]["answer"] and j["parts"]["where"], j
assert all(s["link"].startswith("https://github.com/test/selfrepo/blob/HEAD/") for s in j["sources"]), j["sources"]

with client.websocket_connect("/ws") as sock:
    sock.send_json({"question": "and where is it used?", "repo": "selfrepo", "history": [{"q": "What is retrieval?", "a": "Finding chunks."}]})
    seen = []
    while True:
        ev = sock.receive_json()
        seen.append(ev["step"])
        if ev["step"] in ("final", "error"):
            break
    assert seen[-1] == "final" and "condense" in seen and "token" in seen and "retrieve" in seen, seen
print("server + websocket ok:", sorted(set(seen)))

# ---- password gate and rate limit ----
import base64
from starlette.websockets import WebSocketDisconnect
good = {"Authorization": "Basic " + base64.b64encode(b"me:secret").decode()}
bad = {"Authorization": "Basic " + base64.b64encode(b"me:wrong").decode()}
config.APP_PASSWORD = "secret"
assert client.get("/health").status_code == 200, "health check must stay open for the host"
assert client.get("/").status_code == 401
assert client.get("/", headers=bad).status_code == 401
assert client.get("/", headers=good).status_code == 200
try:
    with client.websocket_connect("/ws"):
        raise AssertionError("websocket opened without a password")
except WebSocketDisconnect:
    pass
with client.websocket_connect("/ws", headers=good) as sock:
    sock.send_json({"question": "How does retrieval work?", "repo": "selfrepo"})
    while sock.receive_json()["step"] not in ("final", "error"):
        pass
config.APP_PASSWORD = ""
print("password gate ok")

config.RATE_LIMIT_PER_MIN = 2
codes = [client.post("/ask", json={"question": "What does the router do?", "repo": "selfrepo"}).status_code for _ in range(3)]
assert codes == [200, 200, 429], codes
config.RATE_LIMIT_PER_MIN = 0
server._hits.clear()
print("rate limit ok")

print("\nALL OFFLINE TESTS PASSED")

# Agentic RAG Code Assistant

Ask questions about a codebase. The agent routes the search, grades what it retrieved, rewrites
the query when retrieval is poor, and verifies its own answer. A benchmark compares it with plain RAG.

```
question -> ROUTE (vector | keyword | hybrid | none)
         -> RETRIEVE -> GRADE --irrelevant--> REWRITE -> RETRIEVE (max 3 loops)
         -> GENERATE (with [file:lines] citations)
         -> VERIFY --unsupported claims--> REGENERATE once
```

## Setup (PowerShell)

```powershell
python -m venv venv
Set-ExecutionPolicy -Scope Process Bypass
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

Put your key from aistudio.google.com/apikey in `.env`.

## 1. Check the wiring (no API key needed)

```powershell
python tests/test_offline.py
```

## 2. Index your repos

```powershell
python index_repo.py C:\path\to\Fintrack --name fintrack --reset
python index_repo.py C:\path\to\Ai-agent-final --name voice-agent
```

## 3. Run the app

```powershell
uvicorn server:app --reload
```

Open http://127.0.0.1:8000 . Switch between Agentic and Naive and watch the live trace.

## 4. Benchmark (this produces the numbers for your resume)

```powershell
python gen_questions.py --n 30
```

Open `questions.json` and review it: delete bad questions and fix wrong reference answers.
Then:

```powershell
python benchmark.py --sleep 2
```

Results are saved to `results/results.md` and `results/results.json`.

## Files

| File | Job |
|---|---|
| `index_repo.py` | walk repo, chunk by 40 lines (10 overlap), embed, store in ChromaDB |
| `retrieval.py` | vector, BM25 keyword, and hybrid (Reciprocal Rank Fusion) search |
| `agent.py` | the agentic loop |
| `naive.py` | baseline: retrieve once, answer once |
| `server.py`, `static/index.html` | FastAPI + WebSocket live trace UI |
| `benchmark.py`, `gen_questions.py` | naive vs agentic comparison |
| `prompts.py`, `llm.py`, `config.py` | prompts, Gemini wrapper, settings |

## Design decisions (good interview answers)

- **Plain Python loop, not LangGraph:** the flow is small, so a loop is easier to debug and explain.
- **Grading is one LLM call for all chunks**, not one per chunk, to cut cost and latency.
- **Retries widen the search:** attempt 1 uses the router's choice, later attempts use hybrid.
- **Retries are capped at 3** so it cannot loop forever or burn the API quota.
- **If the grader or verifier returns bad JSON,** the code keeps the chunks or the answer instead of failing.
- **If nothing relevant is found,** it answers "I don't know" instead of guessing.
- **Benchmark judging is separate** from the agent's own calls, so the "LLM calls" column is fair.

## Known limitations (say these honestly)

- The grader and verifier are LLMs, so they can be wrong too.
- Auto-generated benchmark questions come from single chunks, so review them by hand.
- No "read whole file" tool yet. Good next step.

## Troubleshooting

- `429` or quota errors: raise `--sleep` in the benchmark, or wait a minute (free tier limits).
- A model name errors: set `GEN_MODEL` in `.env` to a current name from ai.google.dev.
- `chromadb` fails to install on Windows with a "Visual C++ 14.0 required" message: run
  `pip install --upgrade pip` and retry, or use Python 3.11.

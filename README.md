# Agentic RAG Code Assistant

**Ask questions about any public GitHub repository and get answers from the actual code, checked by the agent before you see them.**

```
GitHub link -> clone -> chunk -> embed -> ChromaDB  (+ plain-English overview)

question -> ROUTE (vector | keyword | hybrid | none)
         -> RETRIEVE -> GRADE --irrelevant--> REWRITE -> RETRIEVE (max 3 loops)
         -> GENERATE  (Answer / In simple terms / Where in the code)
         -> VERIFY --unsupported claims--> REGENERATE once
```

## What the output looks like

- **Answer:** one or two sentences that answer exactly what was asked.
- **In simple terms:** a beginner-friendly explanation; technical terms are explained in brackets.
- **Where in the code:** file and line ranges, each with a clickable GitHub link.
- **Agent steps:** a collapsible live trace of route, retrieve, grade, rewrite, verify.

## Setup (PowerShell)

```powershell
python -m venv venv
Set-ExecutionPolicy -Scope Process Bypass
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

Put your key from aistudio.google.com/apikey in `.env`. `git` must be installed (the app uses it to download repos).

## Run

```powershell
python tests/test_offline.py     # optional: checks the wiring, no API key needed
uvicorn server:app --reload
```

Open http://127.0.0.1:8000, paste a public GitHub link, and click **Index it**.
Then ask a question, for example: "Where is authentication handled?"

Command-line indexing also works:

```powershell
python index_repo.py C:\path\to\repo --name myrepo --github owner/repo
```

## Evaluate (optional)

```powershell
python gen_questions.py --n 15 --repo ai-agent-app
python benchmark.py --repo ai-agent-app --sleep 5
```

Open `questions.json` between the two commands and fix any bad questions. The benchmark reports
correctness, faithfulness, latency, LLM calls and search attempts per question in `results/results.md`.

## Scale and limits

- Indexing is **resumable** and saves after every batch. A rate-limit stop never loses progress.
- **Duplicate files** (for example a copied folder) are skipped.
- Each repo is capped at `MAX_CHUNKS` chunks (default 400, set in `.env`). README and top-level files are indexed first.
- Free-tier embedding limits make big repos slow: expect roughly 1 to 2 minutes per 100 chunks.
- Questions can be scoped to one repo, so results from other repos never mix in.
- Repos are downloaded to `~/rag_repos`, outside the project, so the dev server never restarts mid-index.
- Only public GitHub repositories work.

## Files

| File | Job |
|---|---|
| `repo_manager.py` | paste-a-link job: clone, index, overview, repo list |
| `index_repo.py` | walk repo, chunk by 40 lines (10 overlap), dedupe, embed, store |
| `retrieval.py` | vector, BM25 keyword, hybrid (Reciprocal Rank Fusion), repo filter |
| `agent.py` | the agentic loop |
| `formatting.py` | turns model output into Answer / Explanation / Where, builds GitHub links |
| `server.py`, `static/index.html` | FastAPI + WebSocket UI |
| `benchmark.py`, `gen_questions.py` | optional quality check |
| `prompts.py`, `llm.py`, `config.py`, `store.py` | prompts, Gemini wrapper, settings, ChromaDB |

## Design decisions (good interview answers)

- **Plain Python loop, not LangGraph:** the flow is small, so a loop is easier to debug and explain.
- **Grading is one LLM call for all chunks**, not one per chunk, to cut cost and latency.
- **Retries widen the search:** attempt 1 uses the router's choice, later attempts use hybrid.
- **Retries are capped at 3** so it cannot loop forever or burn the API quota.
- **If nothing relevant is found,** it answers "I don't know" instead of guessing.
- **The verifier ignores general definitions** and checks only claims about the codebase.
- **Only rate limits and server errors are retried;** a 400 error fails immediately.
- **The repo URL is validated** and `git` runs without a shell, so a pasted link cannot inject commands.

## Known limitations (say these honestly)

- The grader and verifier are LLMs, so they can be wrong too.
- Auto-generated benchmark questions come from single chunks, so review them by hand.
- No "read whole file" tool yet. Good next step.
- Indexing runs one repo at a time to stay inside free-tier limits.

## Troubleshooting

- `429` or quota errors: wait a minute, or raise `--sleep` in the benchmark.
- A model name errors: set `GEN_MODEL` in `.env` to a current name from ai.google.dev.
- "Could not download the repo": check that the repo is public and `git --version` works.
- `chromadb` fails to install on Windows with a "Visual C++ 14.0 required" message: run
  `pip install --upgrade pip` and retry, or use Python 3.11.

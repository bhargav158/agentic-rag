# Agentic RAG Code Assistant

![tests](https://github.com/bhargav158/agentic-rag/actions/workflows/tests.yml/badge.svg)

**Ask questions about any public GitHub repository and get answers from the actual code, checked by the agent before you see them.**

```
GitHub link -> clone -> chunk by function/class -> embed -> ChromaDB  (+ plain-English overview)

question -> [CONDENSE follow-up into a standalone question]
         -> names a file?  yes: READ THE WHOLE FILE
                           no:  ROUTE (vector | keyword | hybrid | none)
                                -> RETRIEVE -> GRADE --irrelevant--> REWRITE -> RETRIEVE (max 3 loops)
         -> GENERATE (streamed word by word)
         -> VERIFY --unsupported claims--> REGENERATE once
```

## Features

- **Paste a link, get an overview:** downloads the repo, indexes it, and explains it in plain English.
- **Answers that stay on the question:** a short answer, a beginner-friendly explanation, and file and line
  references with clickable GitHub links.
- **Follow-up questions:** "Where is that called from?" works, because the agent rewrites follow-ups into
  standalone questions using the last 3 turns. Click **New chat** to start fresh.
- **Function-level chunking:** Python (via `ast`), JS/TS and Markdown are split at functions, classes and
  headings, so each chunk is a whole unit of code. Other files use line windows.
- **Read-whole-file tool:** "Explain server.py" reads that file in full instead of searching.
- **Streaming:** the answer appears as it is written; a collapsible trace shows every agent step.
- **Safe to host:** optional password (`APP_PASSWORD`) and per-visitor rate limit (`RATE_LIMIT_PER_MIN`).

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

## Docker and deployment

```powershell
docker compose up --build        # http://localhost:8000
```

See `DEPLOY.md` for a free Render deployment (password-protected, rate-limited).

## Evaluate (optional)

```powershell
python gen_questions.py --n 15 --repo ai-agent-app
python benchmark.py --repo ai-agent-app --sleep 5
```

Open `questions.json` between the two commands and fix any bad questions. The benchmark reports
correctness, faithfulness, latency, LLM calls and search attempts in `results/results.md`.

## Scale and limits

- Indexing is **resumable** and saves after every batch. A rate-limit stop never loses progress.
- **Duplicate files** (for example a copied folder) are skipped.
- Each repo is capped at `MAX_CHUNKS` chunks (default 400). README and top-level files are indexed first.
- Free-tier embedding limits make big repos slow: expect roughly 1 to 2 minutes per 100 chunks.
- Questions can be scoped to one repo, so results from other repos never mix in.
- Repos are downloaded to `~/rag_repos`, outside the project, so the dev server never restarts mid-index.
- Only public GitHub repositories work.

## Files

| File | Job |
|---|---|
| `agent.py` | the agentic loop: condense, file tool, route, retrieve, grade, rewrite, generate, verify |
| `chunking.py` | function/class/heading-aware chunking |
| `index_repo.py` | walk repo, dedupe, chunk, embed, store |
| `retrieval.py` | vector, BM25 keyword, hybrid (Reciprocal Rank Fusion), repo filter, read-whole-file |
| `repo_manager.py` | paste-a-link job: clone, index, overview, repo list |
| `server.py`, `static/index.html` | FastAPI + WebSocket UI, password gate, rate limit |
| `llm.py`, `prompts.py`, `config.py`, `store.py`, `formatting.py` | Gemini wrapper, prompts, settings, ChromaDB, output parsing |
| `benchmark.py`, `gen_questions.py` | optional quality check |
| `Dockerfile`, `docker-compose.yml`, `render.yaml`, `.github/workflows/tests.yml` | Docker, deploy, CI |

## Design decisions (good interview answers)

- **Plain Python loop, not LangGraph:** the flow is small, so a loop is easier to debug and explain.
- **Function-level chunking:** a fixed 40-line window can cut a function in half; splitting at function
  boundaries keeps each retrieved chunk meaningful. Small neighbours are merged and big units are windowed.
- **Follow-ups use one extra LLM call** that rewrites the question to be standalone, so retrieval never sees
  vague words like "it" or "that".
- **The file tool rebuilds files from the index**, so it works even when the repo is not on disk (for example on a host).
- **Grading is one LLM call for all chunks**, not one per chunk, to cut cost and latency.
- **Retries widen the search:** attempt 1 uses the router's choice, later attempts use hybrid. Capped at 3.
- **If nothing relevant is found,** it answers "I don't know" instead of guessing.
- **The verifier ignores general definitions** and checks only claims about the codebase.
- **Streaming never restarts a half-shown answer:** retries happen only before the first word appears.
- **Only rate limits and server errors are retried;** a 400 error fails immediately.
- **The repo URL is validated** and `git` runs without a shell, so a pasted link cannot inject commands.
- **The password gate is ASGI middleware**, so it protects the WebSocket too, and `/health` stays open for the host.

## Known limitations (say these honestly)

- The grader and verifier are LLMs, so they can be wrong too.
- Naming a file reads only that file; it does not also search the rest of the repo.
- Conversation memory keeps the last 3 turns and lives in the browser tab.
- Auto-generated benchmark questions come from single chunks, so review them by hand.
- Indexing runs one repo at a time to stay inside free-tier limits.
- Docker and the Render blueprint are configured but were not run in the development sandbox.

## Troubleshooting

- `429` or quota errors: wait a minute, or raise `--sleep` in the benchmark.
- A model name errors: set `GEN_MODEL` in `.env` to a current name from ai.google.dev.
- "Could not download the repo": check that the repo is public and `git --version` works.
- After upgrading to function-level chunking, paste your repo links again so they are re-indexed.
- `chromadb` fails to install on Windows with a "Visual C++ 14.0 required" message: run
  `pip install --upgrade pip` and retry, or use Python 3.11.

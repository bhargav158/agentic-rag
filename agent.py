"""The agentic RAG loop:
[condense follow-up] -> read-file shortcut OR route -> retrieve -> grade -> (rewrite -> retrieve ...)
-> answer (streamed) -> verify."""
import time

import llm
import prompts
from config import MAX_RETRIES, TOP_K
from formatting import source_dict
from retrieval import files_mentioned, read_file_chunks, retrieve

STRATEGIES = {"vector", "keyword", "hybrid", "none"}
FILE_BUDGET = 24000  # characters of file text sent to the model when a file is read whole


def condense(question, history):
    """Turn a follow-up ('where is that called?') into a standalone question."""
    convo = "\n".join(f"User: {t['q']}\nAnswer: {t['a']}" for t in history[-3:])
    text = llm.generate(prompts.CONDENSE.format(history=convo, question=question))
    return (text or "").strip().strip('"') or question


def route(question):
    out = llm.generate_json(prompts.ROUTE.format(question=question), {}) or {}
    strategy = out.get("strategy", "hybrid")
    return (strategy if strategy in STRATEGIES else "hybrid"), out.get("reason", "")


def grade(question, chunks):
    """One LLM call grades all chunks at once. Returns only the relevant ones."""
    if not chunks:
        return []
    listing = "\n\n".join(f"### {i + 1}\n{c['text'][:900]}" for i, c in enumerate(chunks))
    out = llm.generate_json(prompts.GRADE.format(question=question, chunks=listing), {})
    if not isinstance(out, dict) or "relevant" not in out:
        return chunks  # grader failed to answer: keep everything rather than lose context
    nums = [n for n in out["relevant"] if isinstance(n, int) and 1 <= n <= len(chunks)]
    return [chunks[n - 1] for n in nums]


def rewrite(question, query, chunks):
    files = ", ".join(sorted({c["file"] for c in chunks})) or "none"
    text = llm.generate(prompts.REWRITE.format(question=question, query=query, files=files))
    return (text or query).strip().strip('"')


def verify(answer, chunks):
    """Returns (supported, unsupported_claims)."""
    context = "\n\n".join(c["text"] if isinstance(c, dict) else c for c in chunks)
    out = llm.generate_json(prompts.VERIFY.format(context=context, answer=answer), {}) or {}
    return bool(out.get("supported", True)), out.get("unsupported_claims", [])


def _answer(question, pool, feedback="", on_token=None):
    context = "\n\n".join(c["text"] for c in pool)
    prompt = prompts.ANSWER.format(feedback=feedback, context=context, question=question)
    return llm.generate_stream(prompt, on_token)


def run_agent(question, emit=None, repo=None, history=None):
    streaming = emit is not None
    emit = emit or (lambda e: None)
    on_token = (lambda t: emit({"step": "token", "text": t})) if streaming else None
    counter = llm.start_counting()
    t0 = time.time()
    trace = []
    q = question  # the standalone question (differs from `question` for follow-ups)

    def log(step, **data):
        ev = {"step": step, "t": round(time.time() - t0, 2), **data}
        trace.append(ev)
        emit(ev)

    def result(answer, pool, verified, attempts):
        return {
            "answer": answer, "verified": verified, "attempts": attempts, "standalone": q,
            "sources": [source_dict(c) for c in pool],
            "llm_calls": counter[0], "latency_s": round(time.time() - t0, 2),
            "trace": trace, "contexts": [c["text"] for c in pool],
        }

    # 0. FOLLOW-UP: make the question standalone using the conversation so far
    if history:
        q = condense(question, history)
        log("condense", standalone=q)

    pool, seen, attempts = [], set(), 0
    named = files_mentioned(q, repo)
    if named:
        # TOOL: the question names a file, so read it whole instead of searching
        for label in named:
            pool += read_file_chunks(label, FILE_BUDGET // len(named))
        log("read_file", files=named, chunks=len(pool))
    else:
        # 1. ROUTE
        strategy, reason = route(q)
        log("route", strategy=strategy, reason=reason)
        if strategy == "none":
            answer = llm.generate(prompts.DIRECT.format(question=q))
            log("generate", note="answered without retrieval")
            return result(answer, [], None, 0)

        # 2. RETRIEVE -> GRADE -> REWRITE loop
        query = q
        for attempt in range(1, MAX_RETRIES + 1):
            attempts = attempt
            strat = strategy if attempt == 1 else "hybrid"  # widen the search on retries
            chunks = retrieve(query, strat, TOP_K, repo)
            log("retrieve", attempt=attempt, strategy=strat, query=query,
                files=[c["file"] for c in chunks])
            relevant = grade(q, chunks)
            log("grade", attempt=attempt, relevant=len(relevant), total=len(chunks))
            for c in relevant:
                if c["id"] not in seen:
                    seen.add(c["id"])
                    pool.append(c)
            if pool:
                break
            if attempt < MAX_RETRIES:
                query = rewrite(q, query, chunks)
                log("rewrite", new_query=query)

        if not pool:
            log("give_up", note=f"no relevant code after {attempts} attempts")
            return result("I don't know based on the codebase.", [], None, attempts)

    # 3. GENERATE (streamed to the UI token by token)
    answer = _answer(q, pool, "", on_token)
    log("generate", chunks=len(pool))

    # 4. VERIFY (regenerate once if claims are unsupported)
    supported, unsupported = verify(answer, pool)
    log("verify", supported=supported, unsupported=unsupported)
    if not supported:
        feedback = ("A previous draft contained claims not supported by the context: "
                    + "; ".join(map(str, unsupported)) + ". Do not repeat them.\n")
        log("regenerate")
        answer = _answer(q, pool, feedback, on_token)
        supported, unsupported = verify(answer, pool)
        log("verify", supported=supported, unsupported=unsupported)

    return result(answer, pool, supported, attempts)

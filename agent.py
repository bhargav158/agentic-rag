"""The agentic RAG loop: route -> retrieve -> grade -> (rewrite -> retrieve ...) -> answer -> verify."""
import time

import llm
import prompts
from config import MAX_RETRIES, TOP_K
from retrieval import retrieve

STRATEGIES = {"vector", "keyword", "hybrid", "none"}


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
    context = "\n\n".join(c["text"] for c in chunks) if isinstance(chunks[0], dict) else "\n\n".join(chunks)
    out = llm.generate_json(prompts.VERIFY.format(context=context, answer=answer), {}) or {}
    return bool(out.get("supported", True)), out.get("unsupported_claims", [])


def _answer(question, pool, feedback=""):
    context = "\n\n".join(c["text"] for c in pool)
    return llm.generate(prompts.ANSWER.format(feedback=feedback, context=context, question=question))


def run_agent(question, emit=None):
    emit = emit or (lambda e: None)
    counter = llm.start_counting()
    t0 = time.time()
    trace = []

    def log(step, **data):
        ev = {"step": step, "t": round(time.time() - t0, 2), **data}
        trace.append(ev)
        emit(ev)

    def result(answer, pool, verified, attempts):
        return {
            "answer": answer, "verified": verified, "attempts": attempts,
            "sources": [{"file": c["file"], "start": c["start"], "end": c["end"]} for c in pool],
            "llm_calls": counter[0], "latency_s": round(time.time() - t0, 2),
            "trace": trace, "contexts": [c["text"] for c in pool],
        }

    # 1. ROUTE
    strategy, reason = route(question)
    log("route", strategy=strategy, reason=reason)
    if strategy == "none":
        answer = llm.generate(prompts.DIRECT.format(question=question))
        log("generate", note="answered without retrieval")
        return result(answer, [], None, 0)

    # 2. RETRIEVE -> GRADE -> REWRITE loop
    pool, seen, query, attempts = [], set(), question, 0
    for attempt in range(1, MAX_RETRIES + 1):
        attempts = attempt
        strat = strategy if attempt == 1 else "hybrid"  # widen the search on retries
        chunks = retrieve(query, strat, TOP_K)
        log("retrieve", attempt=attempt, strategy=strat, query=query,
            files=[c["file"] for c in chunks])
        relevant = grade(question, chunks)
        log("grade", attempt=attempt, relevant=len(relevant), total=len(chunks))
        for c in relevant:
            if c["id"] not in seen:
                seen.add(c["id"])
                pool.append(c)
        if pool:
            break
        if attempt < MAX_RETRIES:
            query = rewrite(question, query, chunks)
            log("rewrite", new_query=query)

    if not pool:
        log("give_up", note=f"no relevant code after {attempts} attempts")
        return result("I don't know based on the codebase.", [], None, attempts)

    # 3. GENERATE
    answer = _answer(question, pool)
    log("generate", chunks=len(pool))

    # 4. VERIFY (regenerate once if claims are unsupported)
    supported, unsupported = verify(answer, pool)
    log("verify", supported=supported, unsupported=unsupported)
    if not supported:
        feedback = ("A previous draft contained claims not supported by the context: "
                    + "; ".join(map(str, unsupported)) + ". Do not repeat them.\n")
        answer = _answer(question, pool, feedback)
        log("regenerate")
        supported, unsupported = verify(answer, pool)
        log("verify", supported=supported, unsupported=unsupported)

    return result(answer, pool, supported, attempts)

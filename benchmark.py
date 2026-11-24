"""Measure how well the agentic RAG answers questions about your indexed repo.

Usage:  python benchmark.py --repo ai-agent-app             (uses questions.json)
        python benchmark.py --limit 5 --sleep 2             (quick test)
"""
import argparse
import json
import os
import time

import llm
import prompts
from agent import run_agent, verify


def judge(question, reference, candidate):
    out = llm.generate_json(
        prompts.JUDGE.format(question=question, reference=reference, candidate=candidate), {}
    ) or {}
    return bool(out.get("correct", False))


def faithful(answer, contexts):
    if not contexts:
        return None  # nothing was retrieved (e.g. "I don't know"), so not measurable
    return verify(answer, contexts)[0]


def pct(values):
    vals = [v for v in values if v is not None]
    return round(100 * sum(vals) / len(vals), 1) if vals else None


def avg(values):
    return round(sum(values) / len(values), 2) if values else None


def run(questions_path="questions.json", limit=None, sleep=1.0, out_dir="results", repo=None):
    with open(questions_path, encoding="utf-8") as f:
        questions = json.load(f)
    if limit:
        questions = questions[:limit]

    rows = []
    for i, q in enumerate(questions, 1):
        try:
            res = run_agent(q["question"], None, repo)
            row = {
                "question": q["question"], "answer": res["answer"],
                "correct": judge(q["question"], q["answer"], res["answer"]),
                "faithful": faithful(res["answer"], res["contexts"]),
                "latency_s": res["latency_s"], "llm_calls": res["llm_calls"],
                "attempts": res["attempts"],
            }
        except Exception as e:
            row = {"question": q["question"], "error": str(e)}
        rows.append(row)
        print(f"[{i}/{len(questions)}] correct={row.get('correct')} faithful={row.get('faithful')} "
              f"latency={row.get('latency_s')}s calls={row.get('llm_calls')}")
        time.sleep(sleep)

    ok = [r for r in rows if "error" not in r]
    table = "\n".join([
        "| Questions | Correct % | Faithful % | Avg latency (s) | Avg LLM calls | Avg search attempts |",
        "|---|---|---|---|---|---|",
        f"| {len(ok)} | {pct([r['correct'] for r in ok])} | {pct([r['faithful'] for r in ok])} | "
        f"{avg([r['latency_s'] for r in ok])} | {avg([r['llm_calls'] for r in ok])} | "
        f"{avg([r['attempts'] for r in ok])} |",
    ])
    print("\n" + table)

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    with open(os.path.join(out_dir, "results.md"), "w", encoding="utf-8") as f:
        f.write(table + "\n")
    print(f"\nSaved to {out_dir}/results.json and {out_dir}/results.md")
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default="questions.json")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--sleep", type=float, default=1.0)
    ap.add_argument("--repo", help="only search this indexed repo")
    a = ap.parse_args()
    run(a.questions, a.limit, a.sleep, repo=a.repo)

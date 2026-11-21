"""Compare naive RAG vs agentic RAG on your own question set.

Usage:  python benchmark.py                      (uses questions.json)
        python benchmark.py --limit 5 --sleep 2  (quick test, slower to respect rate limits)
"""
import argparse
import json
import os
import time

import llm
import prompts
from agent import run_agent, verify
from naive import run_naive


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


def run(questions_path="questions.json", limit=None, sleep=1.0, out_dir="results"):
    with open(questions_path, encoding="utf-8") as f:
        questions = json.load(f)
    if limit:
        questions = questions[:limit]

    rows = []
    for i, q in enumerate(questions, 1):
        for mode, fn in (("naive", run_naive), ("agentic", run_agent)):
            try:
                res = fn(q["question"])
                row = {
                    "mode": mode, "question": q["question"], "answer": res["answer"],
                    "correct": judge(q["question"], q["answer"], res["answer"]),
                    "faithful": faithful(res["answer"], res["contexts"]),
                    "latency_s": res["latency_s"], "llm_calls": res["llm_calls"],
                }
            except Exception as e:
                row = {"mode": mode, "question": q["question"], "error": str(e)}
            rows.append(row)
            print(f"[{i}/{len(questions)}] {mode:8s} correct={row.get('correct')} "
                  f"latency={row.get('latency_s')}s calls={row.get('llm_calls')}")
            time.sleep(sleep)

    lines = ["| Mode | Questions | Correct % | Faithful % | Avg latency (s) | Avg LLM calls |",
             "|---|---|---|---|---|---|"]
    for mode in ("naive", "agentic"):
        ok = [r for r in rows if r["mode"] == mode and "error" not in r]
        lines.append(
            f"| {mode} | {len(ok)} | {pct([r['correct'] for r in ok])} | "
            f"{pct([r['faithful'] for r in ok])} | {avg([r['latency_s'] for r in ok])} | "
            f"{avg([r['llm_calls'] for r in ok])} |"
        )
    table = "\n".join(lines)
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
    a = ap.parse_args()
    run(a.questions, a.limit, a.sleep)

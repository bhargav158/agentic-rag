"""Auto-draft benchmark questions from random chunks of your indexed repo.

Usage:  python gen_questions.py --n 30
Then OPEN questions.json AND REVIEW IT: delete bad questions, fix wrong answers.
"""
import argparse
import json
import random

import llm
import prompts
import store


def main(n=30, out="questions.json", seed=42, repo=None):
    col = store.collection()
    data = col.get(where={"repo": repo}, include=["documents"]) if repo else col.get(include=["documents"])
    pool = [d for d in data["documents"] if len(d) > 400]
    random.Random(seed).shuffle(pool)
    questions = []
    for doc in pool:
        if len(questions) >= n:
            break
        qa = llm.generate_json(prompts.GENQ.format(chunk=doc[:2500]), {}) or {}
        if qa.get("question") and qa.get("answer"):
            questions.append({"question": qa["question"], "answer": qa["answer"]})
            print(f"{len(questions)}/{n}: {qa['question']}")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(questions, f, indent=2)
    print(f"Wrote {len(questions)} questions to {out}. Review them before benchmarking.")
    return questions


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--out", default="questions.json")
    ap.add_argument("--repo", help="only draw questions from this indexed repo")
    a = ap.parse_args()
    main(a.n, a.out, repo=a.repo)

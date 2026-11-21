"""Index a code repository: walk files -> chunk by lines -> embed -> store in ChromaDB.

Usage:  python index_repo.py C:\\path\\to\\repo
        python index_repo.py C:\\path\\to\\another-repo --name voice-agent
        python index_repo.py C:\\path\\to\\repo --reset      (wipe the index first)
"""
import argparse
import os

import llm
import store
from config import CHUNK_LINES, CHUNK_OVERLAP

EXTS = {".py", ".js", ".jsx", ".ts", ".tsx", ".md", ".html", ".sql",
        ".yml", ".yaml", ".toml", ".txt", ".java", ".c", ".cpp", ".go"}
SKIP_DIRS = {".git", "node_modules", "venv", ".venv", "__pycache__", "dist", "build",
             ".next", "chroma_db", ".idea", ".vscode", "coverage"}
SKIP_FILES = {"package-lock.json", "yarn.lock", "pnpm-lock.yaml"}
MAX_BYTES = 200_000


def chunk_file(text):
    """Yield (start_line, end_line, body). Chunks overlap by CHUNK_OVERLAP lines."""
    lines = text.splitlines()
    step = CHUNK_LINES - CHUNK_OVERLAP
    for start in range(0, max(len(lines), 1), step):
        end = min(start + CHUNK_LINES, len(lines))
        body = "\n".join(lines[start:end])
        if body.strip():
            yield start + 1, end, body
        if end >= len(lines):
            break


def index_repo(path, name=None, reset=False):
    path = os.path.abspath(path)
    name = name or os.path.basename(path)
    if reset:
        store.reset()

    ids, docs, metas = [], [], []
    files = 0
    for root, dirs, filenames in os.walk(path):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for fn in filenames:
            if fn in SKIP_FILES or os.path.splitext(fn)[1].lower() not in EXTS:
                continue
            full = os.path.join(root, fn)
            if os.path.getsize(full) > MAX_BYTES:
                continue
            rel = f"{name}/" + os.path.relpath(full, path).replace("\\", "/")
            with open(full, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
            files += 1
            for s, e, body in chunk_file(text):
                ids.append(f"{rel}:{s}-{e}")
                docs.append(f"[{rel}:{s}-{e}]\n{body}")
                metas.append({"file": rel, "start": s, "end": e, "repo": name})

    if not ids:
        print("No indexable files found.")
        return 0

    # Resumable: skip chunks already stored, and save after every batch so a
    # rate-limit error never loses progress. Just run the same command again.
    done = set(store.collection().get(include=[])["ids"])
    todo = [i for i, cid in enumerate(ids) if cid not in done]
    print(f"Found {len(ids)} chunks in {files} files; {len(ids) - len(todo)} already indexed, "
          f"embedding {len(todo)}...")
    BATCH = 20
    for b in range(0, len(todo), BATCH):
        sel = todo[b:b + BATCH]
        b_docs = [docs[i] for i in sel]
        vecs = llm.embed(b_docs, "RETRIEVAL_DOCUMENT")
        store.add([ids[i] for i in sel], b_docs, vecs, [metas[i] for i in sel])
        print(f"  stored {min(b + BATCH, len(todo))}/{len(todo)}")
    print(f"Done. '{name}' has {len(ids)} chunks indexed.")
    return len(ids)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--name")
    ap.add_argument("--reset", action="store_true")
    a = ap.parse_args()
    index_repo(a.path, a.name, a.reset)
"""Index a code repository: walk files -> chunk by function/class -> embed -> store in ChromaDB.

Usage:  python index_repo.py C:\\path\\to\\repo
        python index_repo.py C:\\path\\to\\repo --name myrepo --github owner/repo
        python index_repo.py C:\\path\\to\\repo --reset      (wipe the whole index first)
"""
import argparse
import hashlib
import os

import llm
import store
from chunking import chunk_text
from config import MAX_CHUNKS

EXTS = {".py", ".js", ".jsx", ".ts", ".tsx", ".md", ".html", ".sql",
        ".yml", ".yaml", ".toml", ".txt", ".java", ".c", ".cpp", ".go"}
SKIP_DIRS = {".git", "node_modules", "venv", ".venv", "__pycache__", "dist", "build",
             ".next", "chroma_db", ".idea", ".vscode", "coverage"}
SKIP_FILES = {"package-lock.json", "yarn.lock", "pnpm-lock.yaml"}
MAX_BYTES = 200_000


def collect_files(path):
    found = []
    for root, dirs, filenames in os.walk(path):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for fn in filenames:
            if fn in SKIP_FILES or os.path.splitext(fn)[1].lower() not in EXTS:
                continue
            full = os.path.join(root, fn)
            if os.path.getsize(full) > MAX_BYTES:
                continue
            found.append((os.path.relpath(full, path).replace("\\", "/"), full))
    # README and top-level files first, so the chunk cap keeps the most important code
    found.sort(key=lambda f: (0 if os.path.basename(f[0]).lower().startswith("readme") else 1,
                              f[0].count("/"), f[0]))
    return found


def index_repo(path, name=None, reset=False, github="", progress=None):
    """Returns the number of chunks the repo has in the index.
    progress(done, total) is called after every stored batch."""
    path = os.path.abspath(path)
    name = name or os.path.basename(path)
    if reset:
        store.reset()

    ids, docs, metas = [], [], []
    seen, files, dupes, capped = set(), 0, 0, False
    for rel, full in collect_files(path):
        with open(full, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        if not text.strip():
            continue
        digest = hashlib.sha1(text.encode("utf-8", "ignore")).hexdigest()
        if digest in seen:  # identical file already indexed (e.g. a copied folder)
            dupes += 1
            continue
        seen.add(digest)
        files += 1
        for s, e, body, syms in chunk_text(rel, text):
            if len(ids) >= MAX_CHUNKS:
                capped = True
                break
            label = f"{name}/{rel}"
            head = f"[{label}:{s}-{e}]"
            if syms:
                head += "\n# defines: " + ", ".join(syms[:8])
            ids.append(f"{label}:{s}-{e}")
            docs.append(f"{head}\n{body}")
            metas.append({"file": label, "path": rel, "start": s, "end": e,
                          "repo": name, "github": github or "", "symbols": ", ".join(syms[:8])})
        if capped:
            break

    if not ids:
        print("No indexable files found.")
        return 0
    if dupes:
        print(f"Skipped {dupes} duplicate files.")
    if capped:
        print(f"Reached the {MAX_CHUNKS}-chunk limit; README and top-level files were indexed first.")

    # Resumable: skip chunks already stored, and save after every batch so a
    # rate-limit error never loses progress. Just run the same command again.
    done = set(store.collection().get(include=[])["ids"])
    todo = [i for i, cid in enumerate(ids) if cid not in done]
    print(f"Found {len(ids)} chunks in {files} files; {len(ids) - len(todo)} already indexed, "
          f"embedding {len(todo)}...")
    if progress:
        progress(0, len(todo))
    BATCH = 20
    for b in range(0, len(todo), BATCH):
        sel = todo[b:b + BATCH]
        b_docs = [docs[i] for i in sel]
        vecs = llm.embed(b_docs, "RETRIEVAL_DOCUMENT")
        store.add([ids[i] for i in sel], b_docs, vecs, [metas[i] for i in sel])
        n_done = min(b + BATCH, len(todo))
        print(f"  stored {n_done}/{len(todo)}")
        if progress:
            progress(n_done, len(todo))
    print(f"Done. '{name}' has {len(ids)} chunks indexed.")
    return len(ids)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--name")
    ap.add_argument("--github", default="", help="owner/repo, enables clickable source links")
    ap.add_argument("--reset", action="store_true")
    a = ap.parse_args()
    index_repo(a.path, a.name, a.reset, a.github)

"""Paste a GitHub link -> clone -> index -> plain-English overview, as a background job."""
import json
import os
import re
import subprocess
import threading

import config
import index_repo
import llm
import prompts
import store
from formatting import parse_overview

JOBS = {}
_lock = threading.Lock()  # one job at a time keeps us inside the free-tier rate limits

_URL = re.compile(r"^(?:https?://)?(?:www\.)?github\.com/([\w.-]+)/([\w.-]+?)(?:\.git)?(?:/.*)?$")


def parse_github_url(url):
    m = _URL.match((url or "").strip())
    if not m:
        raise ValueError("Paste a GitHub link like https://github.com/owner/repo")
    return m.group(1), m.group(2)


def clone(owner, repo):
    os.makedirs(config.REPOS_DIR, exist_ok=True)
    dest = os.path.join(config.REPOS_DIR, repo)
    has_copy = os.path.isdir(os.path.join(dest, ".git"))
    cmd = (["git", "-C", dest, "pull", "--ff-only"] if has_copy else
           ["git", "clone", "--depth", "1", f"https://github.com/{owner}/{repo}.git", dest])
    env = dict(os.environ, GIT_TERMINAL_PROMPT="0")  # private repos fail instead of prompting
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=300, env=env)
    except FileNotFoundError:
        raise RuntimeError("git is not installed or not on PATH.")
    except subprocess.TimeoutExpired:
        raise RuntimeError("Downloading the repository took too long.")
    if r.returncode != 0 and not has_copy:
        raise RuntimeError("Could not download the repo. Is it public? " + r.stderr.strip()[-200:])
    return dest


def _overview_path():
    return os.path.join(config.CHROMA_PATH, "overviews.json")


def _load_overviews():
    try:
        with open(_overview_path(), encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_overview(name, overview):
    data = _load_overviews()
    data[name] = overview
    os.makedirs(config.CHROMA_PATH, exist_ok=True)
    with open(_overview_path(), "w", encoding="utf-8") as f:
        json.dump(data, f)


def make_overview(name, dest):
    readme = ""
    for fn in sorted(os.listdir(dest)):
        if fn.lower().startswith("readme"):
            with open(os.path.join(dest, fn), encoding="utf-8", errors="ignore") as f:
                readme = f.read()[:6000]
            break
    metas = store.collection().get(where={"repo": name}, include=["metadatas"])["metadatas"]
    files = sorted({m["path"] for m in metas if "path" in m})[:60]
    text = llm.generate(prompts.OVERVIEW.format(readme=readme or "(no README)", files="\n".join(files)))
    return parse_overview(text) or {"what": (text or "").strip(), "stack": "", "files": [], "terms": []}


def start_job(url):
    owner, repo = parse_github_url(url)
    if not _lock.acquire(blocking=False):
        raise RuntimeError("Another repository is still being indexed. Wait for it to finish.")
    name = repo
    JOBS[name] = {"status": "cloning", "message": "Downloading the repository...",
                  "done": 0, "total": 0, "overview": None}

    def run():
        st = JOBS[name]
        try:
            dest = clone(owner, repo)
            st.update(status="indexing", message="Reading and embedding the code...")
            store.delete_repo(name)  # re-pasting a link replaces the old copy

            def progress(done, total):
                st.update(done=done, total=total, message=f"Embedded {done} of {total} chunks")

            n = index_repo.index_repo(dest, name=name, github=f"{owner}/{repo}", progress=progress)
            if n == 0:
                raise RuntimeError("No code files found in this repository.")
            st.update(status="summarizing", message="Writing a plain-English overview...")
            overview = make_overview(name, dest)
            _save_overview(name, overview)
            st.update(status="done", message=f"Indexed {n} chunks.", chunks=n, overview=overview)
        except Exception as e:
            st.update(status="error", message=str(e))
        finally:
            _lock.release()

    try:
        threading.Thread(target=run, daemon=True).start()
    except Exception:
        _lock.release()
        raise
    return name


def list_repos():
    col = store.collection()
    if col.count() == 0:
        return []
    counts, github = {}, {}
    for m in col.get(include=["metadatas"])["metadatas"]:
        r = m.get("repo", "?")
        counts[r] = counts.get(r, 0) + 1
        github.setdefault(r, m.get("github", ""))
    overviews = _load_overviews()
    return [{"name": r, "chunks": n, "github": github[r], "overview": overviews.get(r)}
            for r, n in sorted(counts.items())]

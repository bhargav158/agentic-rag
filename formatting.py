"""Turns the model's labelled text into clean parts the UI can show clearly."""
import re

ANSWER_LABELS = ["ANSWER", "EXPLANATION", "WHERE"]
OVERVIEW_LABELS = ["WHAT IT DOES", "TECH STACK", "KEY FILES", "KEY TERMS"]


def parse_sections(text, labels):
    """Split 'LABEL: body' blocks. Tolerates **bold** labels. Returns {LABEL: body}."""
    pattern = r"^[ \t*#-]*(" + "|".join(re.escape(l) for l in labels) + r")[ \t*]*:[ \t*]*"
    pieces = re.split(pattern, text or "", flags=re.M | re.I)
    return {pieces[i].upper(): pieces[i + 1].strip() for i in range(1, len(pieces) - 1, 2)}


def _lines(block):
    return [re.sub(r"^[-*\u2022]\s*", "", l).strip() for l in (block or "").splitlines() if l.strip()]


def parse_answer(text):
    sec = parse_sections(text, ANSWER_LABELS)
    if "ANSWER" not in sec:
        return None
    return {"answer": sec["ANSWER"], "explanation": sec.get("EXPLANATION", ""),
            "where": _lines(sec.get("WHERE", ""))}


def parse_overview(text):
    sec = parse_sections(text, OVERVIEW_LABELS)
    if "WHAT IT DOES" not in sec:
        return None
    return {"what": sec["WHAT IT DOES"], "stack": sec.get("TECH STACK", ""),
            "files": _lines(sec.get("KEY FILES", "")), "terms": _lines(sec.get("KEY TERMS", ""))}


def source_dict(c):
    """A source for the UI, with a clickable GitHub link when we know the repo."""
    link = ""
    if c.get("github"):
        link = f"https://github.com/{c['github']}/blob/HEAD/{c['path']}#L{c['start']}-L{c['end']}"
    return {"file": c["file"], "start": c["start"], "end": c["end"], "link": link}

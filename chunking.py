"""Code-aware chunking.

Python, JS/TS and Markdown are split at natural boundaries (functions, classes, headings)
so every chunk is a whole unit of code. Other files use plain line windows.
Every line of a file ends up in at least one chunk."""
import ast
import os
import re

from config import CHUNK_LINES, CHUNK_OVERLAP

TARGET = 50    # merge small neighbouring units up to this many lines
MAX_UNIT = 90  # units longer than this are split into overlapping windows

JS_EXTS = {".js", ".jsx", ".ts", ".tsx"}
_JS_BOUNDARY = re.compile(
    r"^(?:export\s+)?(?:default\s+)?(?:async\s+)?"
    r"(?:function\*?\s+(\w+)|class\s+(\w+)|"
    r"(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s*)?(?:function\b|\([^)]*\)\s*=>|\w+\s*=>))"
)
_MD_HEADING = re.compile(r"^#{1,3}\s+(.+?)\s*$")


def _windows(s, e, syms):
    """Overlapping line windows over lines s..e (1-based, inclusive)."""
    step = CHUNK_LINES - CHUNK_OVERLAP
    out, ws = [], s
    while True:
        we = min(ws + CHUNK_LINES - 1, e)
        out.append((ws, we, list(syms)))
        if we >= e:
            return out
        ws += step


def _units_from_bounds(n, bounds):
    """bounds: [(start_line, symbol)]. Returns contiguous (start, end, [symbol]) units."""
    first = {}
    for s, sym in sorted(bounds):
        first.setdefault(s, sym)
    starts = sorted(first)
    if not starts:
        return None
    units = []
    if starts[0] > 1:
        units.append((1, starts[0] - 1, []))  # imports, constants, intro text
    for i, s in enumerate(starts):
        e = starts[i + 1] - 1 if i + 1 < len(starts) else n
        units.append((s, e, [first[s]]))
    return units


def _py_start(node, lines):
    s = min([node.lineno] + [d.lineno for d in node.decorator_list])
    while s - 2 >= 0 and lines[s - 2].lstrip().startswith("#"):  # keep the comment above it
        s -= 1
    return s


def _py_units(text, lines):
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return None
    defs = (ast.FunctionDef, ast.AsyncFunctionDef)
    bounds = []
    for node in tree.body:
        if isinstance(node, defs + (ast.ClassDef,)):
            s = _py_start(node, lines)
            bounds.append((s, node.name))
            if isinstance(node, ast.ClassDef) and node.end_lineno - s + 1 > MAX_UNIT:
                for sub in node.body:  # big class: one unit per method
                    if isinstance(sub, defs):
                        ss = _py_start(sub, lines)
                        if ss > s:
                            bounds.append((ss, f"{node.name}.{sub.name}"))
    return _units_from_bounds(len(lines), bounds)


def _js_units(lines):
    bounds = []
    for i, line in enumerate(lines, 1):
        m = _JS_BOUNDARY.match(line)
        if m:
            bounds.append((i, next(g for g in m.groups() if g)))
    return _units_from_bounds(len(lines), bounds)


def _md_units(lines):
    bounds, fenced = [], False
    for i, line in enumerate(lines, 1):
        if line.lstrip().startswith("```"):
            fenced = not fenced
        elif not fenced:
            m = _MD_HEADING.match(line)
            if m:
                bounds.append((i, m.group(1)))
    return _units_from_bounds(len(lines), bounds)


def _merge(units):
    out = []
    for s, e, syms in units:
        if out and e - out[-1][0] + 1 <= TARGET:
            ps, _, psyms = out[-1]
            out[-1] = (ps, e, psyms + syms)
        else:
            out.append((s, e, list(syms)))
    return out


def chunk_text(path, text):
    """Returns [(start_line, end_line, body, symbols)]."""
    lines = text.splitlines()
    n = len(lines)
    if n == 0:
        return []
    ext = os.path.splitext(path)[1].lower()
    units = None
    if ext == ".py":
        units = _py_units(text, lines)
    elif ext in JS_EXTS:
        units = _js_units(lines)
    elif ext == ".md":
        units = _md_units(lines)

    if units is None:
        final = _windows(1, n, [])
    else:
        final = []
        for s, e, syms in _merge(units):
            final += _windows(s, e, syms) if e - s + 1 > MAX_UNIT else [(s, e, syms)]

    result = []
    for s, e, syms in final:
        body = "\n".join(lines[s - 1:e])
        if body.strip():
            result.append((s, e, body, syms))
    return result

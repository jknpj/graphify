"""Dm extractor. Moved verbatim from graphify/extract.py."""
from __future__ import annotations

import re

from pathlib import Path
from typing import Any
from graphify.extractors.base import (
    _file_stem,
    _make_id,
    _read_source_bytes,
    _read_source_text,
    _read_text,
)


_DM_INCLUDE_RE = re.compile(r'^\s*#include\s+["\']([^"\']+)["\']', re.IGNORECASE)
_DM_PATH_RE = re.compile(r'^/([A-Za-z0-9_]+(?:/[A-Za-z0-9_]+)*)')
_DM_PROC_MARK_RE = re.compile(r'^proc/([A-Za-z0-9_]+)\s*\(')
_DM_VAR_MARK_RE = re.compile(r'^var/')
_DM_MEMBER_PROC_RE = re.compile(r'^([A-Za-z0-9_]+)\s*\(')
_DM_CALL_RE = re.compile(r'\b([A-Za-z_][A-Za-z0-9_]*)\s*\(')
_DM_NEW_RE = re.compile(r'\bnew\s+(/[A-Za-z0-9_/]+)')
_DM_KEYWORDS = {
    "if", "else", "for", "while", "do", "switch", "case", "break", "continue",
    "return", "new", "delete", "spawn", "set", "src", "usr", "world", "global",
    "istype", "isnull", "locate", "min", "max", "abs", "rand", "pick", "prob",
    "sleep", "call", "view", "oview", "range", "get_step", "get_dir", "get_dist",
    "text", "sound", "icon", "image", "file", "matrix", "vector", "rgb", "hsv",
    "animate", "filter", "walk", "walk_to", "step", "step_to", "step_towards",
    "hearers", "viewers", "oviewers", "orange", "block", "flick", "alert",
    "input", "list", "assert", "to_chat", "vars", "type", "nameof", "defined",
}


def _dm_var_name(rest: str) -> str:
    """Name of the variable in a ``var/...`` declaration tail.

    Handles ``var/list/foo`` (type path before the name), ``var/foo = 1`` and
    ``var/foo[3]``: the name is the last path segment before ``=``/``[``/space.
    """
    head = re.split(r'[=(\s\[]', rest, maxsplit=1)[0]
    segs = [s for s in head.split("/") if s]
    return segs[-1] if segs else ""


def extract_dm(path: Path) -> dict:
    """Extract types, procs, vars, includes, inheritance and calls from a .dm/.dme file.

    Regex-based on purpose: tree-sitter-dm exposes ``type_definition`` without a
    ``type_path`` child (the leading ``/`` is a sibling token), which made the
    previous tree-sitter version emit nothing but the file node for whole files.
    This version is also much faster on large BYOND codebases.
    """
    try:
        text = _read_source_text(path)
    except Exception as e:
        return {"nodes": [], "edges": [], "error": str(e)}

    str_path = str(path)
    stem = _file_stem(path)
    file_nid = _make_id(str(path))
    nodes: list[dict] = [{"id": file_nid, "label": path.name, "file_type": "code",
                          "source_file": str_path, "source_location": "L1"}]
    edges: list[dict] = []
    raw_calls: list[dict] = []
    seen_ids: set[str] = {file_nid}
    proc_name_index: dict[str, list[str]] = {}
    proc_nodes: list[tuple[str, int, int]] = []

    def add_node(nid: str, label: str, line: int) -> None:
        if nid and nid not in seen_ids:
            seen_ids.add(nid)
            nodes.append({"id": nid, "label": label, "file_type": "code",
                          "source_file": str_path, "source_location": f"L{line}"})

    def add_edge(src: str, tgt: str, relation: str, line: int,
                 context: "str | None" = None) -> None:
        if not src or not tgt or src == tgt:
            return
        edge: dict = {"source": src, "target": tgt, "relation": relation,
                      "confidence": "EXTRACTED", "source_file": str_path,
                      "source_location": f"L{line}", "weight": 1.0}
        if context:
            edge["context"] = context
        edges.append(edge)

    def ensure_type(dm_path: str, line: int) -> str:
        nid = _make_id(dm_path)
        add_node(nid, dm_path, line)
        return nid

    def add_proc(owner_path: "str | None", proc_name: str, line: int,
                 decl_indent: int) -> "str | None":
        if not proc_name:
            return None
        if owner_path:
            owner_nid = ensure_type(owner_path, line)
            nid = _make_id(owner_path, proc_name)
            add_node(nid, f"{owner_path}/{proc_name}()", line)
            add_edge(owner_nid, nid, "method", line)
        else:
            nid = _make_id(stem, proc_name)
            add_node(nid, f"{proc_name}()", line)
            add_edge(file_nid, nid, "contains", line)
        proc_name_index.setdefault(proc_name.lower(), []).append(nid)
        proc_nodes.append((nid, line, decl_indent))
        return nid

    def add_var(owner_path: str, var_name: str, line: int) -> None:
        if not var_name:
            return
        owner_nid = ensure_type(owner_path, line)
        nid = _make_id(owner_path, "var", var_name)
        add_node(nid, f"{owner_path}/var/{var_name}", line)
        add_edge(owner_nid, nid, "contains", line)

    current_type: "str | None" = None
    current_type_indent = 0
    member_indent: "int | None" = None

    for idx, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("//") or stripped.startswith("/*"):
            continue
        indent = len(line) - len(line.lstrip())

        if stripped.startswith("#"):
            m = _DM_INCLUDE_RE.match(stripped)
            if m:
                raw = m.group(1).replace("\\", "/")
                norm = re.sub(r"^\./", "", raw)
                resolved = (path.parent / norm).resolve()
                tgt = _make_id(str(resolved)) if resolved.exists() else _make_id(norm)
                edge = {"source": file_nid, "target": tgt,
                        "relation": "imports_from" if resolved.exists() else "imports",
                        "context": "import", "confidence": "EXTRACTED",
                        "source_file": str_path, "source_location": f"L{idx}",
                        "weight": 1.0}
                if not resolved.exists():
                    edge["external"] = True
                edges.append(edge)
            continue

        if stripped.startswith("/"):
            m = _DM_PATH_RE.match(stripped)
            if not m:
                continue
            dm_path = "/" + m.group(1)
            rest = stripped[m.end():].strip()
            current_type = None
            member_indent = None

            if "/proc/" in dm_path:
                owner, name = dm_path.split("/proc/", 1)
                add_proc(owner, name.split("(")[0].strip(), idx, indent)
                current_type, current_type_indent = owner, indent
                member_indent = indent
            elif "/var/" in dm_path:
                owner, rest_var = dm_path.split("/var/", 1)
                add_var(owner, _dm_var_name(rest_var + " " + rest), idx)
                current_type, current_type_indent = owner, indent
                member_indent = indent
            elif rest.startswith("("):
                owner, _, name = dm_path.rpartition("/")
                add_proc(owner or None, name, idx, indent)
                current_type, current_type_indent = owner or None, indent
                member_indent = indent
            elif "=" in rest:
                continue
            else:
                type_nid = ensure_type(dm_path, idx)
                add_edge(file_nid, type_nid, "contains", idx)
                current_type, current_type_indent = dm_path, indent
                parent = dm_path.rpartition("/")[0]
                if parent and parent != "/":
                    add_edge(type_nid, _make_id(parent), "inherits", idx)
            continue

        if current_type is None or indent <= current_type_indent:
            continue
        if member_indent is None:
            member_indent = indent
        if indent != member_indent:
            continue

        m = _DM_PROC_MARK_RE.match(stripped)
        if m:
            add_proc(current_type, m.group(1), idx, indent)
            continue
        if _DM_VAR_MARK_RE.match(stripped):
            add_var(current_type, _dm_var_name(stripped[len("var/"):]), idx)
            continue
        m = _DM_MEMBER_PROC_RE.match(stripped)
        if m and m.group(1).lower() not in _DM_KEYWORDS:
            add_proc(current_type, m.group(1), idx, indent)
            continue

    lines = text.splitlines()
    total = len(lines)
    for nid, decl_line, decl_indent in proc_nodes:
        start = decl_line
        end = start
        while end < total:
            ln = lines[end]
            if ln.strip() and (len(ln) - len(ln.lstrip())) <= decl_indent:
                break
            end += 1
        body = "\n".join(lines[start:end])
        if not body.strip():
            continue
        base_line = start + 1
        for cm in _DM_CALL_RE.finditer(body):
            callee = cm.group(1)
            if callee.lower() in _DM_KEYWORDS:
                continue
            line_no = base_line + body.count("\n", 0, cm.start())
            candidates = proc_name_index.get(callee.lower(), [])
            if len(candidates) == 1:
                add_edge(nid, candidates[0], "calls", line_no, context="call")
            else:
                raw_calls.append({"caller_nid": nid, "callee": callee,
                                  "is_member_call": False, "source_file": str_path,
                                  "source_location": f"L{line_no}"})
        for nm in _DM_NEW_RE.finditer(body):
            target = "/" + nm.group(1).strip("/")
            line_no = base_line + body.count("\n", 0, nm.start())
            add_edge(nid, _make_id(target), "instantiates", line_no, context="call")

    return {"nodes": nodes, "edges": edges, "raw_calls": raw_calls}


def _read_dmi_description(data: bytes) -> str:
    """Pull the BYOND metadata text out of a .dmi PNG, or empty string on failure."""
    import struct
    import zlib as _zlib
    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        return ""
    i = 8
    while i + 8 <= len(data):
        length = struct.unpack(">I", data[i:i + 4])[0]
        chunk_type = data[i + 4:i + 8]
        payload = data[i + 8:i + 8 + length]
        if chunk_type in (b"tEXt", b"zTXt"):
            try:
                null = payload.index(b"\x00")
            except ValueError:
                return ""
            keyword = payload[:null]
            if keyword == b"Description":
                if chunk_type == b"zTXt":
                    return _zlib.decompressobj().decompress(payload[null + 2:], max_length=1024 * 1024).decode("utf-8", errors="replace")
                return payload[null + 1:].decode("utf-8", errors="replace")
        i += 8 + length + 4
    return ""

def extract_dmi(path: Path) -> dict:
    """Extract icon state names from a .dmi (BYOND PNG icon sheet)."""
    try:
        data = path.read_bytes()
    except Exception as e:
        return {"nodes": [], "edges": [], "error": str(e)}

    str_path = str(path)
    stem = _file_stem(path)
    file_nid = _make_id(str(path))
    nodes: list[dict] = [{"id": file_nid, "label": path.name, "file_type": "code",
                           "source_file": str_path, "source_location": "L1"}]
    edges: list[dict] = []
    seen: set[str] = {file_nid}

    description = _read_dmi_description(data)
    if not description:
        return {"nodes": nodes, "edges": edges}

    line_no = 0
    for raw_line in description.splitlines():
        line_no += 1
        stripped = raw_line.strip()
        if not stripped.startswith("state ="):
            continue
        value = stripped.split("=", 1)[1].strip()
        if value.startswith('"') and value.endswith('"') and len(value) >= 2:
            state_name = value[1:-1]
        else:
            state_name = value
        if not state_name:
            continue
        nid = _make_id(stem, "state", state_name)
        if nid in seen:
            continue
        seen.add(nid)
        nodes.append({"id": nid, "label": f'"{state_name}"', "file_type": "code",
                      "source_file": str_path, "source_location": f"L{line_no}"})
        edges.append({"source": file_nid, "target": nid, "relation": "contains",
                      "confidence": "EXTRACTED", "source_file": str_path,
                      "source_location": f"L{line_no}", "weight": 1.0})

    return {"nodes": nodes, "edges": edges}

_DMM_GRID_RE = re.compile(r"^\(\s*\d+\s*,\s*\d+\s*,\s*\d+\s*\)\s*=", re.MULTILINE)

def _split_dmm_tile(body: str) -> list[str]:
    out: list[str] = []
    buf: list[str] = []
    depth = 0
    in_string = False
    escape = False
    for ch in body:
        if escape:
            buf.append(ch)
            escape = False
            continue
        if in_string:
            buf.append(ch)
            if ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            buf.append(ch)
        elif ch in "({[":
            depth += 1
            buf.append(ch)
        elif ch in ")}]":
            depth -= 1
            buf.append(ch)
        elif ch == "," and depth == 0:
            out.append("".join(buf).strip())
            buf = []
        else:
            buf.append(ch)
    tail = "".join(buf).strip()
    if tail:
        out.append(tail)
    return out

def _dmm_type_path(entry: str) -> str:
    brace = entry.find("{")
    if brace != -1:
        entry = entry[:brace]
    return entry.strip()

def extract_dmm(path: Path) -> dict:
    """Extract type-path references from a .dmm map file's tile dictionary."""
    try:
        if path.stat().st_size > 50 * 1024 * 1024:
            return {"nodes": [], "edges": [], "error": "file too large (>50 MB)"}
        text = _read_source_text(path)
    except Exception as e:
        return {"nodes": [], "edges": [], "error": str(e)}

    str_path = str(path)
    file_nid = _make_id(str(path))
    nodes: list[dict] = [{"id": file_nid, "label": path.name, "file_type": "code",
                           "source_file": str_path, "source_location": "L1"}]
    edges: list[dict] = []

    grid_match = _DMM_GRID_RE.search(text)
    dict_text = text[:grid_match.start()] if grid_match else text

    seen_targets: set[str] = set()
    buf: list[str] = []
    open_line = 0
    depth = 0
    in_string = False
    escape = False
    for line_idx, line in enumerate(dict_text.splitlines(), start=1):
        for ch in line:
            if escape:
                escape = False
            elif in_string:
                if ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
            elif ch == '"':
                in_string = True
            elif ch == "(":
                if depth == 0:
                    open_line = line_idx
                depth += 1
            elif ch == ")":
                depth -= 1
            buf.append(ch)
        buf.append("\n")
        if depth == 0 and buf:
            chunk = "".join(buf)
            buf = []
            lp = chunk.find("(")
            rp = chunk.rfind(")")
            if lp == -1 or rp == -1 or rp <= lp:
                continue
            inner = chunk[lp + 1:rp]
            for entry in _split_dmm_tile(inner):
                tpath = _dmm_type_path(entry)
                if not tpath.startswith("/"):
                    continue
                tgt = _make_id(tpath)
                if tgt in seen_targets:
                    continue
                seen_targets.add(tgt)
                edges.append({"source": file_nid, "target": tgt, "relation": "uses",
                              "context": "map", "confidence": "EXTRACTED",
                              "source_file": str_path,
                              "source_location": f"L{open_line}", "weight": 1.0})

    return {"nodes": nodes, "edges": edges}

_DMF_WINDOW_RE = re.compile(r'^\s*window\s+"([^"]+)"\s*$')

_DMF_ELEM_RE = re.compile(r'^\s*elem\s+"([^"]+)"\s*$')

_DMF_TYPE_RE = re.compile(r'^\s*type\s*=\s*(\S+)\s*$')

def extract_dmf(path: Path) -> dict:
    """Extract windows and controls from a .dmf interface file."""
    try:
        text = _read_source_text(path)
    except Exception as e:
        return {"nodes": [], "edges": [], "error": str(e)}

    str_path = str(path)
    stem = _file_stem(path)
    file_nid = _make_id(str(path))
    nodes: list[dict] = [{"id": file_nid, "label": path.name, "file_type": "code",
                           "source_file": str_path, "source_location": "L1"}]
    edges: list[dict] = []
    seen: set[str] = {file_nid}

    current_window_nid: str | None = None
    current_window_name: str | None = None
    current_elem_nid: str | None = None
    current_elem_name: str | None = None

    for line_idx, line in enumerate(text.splitlines(), start=1):
        m = _DMF_WINDOW_RE.match(line)
        if m:
            name = m.group(1)
            nid = _make_id(stem, "window", name)
            if nid not in seen:
                seen.add(nid)
                nodes.append({"id": nid, "label": f'window "{name}"', "file_type": "code",
                              "source_file": str_path, "source_location": f"L{line_idx}"})
                edges.append({"source": file_nid, "target": nid, "relation": "contains",
                              "confidence": "EXTRACTED", "source_file": str_path,
                              "source_location": f"L{line_idx}", "weight": 1.0})
            current_window_nid = nid
            current_window_name = name
            current_elem_nid = None
            current_elem_name = None
            continue
        m = _DMF_ELEM_RE.match(line)
        if m and current_window_nid is not None and current_window_name is not None:
            name = m.group(1)
            # Key on the window NAME, not its node id: the id already embeds
            # `stem`, which is absolute at extraction time. extract()'s
            # id-remap only rewrites the leading stem, so the copy nested in
            # the middle kept the checkout path (and OS username) in every
            # element id.
            nid = _make_id(stem, "elem", current_window_name, name)
            if nid not in seen:
                seen.add(nid)
                nodes.append({"id": nid, "label": f'elem "{name}"', "file_type": "code",
                              "source_file": str_path, "source_location": f"L{line_idx}"})
                edges.append({"source": current_window_nid, "target": nid,
                              "relation": "contains", "confidence": "EXTRACTED",
                              "source_file": str_path, "source_location": f"L{line_idx}",
                              "weight": 1.0})
            current_elem_nid = nid
            current_elem_name = name
            continue
        m = _DMF_TYPE_RE.match(line)
        if m and current_elem_nid is not None and current_elem_name is not None:
            ctype = m.group(1)
            for n in nodes:
                if n["id"] == current_elem_nid and " [" not in n["label"]:
                    n["label"] = f'elem "{current_elem_name}" [{ctype}]'
                    break

    return {"nodes": nodes, "edges": edges}

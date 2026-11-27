"""DM (BYOND DreamMaker) extractor.

Regex/line based on purpose: tree-sitter-dm exposes ``type_definition`` without a
``type_path`` child (the leading ``/`` is a sibling token), which made the
previous tree-sitter version emit nothing but the file node for whole files. A
line walker is also far faster on a 27 MB BYOND tree.

Before parsing, every string literal and comment span is blanked (newlines are
kept) so commented-out code and prose in strings cannot invent types, procs or
calls. String interpolation (``"[foo()]"``) is real code in BYOND and stays
visible to the call walk.

Besides graph nodes/edges the result carries a side table consumed by the
cross-file resolver in ``graphify.extract``:

``dm_types``
    ``{type_path: {"parent": path | None, "vars": {name: type_path}}}`` — typed
    member vars per type plus an explicit ``parent_type`` when one is written.
"""
from __future__ import annotations

import re

from pathlib import Path
from typing import Any
from graphify.extractors.base import (
    _file_stem,
    _make_id,
    _read_source_text,
)


_DM_INCLUDE_RE = re.compile(r'^\s*#include\s+["\']([^"\']+)["\']', re.IGNORECASE)
_DM_DEFINE_RE = re.compile(r'^#\s*define\s+([A-Za-z_][A-Za-z0-9_]*)')
_DM_PATH_RE = re.compile(r'^/([A-Za-z0-9_]+(?:/[A-Za-z0-9_]+)*)')
_DM_PROC_MARK_RE = re.compile(r'^proc/([A-Za-z0-9_]+)\s*\(')
_DM_VERB_MARK_RE = re.compile(r'^verb/([A-Za-z0-9_]+)\s*\(')
_DM_MEMBER_PROC_RE = re.compile(r'^([A-Za-z0-9_]+)\s*\(')
_DM_CALL_RE = re.compile(r'\b([A-Za-z_][A-Za-z0-9_]*)\s*\(')
_DM_NEW_RE = re.compile(r'\bnew\s+(/[A-Za-z0-9_/]+)')
_DM_NEW_TYPED_RE = re.compile(
    r'\bvar/([A-Za-z0-9_/]+?)(?:\s*\[[^\]]*\])?\s*=\s*new\s*\(')
_DM_ISTYPE_RE = re.compile(
    r'\b(?:istype|ispath|isobj|ismob|isatom)\s*\([^,()]*,\s*(/[A-Za-z0-9_/]+)')
_DM_LOCAL_VAR_RE = re.compile(r'\bvar/([A-Za-z0-9_/]+)')
_DM_PARENT_TYPE_RE = re.compile(r'^parent_type\s*=\s*(/[A-Za-z0-9_/]+)')
_DM_SUPER_RE = re.compile(r'\.\.\s*\(')
# DM identifiers are case-insensitive; only the runtime/builtin callables live
# here so a user proc that happens to share a name (``to_chat`` in vgstation) is
# still resolvable.
_DM_KEYWORDS = {
    "if", "else", "for", "while", "do", "switch", "case", "break", "continue",
    "return", "new", "delete", "spawn", "set", "src", "usr", "world", "global",
    "istype", "isnull", "locate", "min", "max", "abs", "rand", "pick", "prob",
    "sleep", "call", "view", "oview", "range", "get_step", "get_dir", "get_dist",
    "text", "sound", "icon", "image", "file", "matrix", "vector", "rgb", "hsv",
    "animate", "filter", "walk", "walk_to", "step", "step_to", "step_towards",
    "hearers", "viewers", "oviewers", "orange", "block", "flick", "alert",
    "input", "list", "vars", "type", "nameof", "defined", "assert",
}
_DM_BUILTIN_TYPES = frozenset({"list"})
# BYOND's built-in type tree. The path prefix is the implicit parent for user
# types, but the engine's top-level types are not path-nested: /obj and /turf are
# /atoms, /mob is an /atom/movable, and /atom and /client hang off /datum. Without
# this a chain walk stops at /mob or /obj and misses every /atom proc.
_DM_IMPLICIT_PARENTS = {
    "/atom": "/datum",
    "/atom/movable": "/atom",
    "/mob": "/atom/movable",
    "/obj": "/atom",
    "/turf": "/atom",
    "/area": "/atom",
    "/client": "/datum",
}
# Statements that can start a line with `(`. `new` is deliberately absent: a
# member proc called `New()` is a real declaration, while `new /obj/x()` starts
# with a space and a path, never `new(`.
_DM_STMT_KEYWORDS = frozenset({
    "if", "else", "for", "while", "do", "switch", "case", "break", "continue",
    "return", "spawn", "set", "del", "delete", "goto",
})


def _dm_mask(text: str) -> str:
    """Blank string literals and comments, keeping offsets and newlines intact.

    A plain regex cannot do this: DM string interpolation (``"[foo("]")]"``)
    nests quotes inside strings, and a ``//`` or ``/*`` inside a literal is not a
    comment. The mode stack handles:
      * code (normal),
      * string (``\\`` escapes, ``[`` opens an interpolation),
      * interp (real code, ``]`` returns to the enclosing string).
    Interpolated code is left untouched so calls inside ``"[get_area(src)]"``
    still count.
    """
    n = len(text)
    chars = list(text)
    modes = ["code"]
    i = 0
    while i < n:
        ch = text[i]
        mode = modes[-1]
        if mode == "code":
            if ch == '"':
                chars[i] = " "
                modes.append("string")
                i += 1
                continue
            if ch == "/" and i + 1 < n and text[i + 1] == "/":
                j = text.find("\n", i)
                if j < 0:
                    j = n
                for k in range(i, j):
                    chars[k] = " "
                i = j
                continue
            if ch == "/" and i + 1 < n and text[i + 1] == "*":
                j = text.find("*/", i + 2)
                j = n if j < 0 else j + 2
                for k in range(i, j):
                    if text[k] != "\n":
                        chars[k] = " "
                i = j
                continue
            i += 1
            continue
        if mode == "string":
            if ch == "\\":
                chars[i] = " "
                if i + 1 < n:
                    chars[i + 1] = " "
                i += 2
                continue
            if ch == '"':
                chars[i] = " "
                modes.pop()
                i += 1
                continue
            if ch == "[":
                chars[i] = " "
                modes.append("interp")
                i += 1
                continue
            chars[i] = " "
            i += 1
            continue
        # interp: real code until the matching `]`
        if ch == "]":
            chars[i] = " "
            modes.pop()
            i += 1
            continue
        if ch == '"':
            chars[i] = " "
            modes.append("string")
            i += 1
            continue
        if ch == "[":
            chars[i] = " "
            modes.append("interp")
            i += 1
            continue
        i += 1
    return "".join(chars)


def _dm_split_top(text: str, sep: str = ",") -> list[str]:
    """Split on ``sep`` at paren/bracket depth 0 (already masked, no strings)."""
    out: list[str] = []
    buf: list[str] = []
    depth = 0
    for ch in text:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == sep and depth == 0:
            out.append("".join(buf))
            buf = []
        else:
            buf.append(ch)
    tail = "".join(buf)
    if tail.strip():
        out.append(tail)
    return out


def _dm_decl_parts(decl: str) -> tuple[str, "str | None"]:
    """Split one DM declaration (``obj/item/thing/T``) into (name, type path).

    Accepts the shapes used by proc parameters, member vars and locals:
    ``var/mob/living/M as mob``, ``var/obj/item/thing/T[3] = null``,
    ``list/entries = list()``, and a bare ``count = 0``.
    """
    part = decl.strip()
    if not part:
        return "", None
    part = re.split(r"\s+as\s+", part, maxsplit=1)[0]
    part = re.split(r"\s*=\s*", part, maxsplit=1)[0].strip()
    if part.startswith("var/"):
        part = part[4:]
    part = re.split(r"[\s\[]", part, maxsplit=1)[0]
    segments = [s for s in part.split("/") if s]
    if not segments:
        return "", None
    name = segments[-1]
    type_path = "/" + "/".join(segments[:-1]) if len(segments) > 1 else None
    return name, type_path


def _dm_var_decls(body: str) -> list[tuple[str, "str | None"]]:
    """Every ``(name, type)`` declared in a ``var/...`` tail (comma lists too)."""
    out: list[tuple[str, "str | None"]] = []
    for part in _dm_split_top(body, ","):
        name, type_path = _dm_decl_parts(part)
        if name:
            out.append((name, type_path))
    return out


def _dm_signature(mlines: list[str], start: int, offset: int) -> tuple[str, int]:
    """Buffered proc signature; DM allows arguments to wrap onto later lines."""
    buf = mlines[start][offset:]
    depth = buf.count("(") - buf.count(")")
    end = start
    while depth > 0 and end + 1 < len(mlines):
        end += 1
        buf += " " + mlines[end]
        depth += mlines[end].count("(") - mlines[end].count(")")
    return buf, end


def _dm_params(sig: str) -> dict[str, str]:
    lp = sig.find("(")
    if lp < 0:
        return {}
    rp = sig.rfind(")")
    inner = sig[lp + 1: rp if rp > lp else len(sig)]
    out: dict[str, str] = {}
    for part in _dm_split_top(inner, ","):
        name, type_path = _dm_decl_parts(part)
        if name and type_path:
            out[name] = type_path
    return out


def _dm_call_kind(body: str, start: int) -> tuple["str | None", str]:
    """Classify a call site by what precedes the callee: (receiver, kind)."""
    j = start - 1
    while j >= 0 and body[j] in " \t":
        j -= 1
    if j < 0 or body[j] not in ".:":
        return None, "global"
    k = j - 1
    while k >= 0 and body[k] in " \t":
        k -= 1
    if body[j] == "." and k >= 0 and body[k] == ".":
        return None, "parent"
    if k < 0:
        return None, "self"
    if body[k].isalnum() or body[k] == "_":
        end = k
        while k >= 0 and (body[k].isalnum() or body[k] == "_"):
            k -= 1
        return body[k + 1:end + 1], "member"
    # `list["k"].foo()` / `f().foo()`: a receiver we cannot type here.
    return None, "member"


def _dm_parent_path(path: str) -> str:
    parent = path.rpartition("/")[0]
    if parent and parent != "/":
        return parent
    if path == "/datum":
        return ""
    # Every other top-level type inherits /datum; the engine's built-ins are the
    # only ones with a non-path parent (/obj is an /atom, /mob an /atom/movable).
    return _DM_IMPLICIT_PARENTS.get(path, "/datum")


def extract_dm(path: Path) -> dict:
    """Extract types, procs, verbs, vars, macros, includes and calls."""
    try:
        text = _read_source_text(path)
    except Exception as e:
        return {"nodes": [], "edges": [], "raw_calls": [], "dm_types": {},
                "error": str(e)}

    str_path = str(path)
    stem = _file_stem(path)
    file_nid = _make_id(str(path))
    nodes: list[dict] = [{"id": file_nid, "label": path.name, "file_type": "code",
                          "source_file": str_path, "source_location": "L1"}]
    edges: list[dict] = []
    raw_calls: list[dict] = []
    dm_types: dict[str, dict] = {}
    seen_ids: set[str] = {file_nid}
    proc_by_name: dict[str, list[str]] = {}
    proc_decls: list[dict] = []

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

    def type_record(dm_path: str) -> dict:
        return dm_types.setdefault(dm_path, {"parent": None, "vars": {}})

    def add_proc(owner: "str | None", proc_name: str, line: int,
                 body_start: int, decl_indent: int, sig: str = "") -> "str | None":
        if not proc_name:
            return None
        if owner:
            owner_nid = ensure_type(owner, line)
            nid = _make_id(owner, proc_name)
            add_node(nid, f"{owner}/{proc_name}()", line)
            add_edge(owner_nid, nid, "method", line)
        else:
            nid = _make_id(stem, proc_name)
            add_node(nid, f"{proc_name}()", line)
            add_edge(file_nid, nid, "contains", line)
        proc_by_name.setdefault(proc_name.lower(), []).append(nid)
        proc_decls.append({"nid": nid, "owner": owner, "name": proc_name,
                           "body_start": body_start, "indent": decl_indent,
                           "params": _dm_params(sig)})
        return nid

    def add_var(owner: str, type_path: "str | None", var_name: str,
                line: int) -> None:
        if not var_name:
            return
        owner_nid = ensure_type(owner, line)
        nid = _make_id(owner, "var", var_name)
        add_node(nid, f"{owner}/var/{var_name}", line)
        add_edge(owner_nid, nid, "contains", line)
        if type_path and type_path.lstrip("/").lower() not in _DM_BUILTIN_TYPES:
            type_record(owner)["vars"].setdefault(var_name, type_path)

    def add_macro(macro_name: str, line: int, body: str) -> None:
        nid = _make_id(stem, "define", macro_name)
        add_node(nid, f"#define {macro_name}", line)
        add_edge(file_nid, nid, "contains", line)
        # Calls expanded from the macro body belong to the macro, not to a proc.
        for cm in _DM_CALL_RE.finditer(body):
            callee = cm.group(1)
            if callee.lower() in _DM_KEYWORDS:
                continue
            raw_calls.append({
                "caller_nid": nid, "callee": callee, "kind": "global",
                "receiver": None, "is_member_call": False,
                "source_file": str_path,
                "source_location": f"L{line + body.count(chr(10), 0, cm.start())}",
                "language": "dm", "owner": None, "receiver_type": None,
            })

    masked = _dm_mask(text)
    lines = text.splitlines()
    mlines = masked.splitlines()

    # Bracket depth at the start of each masked line. A line that continues an
    # open `(`/`[` is expression text, not a declaration: a multi-line list such
    # as `var/blocked = list(\n\t/obj/item/x,\n\t)` otherwise reads as a type
    # definition and the following statements become members of that type.
    line_depth: list[int] = []
    depth = 0
    for mline in mlines:
        line_depth.append(depth)
        depth += (mline.count("(") + mline.count("[")
                  - mline.count(")") - mline.count("]"))
        if depth < 0:
            depth = 0

    current_type: "str | None" = None
    current_type_indent = 0
    member_indent: "int | None" = None

    for idx, mline in enumerate(mlines):
        s = mline.strip()
        if not s:
            continue
        indent = len(mline) - len(mline.lstrip())
        if line_depth[idx] > 0:
            continue

        if s.startswith("#"):
            original = lines[idx] if idx < len(lines) else mline
            inc = _DM_INCLUDE_RE.match(original)
            if inc:
                raw = inc.group(1).replace("\\", "/")
                norm = re.sub(r"^\./", "", raw)
                resolved = (path.parent / norm).resolve()
                tgt = _make_id(str(resolved)) if resolved.exists() else _make_id(norm)
                edges.append({
                    "source": file_nid, "target": tgt,
                    "relation": "imports_from" if resolved.exists() else "imports",
                    "context": "import", "confidence": "EXTRACTED",
                    "source_file": str_path, "source_location": f"L{idx + 1}",
                    "weight": 1.0,
                    **({} if resolved.exists() else {"external": True}),
                })
                continue
            if s.startswith("#define"):
                joined = [mline]
                j = idx
                while joined[-1].rstrip().endswith("\\") and j + 1 < len(mlines):
                    j += 1
                    joined.append(mlines[j])
                define = " ".join(joined)
                dm = _DM_DEFINE_RE.match(define)
                if dm:
                    add_macro(dm.group(1), idx + 1, define[dm.end():])
                continue
            continue

        if s.startswith("/"):
            m = _DM_PATH_RE.match(s)
            if not m:
                continue
            dm_path = "/" + m.group(1)
            rest = s[m.end():].strip()
            current_type = None
            member_indent = None

            if "/proc/" in dm_path or "/verb/" in dm_path:
                seg = "proc" if "/proc/" in dm_path else "verb"
                owner, _, name = dm_path.partition("/" + seg + "/")
                name = name.split("/")[0]
                if "(" in s or "(" in mline:
                    sig, sig_end = _dm_signature(mlines, idx, m.end())
                    add_proc(owner or None, name, idx + 1, sig_end + 1,
                             indent, sig)
                continue
            if "/var/" in dm_path:
                owner, _, rest_var = dm_path.partition("/var/")
                for var_name, type_path in _dm_var_decls(rest_var + " " + rest):
                    add_var(owner, type_path, var_name, idx + 1)
                continue
            if rest.startswith("("):
                owner, _, name = dm_path.rpartition("/")
                sig, sig_end = _dm_signature(mlines, idx, m.end())
                add_proc(owner or None, name, idx + 1, sig_end + 1, indent, sig)
                continue
            if "=" in rest:
                continue
            type_nid = ensure_type(dm_path, idx + 1)
            add_edge(file_nid, type_nid, "contains", idx + 1)
            type_record(dm_path)
            parent = _dm_parent_path(dm_path)
            if parent:
                add_edge(type_nid, _make_id(parent), "inherits", idx + 1)
            current_type, current_type_indent = dm_path, indent
            continue

        if current_type is None or indent <= current_type_indent:
            continue
        if member_indent is None:
            member_indent = indent
        if indent != member_indent:
            continue

        m = _DM_PROC_MARK_RE.match(s)
        if m:
            sig, sig_end = _dm_signature(mlines, idx, m.start(1))
            add_proc(current_type, m.group(1), idx + 1, sig_end + 1, indent, sig)
            continue
        m = _DM_VERB_MARK_RE.match(s)
        if m:
            sig, sig_end = _dm_signature(mlines, idx, m.start(1))
            add_proc(current_type, m.group(1), idx + 1, sig_end + 1, indent, sig)
            continue
        if s.startswith("var/"):
            for var_name, type_path in _dm_var_decls(s[4:]):
                add_var(current_type, type_path, var_name, idx + 1)
            continue
        m = _DM_PARENT_TYPE_RE.match(s)
        if m:
            type_record(current_type)["parent"] = m.group(1)
            continue
        m = _DM_MEMBER_PROC_RE.match(s)
        if m and m.group(1).lower() not in _DM_STMT_KEYWORDS:
            sig, sig_end = _dm_signature(mlines, idx, m.start(1))
            add_proc(current_type, m.group(1), idx + 1, sig_end + 1, indent, sig)

    # ── bodies: calls, instantiations, typed locals, type checks ─────────────
    for decl in proc_decls:
        nid = decl["nid"]
        owner = decl["owner"]
        body_start = decl["body_start"]
        end = body_start
        while end < len(mlines):
            ln = mlines[end]
            if ln.strip() and (len(ln) - len(ln.lstrip())) <= decl["indent"]:
                break
            end += 1
        body = "\n".join(mlines[body_start:end])
        if not body.strip():
            continue
        base_line = body_start + 1

        scope: dict[str, str] = dict(decl.get("params") or {})
        for vm in _DM_LOCAL_VAR_RE.finditer(body):
            var_name, type_path = _dm_decl_parts(vm.group(1))
            if var_name and type_path:
                scope.setdefault(var_name, type_path)
        # Typed member vars of the enclosing type (and its same-file ancestors).
        chain = []
        walk = owner or ""
        while walk and walk != "/":
            chain.append(walk)
            walk = _dm_parent_path(walk)
        for type_path in chain:
            for var_name, var_type in (dm_types.get(type_path, {}).get("vars") or {}).items():
                scope.setdefault(var_name, var_type)

        def line_for(pos: int) -> int:
            return base_line + body.count("\n", 0, pos)

        def add_raw(callee: str, kind: str, receiver: "str | None",
                    line_no: int, member: bool, rtype: "str | None") -> None:
            raw_calls.append({
                "caller_nid": nid, "callee": callee, "kind": kind,
                "receiver": receiver, "is_member_call": member,
                "source_file": str_path, "source_location": f"L{line_no}",
                "language": "dm", "owner": owner, "receiver_type": rtype,
            })

        for cm in _DM_CALL_RE.finditer(body):
            callee = cm.group(1)
            if callee.lower() in _DM_KEYWORDS:
                continue
            line_no = line_for(cm.start())
            receiver, kind = _dm_call_kind(body, cm.start())
            if kind == "self":
                continue
            if kind == "parent":
                add_raw(decl["name"], kind, None, line_no, True, None)
                continue
            if kind == "member":
                if receiver in ("world", "global", "vars"):
                    kind, receiver = "global", None
                rtype = None
                if receiver == "src" and owner:
                    rtype = owner
                elif receiver == "usr":
                    rtype = "/mob"
                elif receiver:
                    rtype = scope.get(receiver)
                add_raw(callee, kind, receiver, line_no, True, rtype)
                continue
            hits = proc_by_name.get(callee.lower(), [])
            if len(hits) == 1:
                add_edge(nid, hits[0], "calls", line_no, context="call")
            else:
                add_raw(callee, "global", None, line_no, False, None)

        for sm in _DM_SUPER_RE.finditer(body):
            add_raw(decl["name"], "parent", None, line_for(sm.start()), True, None)

        for nm in _DM_NEW_RE.finditer(body):
            target = "/" + nm.group(1).strip("/")
            add_edge(nid, _make_id(target), "instantiates", line_for(nm.start()),
                     context="call")
        for nm in _DM_NEW_TYPED_RE.finditer(body):
            _, type_path = _dm_decl_parts(nm.group(1))
            if type_path:
                add_edge(nid, _make_id(type_path), "instantiates",
                         line_for(nm.start()), context="call")

        for im in _DM_ISTYPE_RE.finditer(body):
            target = "/" + im.group(1).strip("/")
            add_edge(nid, _make_id(target), "references", line_for(im.start()),
                     context="typecheck")

    return {"nodes": nodes, "edges": edges, "raw_calls": raw_calls,
            "dm_types": dm_types}


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

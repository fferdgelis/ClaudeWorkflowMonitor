"""Pure disk readers and line/literal parsers for the Workflow Monitor. ZERO mutable state.

Reads the state Claude Code writes on disk (without touching the sessions):
  ~/.claude/projects/<project>/<session>/subagents/workflows/<wf_run>/
      journal.jsonl        {"type":"started"|"result","agentId":...}
      agent-<id>.jsonl     the agent's transcript (one JSON line per event)
      agent-<id>.meta.json {"agentType","description","spawnDepth"}
  ~/.claude/projects/<project>/<session>/workflows/scripts/<name>-<wf_run>.js

LINE 0 of each agent-<id>.jsonl is the full prompt the agent was launched with.

Language convention (whole app): code — identifiers, comments, docstrings — is
English; everything that crosses the HTTP boundary (JSON keys and values, state
labels, error messages) stays in SPANISH byte-for-byte — see server.py. The same
goes for regex BODIES that match Spanish prompt text ("SOS EL X", "Directorio").

Root of the import DAG: fsread <- prompts <- api <- server. Stdlib only.
"""
from __future__ import annotations

import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import NamedTuple

# Where Claude Code writes its state. The env var lets tests point at fixtures and
# lets the monitor run as a service without HOME defined. Read ONCE at import time;
# consumers must access it as `fsread.ROOT` (attribute) so tests can monkeypatch it.
ROOT = Path(os.environ.get("CLAUDE_PROJECTS_DIR") or Path.home() / ".claude" / "projects")

RE_TOOL = re.compile(r'"type":"tool_use".{0,200}?"name":"([^"]+)"')
RE_HINT = re.compile(r'"(?:file_path|pattern|sql|tableName|procedureName|command)":"((?:[^"\\]|\\.){5,120})')
RE_TEXT = re.compile(r'"type":"text","text":"((?:[^"\\]|\\.){1,200})')
RE_TS = re.compile(r'"timestamp":"([^"]+)"')
RE_MISSION = re.compile(r'SOS (?:EL |LA )?([A-ZÁÉÍÓÚÑ][A-ZÁÉÍÓÚÑ ]{3,60})')
RE_CONTENT = re.compile(r'"content":"((?:[^"\\]|\\.){10,260})')
RE_MISSION_LOOSE = re.compile(r'\b[Ss]os (?:el |la |un |una )?([^.;\n]{5,70})')
RE_AGENT_ID = re.compile(r'"agentId":"([a-f0-9]+)"')
RE_KEY = re.compile(r'"key":"([^"]+)"')
# Context boilerplate with a path at the start of the prompt: a known label, a Windows
# drive or a typical POSIX root (Linux/macOS). The label alternatives are real prompt
# boilerplate (Spanish included) — single source: both _strip_boilerplate and
# agent_mission skip with THIS regex (they used to have diverging inline copies).
RE_PATH_LABEL = re.compile(r"^(?:Repo|Workdir|Working ?dir|Directorio|Carpeta|Contexto)\b", re.I)
RE_PATH_HINT = re.compile(r"[A-Za-z]:[\\/]|(?:^|[\s:])(?:/(?:home|Users|tmp|var|opt|srv|mnt)/|~/)")

TAIL_BYTES = 512 * 1024
HEAD_BYTES = 64 * 1024
LINE0_CHUNK_BYTES = 4 * 1024 * 1024    # read block for line 0 (the largest prompt measures 400 KB)
LINE0_MAX_TOTAL_BYTES = 32 * 1024 * 1024  # hard cap: beyond this line 0 is deemed unreadable


def unescape(s: str) -> str:
    return s.replace("\\n", " ").replace('\\"', '"').replace("\\\\", "\\")


def local_hhmmss(line: str) -> str:
    m = RE_TS.search(line)
    if not m:
        return ""
    try:
        dt = datetime.fromisoformat(m.group(1).replace("Z", "+00:00"))
        return dt.astimezone().strftime("%H:%M:%S")
    except ValueError:
        return ""


def tail_lines(path: Path, max_bytes: int = TAIL_BYTES) -> list[str]:
    try:
        size = path.stat().st_size
        with path.open("rb") as f:
            if size > max_bytes:
                f.seek(size - max_bytes)
                f.readline()  # discard the partial line
            return f.read().decode("utf-8", "replace").splitlines()
    except OSError:
        return []


def head_lines(path: Path, n: int = 3) -> list[str]:
    try:
        with path.open("rb") as f:
            chunk = f.read(HEAD_BYTES).decode("utf-8", "replace")
        return chunk.splitlines()[:n]
    except OSError:
        return []


class JournalInfo(NamedTuple):
    """What the run's journal.jsonl says. A started agent whose KEY was later
    completed by another agent is a REPLACED one (resume) — see api_run."""
    key_by_agent: dict[str, str | None]   # agentId -> key of its prompt
    done_agents: set[str]                 # agentIds with a "result" line
    done_keys: set[str]                   # keys with a "result" line
    last_type: str | None                 # "started" | "result" | None (empty journal)


def journal_info(run_dir: Path) -> JournalInfo:
    """Dead agents of a run that was later resumed stay 'started' with no 'result'
    forever; their KEY (prompt hash) may have been completed by the replacement
    agent -> that is what tells REEMPLAZADO apart from MUERTO.
    """
    key_by_agent, done_agents, done_keys = {}, set(), set()
    last_type = None
    j = run_dir / "journal.jsonl"
    if j.exists():
        for line in tail_lines(j, 2 * 1024 * 1024):
            a = RE_AGENT_ID.search(line)
            k = RE_KEY.search(line)
            if '"type":"started"' in line:
                last_type = "started"
                if a:
                    key_by_agent[a.group(1)] = k.group(1) if k else None
            elif '"type":"result"' in line:
                last_type = "result"
                if a:
                    done_agents.add(a.group(1))
                if k:
                    done_keys.add(k.group(1))
    return JournalInfo(key_by_agent, done_agents, done_keys, last_type)


def workflow_name(run_dir: Path) -> str:
    session_dir = run_dir.parents[2]
    for s in (session_dir / "workflows" / "scripts").glob(f"*{run_dir.name}.js"):
        return s.stem.removesuffix(f"-{run_dir.name}")
    return ""


def list_agent_files(run_dir: Path) -> list[dict]:
    out = []
    for f in run_dir.glob("agent-*.jsonl"):
        # Some paths exceed MAX_PATH: glob enumerates them but stat() blows up. A whole
        # run cannot be lost because of one unreachable file.
        try:
            st = f.stat()
        except OSError:
            continue
        out.append({"id": f.stem.removeprefix("agent-"), "path": f, "mtime": st.st_mtime, "kb": st.st_size // 1024})
    out.sort(key=lambda a: a["mtime"], reverse=True)
    return out


def clean_project_slug(slug: str) -> str:
    # Windows slug ("c--Net-8-Tools") or Linux/macOS ("-home-<user>-...", "-Users-<user>-...").
    return re.sub(r"^[Cc]--(Net-8-)?|^-(?:home|Users)-[^-]+-", "", slug)


def _content_text(content) -> str:
    """message.content is a str in all 2525 measured prompts, but may come as blocks."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for b in content:
            if not isinstance(b, dict):
                continue
            if b.get("type") == "text":
                # str() and not b.get("text"): a "text" that is an int or a list made the
                # join below blow up with TypeError -> 500 on all four endpoints, including
                # the global /api/search. Same guard as _meta_str, on the very field the
                # prompt text comes from.
                t = b.get("text")
                parts.append(t if isinstance(t, str) else ("" if t is None else str(t)))
            else:
                parts.append("[" + str(b.get("type") or "bloque") + "]")
        return "\n".join(parts)
    return ""


def _agent_meta(path: Path) -> dict:
    """agent-<id>.meta.json: agentType / description / spawnDepth. May not exist."""
    try:
        meta = json.loads(path.with_name(path.stem + ".meta.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    # The .meta.json is written by another process: it can be valid JSON and NOT be an
    # object ([1,2,3], null). Without this, meta.get() raises AttributeError and a single
    # bad file returns 500 on /api/run, /api/prompts, /api/prompt and — worst of all — on
    # /api/search, which sweeps the whole disk.
    return meta if isinstance(meta, dict) else {}


def _meta_str(v) -> str:
    """Metadata field that ends up in the response JSON: discarded unless it is a string.
    A numeric agentType used to break sorted(types) in /api/search (str vs int -> TypeError)."""
    return v if isinstance(v, str) else ""


def _local_ts_label(iso: str) -> str:
    try:
        return datetime.fromisoformat(iso.replace("Z", "+00:00")).astimezone().strftime("%d/%m %H:%M")
    except (ValueError, AttributeError):
        return ""


def _read_line0(path: Path) -> tuple[bytes | None, bool]:
    """Bytes of line 0 and whether it had to be cut. readline(N) cuts at N bytes WITHOUT
    warning: a line 0 larger than the cap came back with the JSON split in half and the
    prompt was reported as nonexistent ("not written yet"), which is false. Keep reading
    until the \\n."""
    parts, bytes_read = [], 0
    try:
        with path.open("rb") as f:
            while bytes_read < LINE0_MAX_TOTAL_BYTES:
                chunk = f.readline(LINE0_CHUNK_BYTES)
                if not chunk:
                    break
                parts.append(chunk)
                bytes_read += len(chunk)
                if chunk.endswith(b"\n"):
                    break
    except OSError:
        return None, False
    return b"".join(parts), bytes_read >= LINE0_MAX_TOTAL_BYTES


def _strip_boilerplate(line: str) -> str:
    """Strips the leading context sentences with a path ("Repo: c:\\Net 8\\MiRepo (rama main).",
    "Workdir: ..."). They are identical across the whole fan-out: left in front, the Mission
    column shows the repo path and stops telling the agents apart.
    CAREFUL: the paths contain spaces ("c:\\Net 8\\...") — cutting on \\S+ does not work.
    Returns "" if the whole line was boilerplate: the caller moves on to the next one."""
    for _ in range(2):  # sometimes there are two context sentences in a row
        head, sep, rest = line.partition(". ")
        if not (RE_PATH_HINT.search(head) or RE_PATH_LABEL.match(head)):
            return line
        if not sep or not rest.strip():
            return ""
        line = rest.lstrip(". ")
    return line


def agent_mission(path: Path) -> str:
    """Raw-regex fallback over the file head, for the just-spawned agent whose line 0
    is not readable yet. The cache-based path is prompts.mission_of."""
    for line in head_lines(path):
        m = RE_MISSION.search(line)
        if m:
            return m.group(1).strip().title()
    for line in head_lines(path):
        m = RE_CONTENT.search(line)
        if m:
            txt = unescape(m.group(1))
            s = RE_MISSION_LOOSE.search(txt)
            if s:
                return s.group(1).strip()
            return _strip_boilerplate(txt)[:70] or "?"
    return "?"


def agent_activity(path: Path) -> str:
    for line in reversed(tail_lines(path, 64 * 1024)):
        if '"type":"assistant"' in line:
            m = RE_TOOL.search(line)
            if m:
                hint = ""
                h = RE_HINT.search(line)
                if h:
                    hint = unescape(h.group(1)).replace("\\", "/").rsplit("/", 1)[-1][:60]
                return f"{m.group(1)} {hint}".strip()
            if '"type":"text"' in line:
                return "(redactando)"
    return ""


# ---------------------------------------------------------------- conversations
# The MAIN conversation of a session is <project>/<session>.jsonl -- a SIBLING of the
# <session>/ directory that holds its subagents. Nothing else in this app reads it: the
# monitor was built to watch subagents, so a project where you only ever chat (no Agent
# tool, no Workflow, no background skill) produced no rows at all and did not even appear
# in the project filter.
#
# Same JSONL event shape as an agent transcript (assistant/user/tool_use/tool_result),
# plus session-level records that exist ONLY here:
#   {"type":"custom-title","customTitle":...}  title the user set        -> wins
#   {"type":"ai-title","aiTitle":...}          title Claude Code derived
#   {"type":"last-prompt","lastPrompt":...}    the last thing you asked
# Each of those is APPENDED AGAIN every time it changes, so the LAST occurrence is the
# current value. They are read from the TAIL and never by parsing the whole file: these
# are the biggest files on disk (31 MB here) and /api/runs sweeps all of them every 4 s.
# \s* around the colon unlike the regexes above: those match Claude Code's own compact
# output, but these three also have to survive a pretty-printed line ('"key": "value"').
RE_CUSTOM_TITLE = re.compile(r'"customTitle"\s*:\s*"((?:[^"\\]|\\.){1,200})')
RE_AI_TITLE = re.compile(r'"aiTitle"\s*:\s*"((?:[^"\\]|\\.){1,200})')
RE_LAST_PROMPT = re.compile(r'"lastPrompt"\s*:\s*"((?:[^"\\]|\\.){1,300})')

CONV_TAIL_BYTES = 128 * 1024


def conversation_paths():
    """<project>/<session>.jsonl. The <session>/ directories and memory/ are not files,
    so this pattern yields exactly the conversation transcripts."""
    yield from ROOT.glob("*/*.jsonl")


def conversation_meta(path: Path) -> dict:
    """Current title and last prompt of a conversation, from its tail.

    Forward pass keeping the last hit: the records repeat, and the last one wins. A chat
    young enough that its whole history still fits before the tail window has its titles
    at the TOP, hence the head fallback -- without it a brand-new chat showed up unnamed.
    """
    custom = ai = last = ""
    for line in tail_lines(path, CONV_TAIL_BYTES):
        m = RE_CUSTOM_TITLE.search(line)
        if m:
            custom = unescape(m.group(1))
        m = RE_AI_TITLE.search(line)
        if m:
            ai = unescape(m.group(1))
        m = RE_LAST_PROMPT.search(line)
        if m:
            last = unescape(m.group(1))
    if not (custom or ai):
        for line in head_lines(path, 8):
            m = RE_CUSTOM_TITLE.search(line)
            if m:
                custom = unescape(m.group(1))
            m = RE_AI_TITLE.search(line)
            if m:
                ai = unescape(m.group(1))
    return {"titulo": (custom or ai).strip(), "ultimoPrompt": last.strip()}


def conversation_first_prompt(path: Path) -> str:
    """The message that opened the conversation. json.loads per line and not a regex:
    this one is shown in full, so it must not come back with broken escapes."""
    for line in head_lines(path, 60):
        try:
            o = json.loads(line)
        except ValueError:
            continue
        if isinstance(o, dict) and o.get("type") == "user":
            msg = o.get("message")
            text = _content_text(msg.get("content") if isinstance(msg, dict) else None)
            if text.strip():
                return text
    return ""


def _iter_agent_paths():
    """The only two path shapes that exist. rglob() is out: it steps on dirs beyond MAX_PATH."""
    for pattern in ("*/*/subagents/agent-*.jsonl", "*/*/subagents/workflows/wf_*/agent-*.jsonl"):
        yield from ROOT.glob(pattern)


def _run_script(run_dir: Path) -> tuple[Path | None, str | None]:
    """(path, text) of the workflow's .js, or (None, None). An OSError here is normal:
    some paths exceed MAX_PATH, glob enumerates them and read_text fails (1 of 159 in
    the corpus)."""
    try:
        candidates = list((run_dir.parents[2] / "workflows" / "scripts").glob(f"*{run_dir.name}.js"))
    except OSError:
        return None, None
    for s in candidates:
        try:
            return s, s.read_text(encoding="utf-8", errors="replace")[:400 * 1024]
        except OSError:
            return s, None
    return None, None


# ------------------------------------------------------------------ action plan
# The declared PLAN of a workflow is the `export const meta = {...}` of its script: name,
# description and the phases with their detail. By contract of the Workflow tool that meta
# is a PURE LITERAL (no variables, no interpolation), so it can be read without evaluating JS.
#
# It is parsed with a small literal reader instead of a regex: the detail fields carry
# commas ("escaneo por categoria: secretos, PII, inventario") and a regex takes those as
# end-of-field. Measured over the 159 scripts in the corpus: 158 parse; the only one that
# does not is a MAX_PATH case.

RE_META = re.compile(r"export\s+const\s+meta\s*=\s*\{")


class _JsLiteralReader:
    """Reads objects/arrays/strings out of a JS literal. Tolerates unquoted keys,
    single/double/backtick quotes, trailing commas and comments."""

    def __init__(self, s):
        self.s, self.i = s, 0

    def _skip_trivia(self):
        while self.i < len(self.s):
            c = self.s[self.i]
            if c in " \t\r\n,":
                self.i += 1
            elif self.s.startswith("//", self.i):
                j = self.s.find("\n", self.i)
                self.i = len(self.s) if j < 0 else j + 1
            elif self.s.startswith("/*", self.i):
                j = self.s.find("*/", self.i)
                self.i = len(self.s) if j < 0 else j + 2
            else:
                return

    def value(self):
        self._skip_trivia()
        if self.i >= len(self.s):
            return None
        c = self.s[self.i]
        if c == "{":
            return self.obj()
        if c == "[":
            return self.array()
        if c in "'\"`":
            return self.string()
        j = self.i
        while j < len(self.s) and self.s[j] not in ",}]\n":
            j += 1
        raw, self.i = self.s[self.i:j].strip(), j
        if raw in ("true", "false"):
            return raw == "true"
        if raw == "null":
            return None
        try:
            return float(raw) if "." in raw else int(raw)
        except ValueError:
            return raw

    def string(self):
        q, buf = self.s[self.i], []
        self.i += 1
        while self.i < len(self.s):
            c = self.s[self.i]
            if c == "\\" and self.i + 1 < len(self.s):
                buf.append({"n": "\n", "t": "\t", "r": "\r"}.get(self.s[self.i + 1], self.s[self.i + 1]))
                self.i += 2
                continue
            if c == q:
                self.i += 1
                break
            buf.append(c)
            self.i += 1
        return "".join(buf)

    def key(self):
        self._skip_trivia()
        if self.i < len(self.s) and self.s[self.i] in "'\"`":
            k = self.string()
        else:
            j = self.i
            while j < len(self.s) and (self.s[j].isalnum() or self.s[j] in "_$"):
                j += 1
            k, self.i = self.s[self.i:j], j
        self._skip_trivia()
        if self.i < len(self.s) and self.s[self.i] == ":":
            self.i += 1
        return k

    def obj(self):
        self.i += 1
        out = {}
        while self.i < len(self.s):
            self._skip_trivia()
            if self.i >= len(self.s) or self.s[self.i] == "}":
                self.i += 1
                break
            k = self.key()
            if not k:
                self.i += 1
                continue
            out[k] = self.value()
        return out

    def array(self):
        self.i += 1
        out = []
        while self.i < len(self.s):
            self._skip_trivia()
            if self.i >= len(self.s) or self.s[self.i] == "]":
                self.i += 1
                break
            out.append(self.value())
        return out


def parse_plan(src: str) -> dict | None:
    """The script's declared plan, or None if it declares no meta."""
    m = RE_META.search(src or "")
    if not m:
        return None
    try:
        o = _JsLiteralReader(src[m.end() - 1:]).obj()
    except Exception:          # noqa: BLE001 - a weird script cannot take the endpoint down
        return None
    if not isinstance(o, dict):
        return None
    phases = []
    for f in (o.get("phases") or []):
        if isinstance(f, dict) and isinstance(f.get("title"), str) and f["title"]:
            phases.append({"titulo": f["title"][:120],
                           "detalle": _meta_str(f.get("detail"))[:400],
                           "model": _meta_str(f.get("model"))[:40]})
    return {"nombre": _meta_str(o.get("name"))[:120],
            "descripcion": _meta_str(o.get("description"))[:400],
            "cuando": _meta_str(o.get("whenToUse"))[:400],
            "fases": phases}

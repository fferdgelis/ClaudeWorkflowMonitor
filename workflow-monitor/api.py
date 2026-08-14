"""Endpoint/domain layer of the Workflow Monitor: one api_* function per endpoint.

Language convention (whole app): code is English; the HTTP contract is SPANISH
byte-for-byte — JSON keys ("agentes", "listos", "estado", "proyecto", "mision",
"molde", ...), state values (ACTIVO/LENTO/TERMINADO/REEMPLAZADO/MUERTO/ESTANCADO),
event types (TOOL/DICE/RES), motivo values ('sueltos'/'sin-script'/'ilegible'/
'sin-meta'), the 'sueltos_' run-id prefix, the '(agentes sueltos)' label and every
error message. See server.py.

ROOT is always read as fsread.ROOT (attribute access) so tests can monkeypatch it
in one place.
"""
from __future__ import annotations

import re
import time
from datetime import datetime
from pathlib import Path
from stat import S_ISREG

import fsread
import prompts

# The ids come from the client and are used to build paths: validate them before
# touching disk.
RE_VALID_RUN_ID = re.compile(
    r"^(?:wf_[A-Za-z0-9_-]{1,60}|sueltos_[A-Za-z0-9-]{1,60}"
    r"|chat_[A-Za-z0-9-]{1,60}|codex_[A-Za-z0-9-]{1,60})$")
RE_VALID_AGENT_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

SEARCH_MAX_AGENTS = 200          # max agents returned (the real total is reported anyway)

# Status thresholds, in seconds — one single place: used by run_status, api_runs and api_run.
AGENT_ACTIVE_SECS = 60           # transcript written less than this ago -> ACTIVO
AGENT_IDLE_SECS = 300            # idle longer than this -> MUERTO/REEMPLAZADO (loose: "listo")
RUN_ACTIVE_SECS = 90             # some agent wrote less than this ago -> run ACTIVO


def run_status(agents: list[dict], last_journal_type: str | None, now: float | None = None) -> str:
    # A run that closed cleanly ends its journal with a "result"; a crashed one is left
    # with dangling "started" lines (deaths do not write the journal).
    # api_runs injects `now` so estado and edadSeg share one clock: the full sweep can
    # take seconds, and a fresh time.time() here could disagree with the row's age.
    freshest = max((a["mtime"] for a in agents), default=0)
    if (time.time() if now is None else now) - freshest < RUN_ACTIVE_SECS:
        return "ACTIVO"
    if agents and last_journal_type == "result":
        return "TERMINADO"
    return "ESTANCADO"


def find_run_dir(run_id: str):
    """Directory holding the run's agent-*.jsonl. 'sueltos_<session>' -> that session's subagents.

    The run_id comes from the client and goes into a glob: it is validated HERE, the one
    point ALL endpoints pass through. /api/run and /api/agent used to glob with the raw
    string and a '../' escaped ~/.claude (the monitor served files from anywhere on
    disk); on top of that, an empty run matched the 'workflows' directory itself and
    answered 200.
    """
    if not RE_VALID_RUN_ID.match(run_id or ""):
        return None
    if run_id.startswith("sueltos_"):
        sess = run_id.removeprefix("sueltos_")
        for sub in fsread.ROOT.glob(f"*/{sess}/subagents"):
            return sub
        return None
    for run_dir in fsread.ROOT.glob(f"*/*/subagents/workflows/{run_id}"):
        return run_dir
    return None


def _run_row(run: str, workflow: str, project: str, session: str,
             agents: int, done: int, active: int, status: str, mtime: float, now: float,
             kind: str = "workflow", kb: int = 0, tokens: int | None = None) -> dict:
    """Canonical shape of one /api/runs row. The only place a row formats ultimaAct
    (prompt timestamps have their own formatter: fsread._local_ts_label).

    `listos` and `activos` do NOT add up to `agentes`: a dead agent counts in neither,
    so the list shows both instead of one ratio that silently hides the difference.

    `tipo` ('workflow' | 'sueltos' | 'chat' | 'codex') exists so the client can tell them
    apart WITHOUT sniffing the run-id prefix: a chat has no agent counters to show, and
    printing "0/0" for it would read like a broken run instead of a conversation.

    `tokens` is None for everything except Codex. Claude Code's transcripts carry no token
    accounting at all, so the column stays empty for its rows rather than showing a zero
    that would read as "consumed nothing".
    """
    return {"run": run, "workflow": workflow, "proyecto": project, "sesion": session,
            "agentes": agents, "listos": done, "activos": active, "estado": status,
            "ultimaAct": datetime.fromtimestamp(mtime).strftime("%d/%m %H:%M"),
            "edadSeg": int(now - mtime), "tipo": kind, "kb": kb, "tokens": tokens}


def _loose_agent_runs(now: float) -> list[dict]:
    # LOOSE agents (Agent tool / skills in background, outside workflows): they live
    # right under <session>/subagents/agent-*.jsonl and have no journal.
    rows = []
    for sub in fsread.ROOT.glob("*/*/subagents"):
        # is_file() does not protect the stat() on the next line: the file can vanish
        # right in between (or exceed MAX_PATH) and there /api/runs returned 500, i.e.
        # the whole dashboard went blank. A run cannot be lost over one unreachable file.
        mtimes = []
        for f in sub.glob("agent-*.jsonl"):
            try:
                st = f.stat()
            except OSError:
                continue
            if S_ISREG(st.st_mode):     # we already have the mode: no extra stat on is_file()
                mtimes.append(st.st_mtime)
        if not mtimes:
            continue
        mtime = max(mtimes)
        sess = sub.parent
        rows.append(_run_row(
            "sueltos_" + sess.name, "(agentes sueltos)",
            fsread.clean_project_slug(sess.parent.name), sess.name[:8],
            agents=len(mtimes),
            done=sum(1 for t in mtimes if now - t > AGENT_IDLE_SECS),
            active=sum(1 for t in mtimes if now - t < AGENT_ACTIVE_SECS),
            status="ACTIVO" if now - mtime < RUN_ACTIVE_SECS else "TERMINADO",
            mtime=mtime, now=now, kind="sueltos"))
    return rows


def find_conversation(run_id: str):
    """<project>/<session>.jsonl for a 'chat_<session>' id, or None.

    Same discipline as find_run_dir: the id comes from the client and goes into a glob,
    so it is validated against the charset FIRST -- no dots, no separators, none of
    glob's metacharacters (*?[]) survive it.
    """
    if not RE_VALID_RUN_ID.match(run_id or "") or not run_id.startswith("chat_"):
        return None
    sess = run_id.removeprefix("chat_")
    for p in fsread.ROOT.glob(f"*/{sess}.jsonl"):
        return p
    return None


def _conversation_runs(now: float) -> list[dict]:
    """One row per CONVERSATION. These are not runs of anything -- they are the chats
    themselves -- but they ride in the same list on purpose: what you want to see is
    everything that is moving right now, and until this existed a project where you only
    ever chatted was invisible to the monitor and missing from the project filter.

    A chat has no journal and no notion of 'finished', so it is only ACTIVO or INACTIVO;
    reporting TERMINADO would claim something the file cannot support.
    """
    rows = []
    for path in fsread.conversation_paths():
        try:
            st = path.stat()
        except OSError:      # vanished mid-sweep, or past MAX_PATH
            continue
        if not S_ISREG(st.st_mode):
            continue
        meta = prompts.conversation_meta(path, st.st_size)
        sess = path.stem
        rows.append(_run_row(
            "chat_" + sess, meta["titulo"] or "(chat sin titulo)",
            fsread.clean_project_slug(path.parent.name), sess[:8],
            agents=0, done=0, active=0,
            status="ACTIVO" if now - st.st_mtime < RUN_ACTIVE_SECS else "INACTIVO",
            mtime=st.st_mtime, now=now, kind="chat", kb=st.st_size // 1024))
    return rows


def _workflow_runs(now: float) -> list[dict]:
    rows = []
    for run_dir in fsread.ROOT.glob("*/*/subagents/workflows/wf_*"):
        if not run_dir.is_dir():
            continue
        agents = fsread.list_agent_files(run_dir)
        ji = fsread.journal_info(run_dir)
        try:  # same reason: a run with no agents leans on the stat of the directory itself
            mtime = max((a["mtime"] for a in agents), default=run_dir.stat().st_mtime)
        except OSError:
            continue
        rows.append(_run_row(
            run_dir.name, fsread.workflow_name(run_dir),
            fsread.clean_project_slug(run_dir.parents[3].name), run_dir.parents[2].name[:8],
            agents=len(agents), done=len(ji.done_agents),
            # Same order of checks as api_run's per-agent status: an agent that just wrote
            # its result would still look "recent", so done wins over recency.
            active=sum(1 for a in agents
                       if a["id"] not in ji.done_agents and now - a["mtime"] < AGENT_ACTIVE_SECS),
            status=run_status(agents, ji.last_type, now), mtime=mtime, now=now))
    return rows


def find_codex(run_id: str):
    """The rollout for a 'codex_<session_id>' id, or None. Same discipline as the other
    finders: the id is validated against its charset BEFORE it reaches a glob."""
    if not RE_VALID_RUN_ID.match(run_id or "") or not run_id.startswith("codex_"):
        return None
    sess = run_id.removeprefix("codex_")
    for p in fsread.CODEX_ROOT.glob(f"*/*/*/rollout-*-{sess}.jsonl"):
        return p
    return None


def _codex_estado(ciclo: str | None, age: int) -> str:
    """Lifecycle first, recency second -- the same order api_run uses per agent.

    A rollout whose last lifecycle event is task_started and that stopped being written
    is a session that died or is hung: that is ESTANCADO, not TERMINADO. And 'ciclo is
    None' is honest ignorance, not a state: only the tail is read, so a very long session
    can have both of its lifecycle events outside the window.
    """
    if ciclo == "complete":
        return "TERMINADO"
    if ciclo == "started":
        return "ACTIVO" if age < RUN_ACTIVE_SECS else "ESTANCADO"
    return "?"


def _codex_runs(now: float) -> list[dict]:
    """One row per Codex session. These are not Claude Code at all -- they ride in the
    same list because what you want to see is everything that is working, whoever runs it,
    and because the cwd of a Codex session frequently points INSIDE the scratchpad of a
    Claude Code session that launched it."""
    rows = []
    for path in fsread.codex_paths():
        sess = fsread.codex_session_id(path)
        if not sess:
            continue
        try:
            st = path.stat()
        except OSError:
            continue
        if not S_ISREG(st.st_mode):
            continue
        e = prompts.codex_entry(path, st.st_size)
        meta = e["meta"] or {}
        age = int(now - st.st_mtime)
        rows.append(_run_row(
            "codex_" + sess,
            e["ultimoMensaje"] or Path(str(meta.get("cwd") or "")).name or "(sesion de Codex)",
            fsread.codex_project(meta),
            sess[:8],
            agents=0, done=0, active=0,
            status=_codex_estado(e["ciclo"], age),
            mtime=st.st_mtime, now=now, kind="codex", kb=st.st_size // 1024,
            tokens=(e["tokens"] or {}).get("total_tokens")))
    return rows


def api_runs() -> list[dict]:
    now = time.time()
    # Mixed in one list and sorted by recency: what matters is what moved last, whether
    # that was a workflow, a loose agent or you typing in a chat.
    runs = (_loose_agent_runs(now) + _workflow_runs(now)
            + _conversation_runs(now) + _codex_runs(now))
    runs.sort(key=lambda r: r["edadSeg"])
    return runs


def _uptime_secs(entry: dict | None, now: float) -> int | None:
    """Seconds since the agent was SPAWNED, or None if its line 0 is not readable yet.

    Two different clocks, both needed and previously conflated in the UI: `edadSeg` is
    now - mtime, i.e. how long the agent has been QUIET (its last write), while this one
    is how long it has been RUNNING. An agent 12 minutes into its work that just wrote a
    tool call showed "hace 0s", which reads as "just started".

    The spawn instant is the timestamp of line 0, written at spawn and never rewritten,
    and prompt_entry has already parsed and cached it: this costs nothing extra.
    """
    if not entry:
        return None
    try:
        dt = datetime.fromisoformat(entry["tsIso"].replace("Z", "+00:00"))
    except (AttributeError, KeyError, TypeError, ValueError):
        return None
    try:
        return max(0, int(now - dt.timestamp()))
    except (OSError, OverflowError, ValueError):   # a bogus year makes timestamp() blow up
        return None


def _chat_run(run_id: str) -> dict | None:
    """A conversation dressed as a run with ONE agent: the chat itself.

    Deliberately the same shape as any other run, so the whole client works on it
    unchanged -- list -> detail -> feed -- instead of needing a second screen.
    """
    path = find_conversation(run_id)
    if not path:
        return None
    try:
        st = path.stat()
    except OSError:
        return None
    now = time.time()
    age = int(now - st.st_mtime)
    meta = prompts.conversation_meta(path, st.st_size)
    return {"run": run_id, "workflow": meta["titulo"] or "(chat sin titulo)",
            "agentes": [{
                "id": "chat", "kb": st.st_size // 1024, "edadSeg": age,
                "estado": "ACTIVO" if age < AGENT_ACTIVE_SECS else "INACTIVO",
                "uptimeSeg": None,   # a chat has no spawn instant: it is not a run
                "mision": meta["ultimoPrompt"] or meta["titulo"] or "?",
                "promptChars": 0,
                # Only for a chat that is still moving: agent_activity reads the tail,
                # and doing that for every idle conversation on every tick is waste.
                "actividad": fsread.agent_activity(path) if age < AGENT_IDLE_SECS else "",
            }]}


def _codex_run(run_id: str) -> dict | None:
    """A Codex session dressed as a run with ONE agent, same as a chat: the whole client
    then works on it unchanged. `tokens` and `ventana` ride along so the detail view can
    show consumption and how much of the context window is spent."""
    path = find_codex(run_id)
    if not path:
        return None
    try:
        st = path.stat()
    except OSError:
        return None
    now = time.time()
    age = int(now - st.st_mtime)
    e = prompts.codex_entry(path, st.st_size)
    tok = e["tokens"] or {}
    estado = _codex_estado(e["ciclo"], age)
    return {"run": run_id,
            "workflow": (e["meta"] or {}).get("cwd") or "(sesion de Codex)",
            "agentes": [{
                "id": "codex", "kb": st.st_size // 1024, "edadSeg": age,
                "estado": estado, "uptimeSeg": None,
                "mision": e["ultimoMensaje"] or "?",
                "promptChars": 0,
                "actividad": e["ultimaTool"] if estado == "ACTIVO" else "",
                # Solo Codex los reporta: Claude Code no escribe consumo en sus transcripts.
                # 'tokens*' es el ACUMULADO de la sesion; 'contextoUso' es lo que ocupa la
                # ventana AHORA, que sale del ultimo turno y no de la suma.
                "tokens": tok.get("total_tokens"),
                "tokensEntrada": tok.get("input_tokens"),
                "tokensCache": tok.get("cached_input_tokens"),
                "tokensSalida": tok.get("output_tokens"),
                "tokensRazonamiento": tok.get("reasoning_output_tokens"),
                "contextoUso": (e["ultimoTurno"] or {}).get("input_tokens"),
                "ventana": e["ventana"],
            }]}


def api_run(run_id: str) -> dict | None:
    if (run_id or "").startswith("codex_"):
        return _codex_run(run_id)
    if (run_id or "").startswith("chat_"):
        return _chat_run(run_id)
    run_dir = find_run_dir(run_id)
    if not run_dir:
        return None
    is_loose_run = run_id.startswith("sueltos_")
    ji = fsread.JournalInfo({}, set(), set(), None) if is_loose_run else fsread.journal_info(run_dir)
    now = time.time()
    agents = []
    for a in fsread.list_agent_files(run_dir):
        age_secs = int(now - a["mtime"])
        key = ji.key_by_agent.get(a["id"])
        if a["id"] in ji.done_agents:
            status = "TERMINADO"
        elif age_secs < AGENT_ACTIVE_SECS:
            status = "ACTIVO"
        elif age_secs < AGENT_IDLE_SECS:
            status = "LENTO"
        elif key and key in ji.done_keys:
            status = "REEMPLAZADO"  # died, but its prompt was completed by another agent (resume)
        elif is_loose_run:
            status = "TERMINADO"  # no journal, no way to tell finished from dead: idle = done
        else:
            status = "MUERTO"
        # The mission comes from the already parsed (cached) prompt; agent_mission is the
        # fallback for the just-spawned agent that has not written its line 0 yet.
        e = prompts.prompt_entry(a["path"])
        agents.append({
            "id": a["id"], "kb": a["kb"], "edadSeg": age_secs, "estado": status,
            "uptimeSeg": _uptime_secs(e, now),   # desde el spawn; edadSeg es desde la ultima escritura
            "mision": prompts.mission_of(e) if e else fsread.agent_mission(a["path"]),
            "promptChars": e["chars"] if e else 0,
            "actividad": "" if status in ("TERMINADO", "REEMPLAZADO") else fsread.agent_activity(a["path"]),
        })
    return {"run": run_id if is_loose_run else run_dir.name,
            "workflow": "(agentes sueltos)" if is_loose_run else fsread.workflow_name(run_dir),
            "agentes": agents}


def _transcript_events(path) -> list[dict]:
    """Feed of a transcript, from its tail. Works for an agent AND for a conversation:
    both are the same JSONL event stream (assistant / tool_use / tool_result), which is
    why the chat view costs no parser of its own."""
    events = []
    for line in fsread.tail_lines(path):
        ts = fsread.local_hhmmss(line)
        if '"type":"assistant"' in line:
            tools = fsread.RE_TOOL.findall(line)
            if tools:
                hint = ""
                h = fsread.RE_HINT.search(line)
                if h:
                    hint = fsread.unescape(h.group(1))[:110]
                events.append({"ts": ts, "tipo": "TOOL", "txt": f"{', '.join(tools)}  {hint}".strip()})
            else:
                m = fsread.RE_TEXT.search(line)
                if m:
                    events.append({"ts": ts, "tipo": "DICE", "txt": fsread.unescape(m.group(1))[:180]})
        elif '"type":"user"' in line and '"type":"tool_result"' in line:
            # preview of what the tool returned (content string or [{text:...}])
            preview = ""
            m = re.search(r'"tool_result".{0,120}?"(?:content|text)":"((?:[^"\\]|\\.){1,110})', line)
            if not m:
                m = re.search(r'"text":"((?:[^"\\]|\\.){1,110})', line)
            if m:
                preview = "  · " + fsread.unescape(m.group(1)).strip()
            events.append({"ts": ts, "tipo": "RES", "txt": f"({max(1, len(line) // 1024)} KB){preview}"})
    return events


def api_agent(run_id: str, agent_id: str, n: int = 120) -> dict | None:
    if (run_id or "").startswith("codex_"):
        path = find_codex(run_id)
        if not path or agent_id != "codex":
            return None
        try:
            size = path.stat().st_size
        except OSError:
            return None
        e = prompts.codex_entry(path, size)
        return {"agente": fsread.codex_session_id(path)[:8],
                "mision": e["ultimoMensaje"] or "?",
                "eventos": fsread.codex_events(path, n)}
    if (run_id or "").startswith("chat_"):
        path = find_conversation(run_id)
        if not path or agent_id != "chat":   # a chat has exactly one pseudo-agent
            return None
        try:
            size = path.stat().st_size
        except OSError:
            return None
        meta = prompts.conversation_meta(path, size)
        return {"agente": path.stem[:8],
                "mision": meta["ultimoPrompt"] or meta["titulo"] or "?",
                "eventos": _transcript_events(path)[-n:]}
    run_dir = find_run_dir(run_id)
    if not run_dir or not RE_VALID_AGENT_ID.match(agent_id or ""):
        return None
    matches = list(run_dir.glob(f"agent-{agent_id}*.jsonl"))
    if not matches:
        return None
    path = matches[0]
    e = prompts.prompt_entry(path)
    return {"agente": path.stem,
            "mision": prompts.mission_of(e) if e else fsread.agent_mission(path),
            "eventos": _transcript_events(path)[-n:]}


def api_prompts(run_id: str) -> dict | None:
    """The run's prompts grouped by family (metadata + preview; full text goes separately)."""
    run_dir = find_run_dir(run_id)
    if not run_dir:
        return None
    is_loose_run = run_id.startswith("sueltos_")
    entries, total = prompts.run_prompts(run_dir)
    fams = [{**f, "moldeChars": f["pre"]} for f in prompts.prompt_families(run_dir, entries)]
    # sinPrompt: agents that exist but whose line 0 is not readable yet (the just-spawned
    # one). Without this datum they vanished from the grouped view with no warning at all.
    return {"run": run_id if is_loose_run else run_dir.name,
            "workflow": "(agentes sueltos)" if is_loose_run else fsread.workflow_name(run_dir),
            "agentes": len(entries), "sinPrompt": total - len(entries), "familias": fams}


# Fields of /api/prompt that come straight from prompt_entry, with their value for the
# empty fallback. ONE single list: the client interpolates every field and painted
# "undefined" when the fallback and the full response (previously enumerated by hand,
# separately) diverged. Keys stay Spanish: they are the API contract.
_PROMPT_FIELDS = {"texto": "", "chars": 0, "kb": 0, "ts": "", "cwd": "", "gitBranch": "",
                  "slug": "", "version": "", "agentType": "", "description": "",
                  "spawnDepth": None, "truncado": False}


def api_prompt(run_id: str, agent_id: str) -> dict | None:
    """FULL prompt of one agent. Untruncated: the 378 KB one is precisely an interesting one."""
    if (run_id or "").startswith("codex_"):
        path = find_codex(run_id)
        if not path or agent_id != "codex":
            return None
        try:
            st = path.stat()
        except OSError:
            return None
        texto = fsread.codex_first_prompt(path)
        meta = prompts.codex_entry(path, st.st_size)["meta"] or {}
        return {"run": run_id, "agente": "codex", **_PROMPT_FIELDS,
                "titulo": prompts.title_of(texto) or "(sesion de Codex)",
                "texto": texto, "chars": len(texto), "kb": st.st_size // 1024,
                "cwd": meta.get("cwd") or "",
                "version": str(meta.get("cli_version") or ""),
                "agentType": "codex",
                "description": str(meta.get("model_provider") or ""),
                "pre": 0, "suf": 0, "familia": 1,
                "aviso": "" if texto else "no pude leer el primer mensaje de esta sesion"}
    if (run_id or "").startswith("chat_"):
        # A chat has no launch prompt, but it does have the message that OPENED it, and
        # that is the closest analogue: what this conversation was started to do.
        path = find_conversation(run_id)
        if not path or agent_id != "chat":
            return None
        try:
            st = path.stat()
        except OSError:
            return None
        texto = fsread.conversation_first_prompt(path)
        meta = prompts.conversation_meta(path, st.st_size)
        return {"run": run_id, "agente": "chat", **_PROMPT_FIELDS,
                "titulo": meta["titulo"] or prompts.title_of(texto),
                "texto": texto, "chars": len(texto), "kb": st.st_size // 1024,
                "description": "mensaje que abrio la conversacion",
                "cwd": "", "pre": 0, "suf": 0, "familia": 1,
                "aviso": "" if texto else "no pude leer el primer mensaje de este chat"}
    run_dir = find_run_dir(run_id)
    if not run_dir or not RE_VALID_AGENT_ID.match(agent_id or ""):
        return None
    path = run_dir / f"agent-{agent_id}.jsonl"
    try:
        if not path.is_file():
            return None
    except OSError:
        return None
    e = prompts.prompt_entry(path)
    if not e:
        return {"run": run_id, "agente": agent_id, "titulo": "", **_PROMPT_FIELDS,
                "pre": 0, "suf": 0, "familia": 1,
                "aviso": "el agente todavia no escribio su prompt (linea 0 vacia, ilegible o "
                         f"mas grande que {fsread.LINE0_MAX_TOTAL_BYTES // (1024 * 1024)} MB)"}
    pre = suf = 0
    family_size = 1
    for f in prompts.prompt_families(run_dir, prompts.run_prompts(run_dir)[0]):
        if any(a["id"] == e["id"] for a in f["agentes"]):
            pre, suf, family_size = f["pre"], f["suf"], f["n"]
            break
    return {"run": run_id, "agente": e["id"], "titulo": prompts.title_of(e["texto"]),
            **{k: e[k] for k in _PROMPT_FIELDS},
            "pre": pre, "suf": suf, "familia": family_size}


def api_search(q: str, proyecto: str = "", tipo: str = "", limit: int = SEARCH_MAX_AGENTS) -> dict:
    """Text search over ALL the prompts. No inverted index: they are ~24 MB (the L0 of
    2500 files, 4% of the corpus) and the full sweep takes less than building the index."""
    q = (q or "").strip()
    if len(q) < 2:
        return {"error": "escribi al menos 2 caracteres"}
    t0 = time.time()
    pattern = prompts.search_pattern(q)
    hits, match_count, indexed_count = [], 0, 0
    projects, types = set(), set()
    for path in fsread._iter_agent_paths():
        e = prompts.prompt_entry(path)
        if not e:
            continue
        indexed_count += 1
        u = prompts._location_of(path)
        projects.add(u["proyecto"])
        if e["agentType"]:
            types.add(e["agentType"])
        if (proyecto and u["proyecto"] != proyecto) or (tipo and e["agentType"] != tipo):
            continue
        n, snippets = prompts._match_snippets(e["texto"], pattern)
        if not n:
            continue
        match_count += n
        hits.append({"run": u["run"], "workflow": u["workflow"], "proyecto": u["proyecto"],
                     "sesion": u["sesion"], "agente": e["id"], "ts": e["ts"], "tsIso": e["tsIso"],
                     "chars": e["chars"], "agentType": e["agentType"], "description": e["description"],
                     "titulo": prompts.title_of(e["texto"], 90), "n": n, "ctx": snippets})
    hits.sort(key=lambda h: h["tsIso"], reverse=True)
    return {"q": q, "agentes": len(hits), "ocurrencias": match_count, "indexados": indexed_count,
            "ms": int((time.time() - t0) * 1000), "truncado": len(hits) > limit,
            "proyectos": sorted(projects), "tipos": sorted(types), "hits": hits[:limit]}


def api_script(run_id: str) -> dict | None:
    """The workflow's .js: the literal template each family's prompts come out of."""
    run_dir = find_run_dir(run_id)
    if not run_dir or run_id.startswith("sueltos_"):
        return None
    s, text = fsread._run_script(run_dir)
    if not s or text is None:
        return None
    return {"run": run_dir.name, "nombre": s.name, "texto": text, "chars": len(text)}


def api_plan(run_id: str) -> dict | None:
    """Declared plan + REAL progress, kept separate on purpose.

    One is not crossed with the other: disk keeps no record of which phase each agent
    belongs to (the journal only carries agentId/key/result and the .meta.json
    agentType/spawnDepth), and with pipeline() the agents of different phases overlap.
    Inferring it would display an invented progression, so the plan is shown as declared
    and the progress separately.
    """
    if (run_id or "").startswith("chat_"):
        if not find_conversation(run_id):
            return None
        return {"run": run_id, "agentes": 1, "terminados": 0,
                "plan": None, "script": "", "motivo": "chat"}
    if (run_id or "").startswith("codex_"):
        if not find_codex(run_id):
            return None
        return {"run": run_id, "agentes": 1, "terminados": 0,
                "plan": None, "script": "", "motivo": "codex"}
    run_dir = find_run_dir(run_id)
    if not run_dir:
        return None
    agent_files = fsread.list_agent_files(run_dir)
    base = {"run": run_id, "agentes": len(agent_files), "terminados": 0,
            "plan": None, "script": "", "motivo": ""}
    if run_id.startswith("sueltos_"):
        base["motivo"] = "sueltos"          # agents outside a workflow: no plan to show
        return base
    base["terminados"] = len(fsread.journal_info(run_dir).done_agents)
    s, text = fsread._run_script(run_dir)
    if not s:
        base["motivo"] = "sin-script"       # 16 of 175 corpus runs: ran with no script on disk
        return base
    base["script"] = s.name
    if text is None:
        base["motivo"] = "ilegible"         # typically MAX_PATH
        return base
    plan = fsread.parse_plan(text)
    if not plan:
        base["motivo"] = "sin-meta"
        return base
    base["plan"] = plan
    return base


# Path -> (function, query params, message when it returns None). All these endpoints
# share the None -> 404 convention: declaring them here avoids repeating the idiom in
# do_GET. Defined LAST in the module: it references the api_* functions by name.
ROUTES_404 = {
    "/api/run":     (api_run,     ("id",),       "run no encontrado"),
    "/api/agent":   (api_agent,   ("run", "id"), "agente no encontrado"),
    "/api/prompts": (api_prompts, ("run",),      "run no encontrado"),
    "/api/prompt":  (api_prompt,  ("run", "id"), "prompt no encontrado"),
    "/api/script":  (api_script,  ("run",),      "este run no tiene script"),
    "/api/plan":    (api_plan,    ("run",),      "run no encontrado"),
}

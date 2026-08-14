"""Prompt cache and everything derived from an agent's line-0 text.

The full prompt of an agent is LINE 0 of its agent-<id>.jsonl (a "user" event).
/api/prompts, /api/prompt and /api/search all lean on it. This module is the sole
OWNER of all mutable state in the app: every cache lives inside the single
MonitorCache instance below, and every mutator stays in this module — api/server
must call functions, never rebind or copy the caches.

Language convention (whole app): code is English; JSON keys/values of the HTTP
contract, run-id prefix 'sueltos_', the '(agentes sueltos)' label and all
user-facing strings stay in SPANISH byte-for-byte — see server.py.
"""
from __future__ import annotations

import json
import os.path
import re
import threading

import fsread

MISSION_SCAN_CHARS = 64 * 1024   # window where the declared role ("SOS EL X") is searched
CTX_CHARS = 120                  # context on each side of a match in the global search
FAMILY_KEY_CHARS = 120           # prefix that defines the "family": same role within the run
TEMPLATE_PREVIEW_CHARS = 3000    # how much family template travels to the client (and is cached)

# Prompt-cache cap in CHARACTERS, not entries: what Python retains is ~2 bytes per
# character, so counting entries underestimated memory 2x (24 MB of text = 55 MB real).
# CAREFUL about lowering it: when the cap trips, /api/search goes to a 100% miss rate (a
# sequential sweep reuses nothing) and jumps from ~280 ms to ~1000 ms. Measured with the
# corpus at 24.0 M chars: with the cap at 110% of the corpus the search costs 285 ms; at
# 95%, 1000+ ms. That is why it stays roomy.
CACHE_MAX_CHARS = 96 * 1024 * 1024


# The prompt is written whole at spawn time and never rewritten -> it is immutable. That
# is why the cache is NOT keyed by (mtime, size) of the whole file: those change with every
# append and gave a 0% hit rate exactly on the ACTIVE runs, the ones the monitor exists to
# watch. The key is the file's IDENTITY (dev, inode) + "did not shrink" (a rewritten file
# does need a re-read).
class MonitorCache:
    """All in-process caches behind ONE lock (the server is a ThreadingHTTPServer).

    Prompts — key: str(path); value: (ident, size_at_read, entry).
      ident is (st_dev, st_ino). Line 0 of an agent file is immutable, so a hit is
      valid while the file is the SAME inode and has not shrunk (a rewritten file
      must be re-read). The budget is measured in CHARACTERS of cached prompt text
      (max_chars), not entries — see the CACHE_MAX_CHARS comment for why.
      Eviction: oldest-inserted first (dict order); put() re-inserts, so a rewrite
      refreshes recency. get() deliberately does NOT reorder (same as today).
    Families — key: str(run_dir); value: (signature, families). signature is the
      tuple of agent ids of the run: any spawn/removal invalidates the hit.
    Locations — key: str(dir); value: location dict. Never evicted (tiny; cached
      even when workflow name is empty, to avoid re-globbing the scripts dir).

    Post-hit mutations on a shared entry (entry['kb'] in prompt_entry and the
    entry['mision'] memoization in mission_of) happen OUTSIDE the lock on purpose:
    dict item assignment is atomic under the GIL and does not touch the char counter.
    """

    def __init__(self, max_chars: int = CACHE_MAX_CHARS) -> None:
        self._lock = threading.Lock()
        self._max_chars = max_chars
        self._prompts: dict[str, tuple[tuple[int, int], int, dict]] = {}
        self._prompt_chars = 0
        self._families: dict[str, tuple[tuple, list[dict]]] = {}
        self._locations: dict[str, dict] = {}
        self._convs: dict[str, tuple[int, dict]] = {}

    # ---------- conversations ----------
    def get_conv(self, key: str) -> tuple[int, dict] | None:
        with self._lock:
            return self._convs.get(key)

    def put_conv(self, key: str, size: int, meta: dict) -> None:
        with self._lock:
            self._convs[key] = (size, meta)

    # ---------- prompts ----------
    def get_prompt(self, key: str, ident: tuple[int, int], size: int) -> dict | None:
        """Cached entry if the file is the same inode and has not shrunk, else None."""
        with self._lock:
            hit = self._prompts.get(key)
            if hit and hit[0] == ident and size >= hit[1]:
                return hit[2]
            return None

    def put_prompt(self, key: str, ident: tuple[int, int], size: int, entry: dict) -> None:
        with self._lock:
            self._drop_locked(key)                # re-insert -> moves key to the end
            self._prompts[key] = (ident, size, entry)
            self._prompt_chars += entry["chars"]
            # Evict oldest-first instead of clearing: a full clear() was a cliff — the
            # next /api/search cost roughly double while rebuilding the whole cache.
            while self._prompt_chars > self._max_chars and len(self._prompts) > 1:
                self._drop_locked(next(iter(self._prompts)))

    def drop_prompt(self, key: str) -> None:
        with self._lock:
            self._drop_locked(key)

    def _drop_locked(self, key: str) -> None:
        old = self._prompts.pop(key, None)
        if old:
            self._prompt_chars -= old[2]["chars"]
        if not self._prompts:
            self._prompt_chars = 0                # no entries -> nothing to count: kills drift

    # ---------- families ----------
    def get_families(self, run_key: str, signature: tuple) -> list[dict] | None:
        with self._lock:
            hit = self._families.get(run_key)
            return hit[1] if hit and hit[0] == signature else None

    def put_families(self, run_key: str, signature: tuple, families: list[dict]) -> None:
        with self._lock:
            self._families[run_key] = (signature, families)

    # ---------- locations ----------
    def get_location(self, dir_key: str) -> dict | None:
        with self._lock:
            return self._locations.get(dir_key)

    def put_location(self, dir_key: str, location: dict) -> None:
        with self._lock:
            self._locations[dir_key] = location


CACHE = MonitorCache()


def prompt_entry(path) -> dict | None:
    """Prompt + metadata of an agent, or None if it is not readable yet.

    Reads ONE line: json.loads over the whole line. Neither head_lines (cuts at 64 KB
    and splits UTF-8 characters) nor unescape (eats the \\n of Windows paths) works here.
    """
    cache_key = str(path)
    try:
        st = path.stat()
    except OSError:
        CACHE.drop_prompt(cache_key)
        return None
    ident = (st.st_dev, st.st_ino)
    # L0 is immutable: the transcript growing does not change it. Re-read only if the
    # file is another one (different inode) or if it shrank (rewrite).
    entry = CACHE.get_prompt(cache_key, ident, st.st_size)
    if entry:
        entry["kb"] = st.st_size // 1024   # only field that depends on the current size
        return entry
    raw, truncated = fsread._read_line0(path)
    if raw is None:
        return None
    if not raw.strip():
        return None  # window between the create and the write of line 0: do not cache
    try:
        ev = json.loads(raw.decode("utf-8", "replace"))
    except ValueError:
        return None
    if not isinstance(ev, dict) or ev.get("type") != "user":
        return None
    msg = ev.get("message")
    text = fsread._content_text(msg.get("content") if isinstance(msg, dict) else None)
    meta = fsread._agent_meta(path)
    entry = {
        "id": path.stem.removeprefix("agent-"),
        "texto": text,
        "chars": len(text),
        "kb": st.st_size // 1024,
        # Used as sort key in run_prompts and api_search: if it is not a string, those
        # sorts mix types and raise TypeError -> 500. A non-string timestamp is not
        # something Claude Code emits today, but the file is written by another process
        # and the cost of hardening it is zero.
        "tsIso": fsread._meta_str(ev.get("timestamp")),
        "ts": fsread._local_ts_label(ev.get("timestamp") or ""),
        "cwd": ev.get("cwd") or "",
        "gitBranch": ev.get("gitBranch") or "",
        "slug": ev.get("slug") or "",
        "version": ev.get("version") or "",
        "agentType": fsread._meta_str(meta.get("agentType")),
        "description": fsread._meta_str(meta.get("description")),
        "spawnDepth": meta.get("spawnDepth") if isinstance(meta.get("spawnDepth"), int) else None,
        "truncado": truncated,
    }
    CACHE.put_prompt(cache_key, ident, st.st_size, entry)
    return entry


def title_of(text: str, n: int = 110) -> str:
    """First line with content. Comes from the ALREADY parsed text, not a regex over bytes."""
    for line in text.split("\n", 40)[:40]:
        s = fsread._strip_boilerplate(line.strip().strip("#*=-").strip())
        if len(s) > 2:
            return s[:n]
    return text.strip()[:n]


def _compute_mission(entry: dict) -> str:
    if entry["description"]:
        return entry["description"]
    head = entry["texto"][:MISSION_SCAN_CHARS]
    m = fsread.RE_MISSION.search(head)
    if m:
        return m.group(1).strip().title()
    m = fsread.RE_MISSION_LOOSE.search(head)
    if m:
        return m.group(1).strip()
    return title_of(entry["texto"]) or "?"


def mission_of(entry: dict) -> str:
    """Short label for the Mission column: what tells THIS agent apart from the rest of
    the fan-out.

    The declared role ("SOS EL VERIFICADOR") beats the title: in runs where the prompt
    starts with tens of KB of shared context it is the only thing that separates the
    agents. Searched only in the prompt's head (some are 380 KB, and the role is
    declared at the top).

    Memoized on the cached entry: the prompt is immutable and this runs for EVERY agent
    on every 4 s tick (sweeping 64 KB per agent with two regexes cost ~6 ms per tick
    of 103).
    """
    m = entry.get("mision")
    if m is None:
        m = entry["mision"] = _compute_mission(entry)
    return m


def run_prompts(run_dir) -> tuple[list[dict], int]:
    """(readable prompts sorted by launch time, how many agents the run has).

    The file's mtime is NO good for sorting: it reflects the last write, not the spawn.
    The total goes separately so the agents whose line 0 is not readable yet can be
    reported: otherwise the just-spawned agent vanishes from the grouped view with no
    warning at all.
    """
    files = fsread.list_agent_files(run_dir)
    entries = [e for e in (prompt_entry(a["path"]) for a in files) if e]
    entries.sort(key=lambda e: e["tsIso"])
    return entries, len(files)


def prompt_families(run_dir, entries: list[dict]) -> list[dict]:
    """Groups by prefix: a fan-out of 103 agents is usually one template with 5 variants.

    Grouping by exact hash does not work (only 9% of the prompts repeat) and the
    journal's 'key' is not the prompt's hash either (verified: 0 matches).
    """
    signature = tuple(e["id"] for e in entries)
    hit = CACHE.get_families(str(run_dir), signature)
    if hit is not None:
        return hit
    groups = {}
    for e in entries:
        group_key = re.sub(r"\s+", " ", e["texto"][:FAMILY_KEY_CHARS]).strip()
        groups.setdefault(group_key, []).append(e)
    families = []
    for group_key, members in groups.items():
        texts = [m["texto"] for m in members]
        shortest = min(len(t) for t in texts)
        pre = suf = 0
        if len(texts) > 1:
            pre = len(os.path.commonprefix(texts))
            if pre < shortest:  # if they are identical there is nothing variable to isolate
                suf = len(os.path.commonprefix([t[::-1] for t in texts]))
                if pre + suf > shortest:
                    suf = shortest - pre
            else:
                pre = shortest
        template = texts[0][:pre]
        families.append({
            "clave": group_key[:80],
            "titulo": title_of(template or texts[0]),
            "n": len(members),
            "pre": pre,
            "suf": suf,
            # Trimmed BEFORE caching: the response shows at most this much, and storing
            # the whole template retained hundreds of KB per run in the cache, forever.
            "molde": template[:TEMPLATE_PREVIEW_CHARS],
            "moldeSuf": (texts[0][len(texts[0]) - suf:] if suf else "")[:TEMPLATE_PREVIEW_CHARS],
            "identicos": len(set(texts)) == 1,
            "agentes": [{
                "id": m["id"], "chars": m["chars"], "kb": m["kb"], "ts": m["ts"],
                "agentType": m["agentType"], "description": m["description"],
                "varChars": max(0, m["chars"] - pre - suf),
                "variable": m["texto"][pre:len(m["texto"]) - suf if suf else None][:240],
            } for m in members],
        })
    families.sort(key=lambda f: (-f["n"], f["titulo"]))
    CACHE.put_families(str(run_dir), signature, families)
    return families


# How much a conversation has to GROW before its title is read again. The title changes
# rarely and the last prompt only once per turn, while /api/runs sweeps every conversation
# every 4 s -- re-reading a 128 KB tail per chat per tick, forever, to almost always get
# the same answer. Trading a slightly stale title for that is the right side of the deal:
# the row's state and age come from stat(), so freshness where it matters is unaffected.
CONV_REFRESH_BYTES = 256 * 1024


def conversation_meta(path, size: int) -> dict:
    """Cached title + last prompt of a conversation. `size` is the caller's stat: the
    tick already has it, and re-stat-ing here would double the syscalls of the sweep."""
    key = str(path)
    hit = CACHE.get_conv(key)
    if hit is not None and abs(size - hit[0]) < CONV_REFRESH_BYTES:
        return hit[1]
    meta = fsread.conversation_meta(path)
    CACHE.put_conv(key, size, meta)
    return meta


def _location_of(path) -> dict:
    """Run / project / session out of the agent's path (the two shapes that exist)."""
    d = path.parent
    dir_key = str(d)
    u = CACHE.get_location(dir_key)
    if u is None:
        if d.name == "subagents":  # loose agent: <project>/<session>/subagents/agent-*.jsonl
            sess = d.parent
            u = {"run": "sueltos_" + sess.name, "workflow": "(agentes sueltos)"}
        else:                      # .../subagents/workflows/<wf_run>/agent-*.jsonl
            sess = d.parents[2]
            u = {"run": d.name, "workflow": fsread.workflow_name(d)}
        u["proyecto"] = fsread.clean_project_slug(sess.parent.name)
        u["sesion"] = sess.name[:8]
        # Cached even when the name comes out empty (16 of 174 runs have no .js):
        # otherwise every search re-globs the scripts dir for all their agents.
        CACHE.put_location(dir_key, u)
    return u


def search_pattern(q: str):
    """Case-insensitive regex applied OVER THE ORIGINAL TEXT.

    The search used to run on texto.lower() and cut the original with those offsets:
    that assumes .lower() never changes the length, and there is one character in all
    of Unicode that does (U+0130 'İ' lowers to 2 code points), so the <mark> ended up
    shifted and highlighted another word. As a bonus this avoids copying the corpus's
    24 MB in lowercase on every search.
    The [Aa] class is built by hand instead of re.IGNORECASE because it measures ~40%
    faster.
    """
    parts = []
    for c in q:
        lower, upper = c.lower(), c.upper()
        if lower != upper and len(lower) == 1 and len(upper) == 1:
            parts.append("[" + re.escape(lower) + re.escape(upper) + "]")
        else:
            parts.append(re.escape(c))  # 'ß'.upper() is 2 chars: that one is searched literally
    return re.compile("".join(parts))


def _match_snippets(text: str, pattern, max_snippets: int = 3, max_matches: int = 60) -> tuple[int, list[dict]]:
    """Match count + snippets ±CTX_CHARS. Returns pre/hit/post separately: the client
    escapes each part and highlights the middle without recomputing offsets over HTML."""
    total, snippets = 0, []
    for m in pattern.finditer(text):
        total += 1
        if len(snippets) < max_snippets:
            i, j = m.span()
            ini, fin = max(0, i - CTX_CHARS), min(len(text), j + CTX_CHARS)
            snippets.append({
                "pre": ("…" if ini else "") + text[ini:i].replace("\n", " "),
                "hit": text[i:j].replace("\n", " "),
                "post": text[j:fin].replace("\n", " ") + ("…" if fin < len(text) else ""),
            })
        if total >= max_matches:
            break
    return total, snippets

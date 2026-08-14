"""Tests for the pure parsers behind the monitor's endpoints.

They do not start the server: they exercise the functions that translate the
on-disk state (journal.jsonl, agent-*.jsonl, workflow scripts) into what the
endpoints serve. The odd cases come from real regressions documented in the
code comments. Asserted JSON keys and state values are the Spanish API
contract and stay verbatim.
"""
import json
import time

import api
import fsread
import prompts


def write_agent(directory, agent_id, prompt="Sos el agente de prueba."):
    """Minimal agent-<id>.jsonl: line 0 = user event carrying the prompt."""
    line0 = {"type": "user", "timestamp": "2026-07-31T12:00:00Z", "cwd": "c:/x",
             "gitBranch": "main", "slug": "prueba", "version": "1.0",
             "message": {"content": prompt}}
    f = directory / f"agent-{agent_id}.jsonl"
    f.write_text(json.dumps(line0) + "\n", encoding="utf-8")
    return f


# ---------------------------------------------------------------- pure helpers

def test_clean_project_slug():
    # Claude Code slugs on the three OSes + a name that is not a slug.
    assert fsread.clean_project_slug("c--Net-8-Tools") == "Tools"
    assert fsread.clean_project_slug("C--MiApp") == "MiApp"
    assert fsread.clean_project_slug("-home-diego-repos-miapp") == "repos-miapp"
    assert fsread.clean_project_slug("-Users-diego-dev-app") == "dev-app"
    assert fsread.clean_project_slug("proyecto-suelto") == "proyecto-suelto"


def test_unescape():
    assert fsread.unescape("hola\\nmundo") == "hola mundo"
    assert fsread.unescape('dijo \\"hola\\"') == 'dijo "hola"'
    assert fsread.unescape("c:\\\\ruta") == "c:\\ruta"


def test_strip_boilerplate():
    f = fsread._strip_boilerplate
    # Context sentence with a path (Windows and POSIX): skipped.
    assert f("Repo: c:\\Net 8\\App (rama main). Sos el auditor.") == "Sos el auditor."
    assert f("Repo: /home/diego/app (rama main). Sos el auditor.") == "Sos el auditor."
    # Two context sentences in a row.
    assert f("Repo: c:\\X. Workdir: c:\\Y. Sos el auditor.") == "Sos el auditor."
    # Label that only the old inline copy in agent_mission knew: now unified.
    assert f("Contexto: c:\\Net 8\\App. Sos el auditor.") == "Sos el auditor."
    # A normal sentence is left alone.
    text = "Analiza el modulo de pagos. Reporta bugs."
    assert f(text) == text
    # 100% boilerplate line -> "" (the caller moves on to the next line).
    assert f("Workdir: c:\\Net 8\\App") == ""


def test_title_of():
    # Decoration or path-context lines are skipped; the first line with content wins.
    assert prompts.title_of("###\nSos el auditor de pagos.\nresto") == "Sos el auditor de pagos."
    assert prompts.title_of("Repo: c:\\X (rama main).\nSos el auditor.\n") == "Sos el auditor."
    # Pin of the unified label set: 'Contexto' (only the old agent_mission copy knew it)
    # now strips here too, even without a path in the sentence.
    assert prompts.title_of("Contexto: preparando el entorno.\nSos el auditor.\n") == "Sos el auditor."
    assert len(prompts.title_of("x" * 500)) <= 110


def test_search_pattern():
    # Case-insensitive without re.IGNORECASE (hand-built [Aa] class).
    p = prompts.search_pattern("hola")
    assert p.search("decir HOLA fuerte")
    assert p.search("Hola")
    assert not p.search("helado")
    # The query is escaped: parentheses and dots are literal.
    assert prompts.search_pattern("f(x)").search("calcular F(X) aca")
    assert not prompts.search_pattern("a.b").search("aXb")
    # U+0130 ('I' with dot): its .lower() is 2 code points -> goes literal, no crash.
    assert prompts.search_pattern("\u0130stanbul").search("\u0130stanbul")


# ---------------------------------------------------------------- journal and states

def test_journal_info(tmp_path):
    (tmp_path / "journal.jsonl").write_text(
        '{"type":"started","agentId":"aa11","key":"k1"}\n'
        '{"type":"result","agentId":"aa11","key":"k1"}\n'
        '{"type":"started","agentId":"bb22","key":"k2"}\n',
        encoding="utf-8")
    ji = fsread.journal_info(tmp_path)
    assert ji.key_by_agent == {"aa11": "k1", "bb22": "k2"}
    assert ji.done_agents == {"aa11"}
    assert ji.done_keys == {"k1"}
    assert ji.last_type == "started"


def test_journal_info_no_journal(tmp_path):
    # Tuple equality on purpose: freezes the NamedTuple's field order.
    assert fsread.journal_info(tmp_path) == ({}, set(), set(), None)


def test_run_status():
    now = time.time()
    old = now - 10 * api.RUN_ACTIVE_SECS
    assert api.run_status([{"mtime": now}], "started") == "ACTIVO"
    assert api.run_status([{"mtime": old}], "result") == "TERMINADO"
    assert api.run_status([{"mtime": old}], "started") == "ESTANCADO"
    assert api.run_status([], None) == "ESTANCADO"


def test_find_run_dir_validates_ids(tmp_path, monkeypatch):
    # Ids come from the client and go into a glob: anything invalid dies BEFORE touching disk.
    monkeypatch.setattr(fsread, "ROOT", tmp_path)
    assert api.find_run_dir("") is None
    assert api.find_run_dir("../../etc") is None
    assert api.find_run_dir("wf_../x") is None
    assert api.find_run_dir("wf_noexiste") is None


# ---------------------------------------------------------------- prompts

def test_prompt_entry_and_api_prompt_shape(tmp_path):
    e = prompts.prompt_entry(write_agent(tmp_path, "ab12cd"))
    assert e["id"] == "ab12cd"
    assert e["texto"] == "Sos el agente de prueba."
    assert e["chars"] == len(e["texto"])
    assert e["truncado"] is False
    assert e["gitBranch"] == "main"
    # The shape of /api/prompt comes from this entry via _PROMPT_FIELDS: if a field
    # disappears from prompt_entry, it blows up here and not as "undefined" client-side.
    assert set(api._PROMPT_FIELDS) <= set(e)


def test_prompt_entry_unreadable(tmp_path):
    f = tmp_path / "agent-ffff.jsonl"
    f.write_text("esto no es json\n", encoding="utf-8")
    assert prompts.prompt_entry(f) is None
    f2 = tmp_path / "agent-eeee.jsonl"
    f2.write_text("", encoding="utf-8")
    assert prompts.prompt_entry(f2) is None


def test_agent_mission_skips_boilerplate(tmp_path):
    # Line 0 is not a valid user event, but has a "content" match with boilerplate in
    # front: agent_mission (the raw fallback) skips it via the canonical stripper.
    f = tmp_path / "agent-aaaa.jsonl"
    f.write_text('{"content":"Repo: c:\\\\Net 8\\\\App (rama main). Analiza pagos."}\n',
                 encoding="utf-8")
    assert fsread.agent_mission(f) == "Analiza pagos."[:70]
    # Whole line is boilerplate -> "?" (better than showing the repo path as the mission).
    f2 = tmp_path / "agent-bbbb.jsonl"
    f2.write_text('{"content":"Workdir: c:\\\\Net 8\\\\App"}\n', encoding="utf-8")
    assert fsread.agent_mission(f2) == "?"


# ---------------------------------------------------------------- workflow plan

def test_parse_plan():
    src = """export const meta = {
  name: 'demo',
  description: "Un plan de prueba",   // comentario al final
  phases: [
    { title: 'Fase 1', detail: 'detalle', },
    { title: "Fase 2" },
  ],
}
phase('Fase 1')"""
    plan = fsread.parse_plan(src)
    assert plan["nombre"] == "demo"
    assert plan["descripcion"] == "Un plan de prueba"
    assert [f["titulo"] for f in plan["fases"]] == ["Fase 1", "Fase 2"]
    assert plan["fases"][0]["detalle"] == "detalle"


def test_parse_plan_no_meta():
    assert fsread.parse_plan("const x = 1") is None
    assert fsread.parse_plan("") is None
    assert fsread.parse_plan(None) is None


# ---------------------------------------------------------------- /api/runs end to end

def test_api_runs_integration(tmp_path, monkeypatch):
    monkeypatch.setattr(fsread, "ROOT", tmp_path)
    # A workflow run with its agent finished...
    wf = tmp_path / "c--Net-8-Demo" / "sesion1234abcd" / "subagents" / "workflows" / "wf_abc123"
    wf.mkdir(parents=True)
    (wf / "journal.jsonl").write_text(
        '{"type":"started","agentId":"aa11","key":"k1"}\n'
        '{"type":"result","agentId":"aa11","key":"k1"}\n',
        encoding="utf-8")
    write_agent(wf, "aa11")
    # ...and a loose agent in another session.
    loose = tmp_path / "c--Net-8-Demo" / "sesionffff9999" / "subagents"
    loose.mkdir(parents=True)
    write_agent(loose, "bb22")

    rows = api.api_runs()
    # Freezes the JSON contract of a /api/runs row.
    for r in rows:
        assert set(r) == {"run", "workflow", "proyecto", "sesion", "agentes",
                          "listos", "activos", "estado", "ultimaAct", "edadSeg",
                          "tipo", "kb", "tokens"}
    by_run = {r["run"]: r for r in rows}
    assert set(by_run) == {"wf_abc123", "sueltos_sesionffff9999"}

    wf_row = by_run["wf_abc123"]
    assert wf_row["proyecto"] == "Demo"           # slug c--Net-8-Demo cleaned
    assert wf_row["agentes"] == 1
    assert wf_row["listos"] == 1                  # the journal has its result
    # Freshly written, but it already has its result: done wins over recency, so the run
    # shows 0 running. listos + activos do NOT have to add up to agentes.
    assert wf_row["activos"] == 0
    assert wf_row["estado"] == "ACTIVO"           # just written to disk

    loose_row = by_run["sueltos_sesionffff9999"]
    assert loose_row["workflow"] == "(agentes sueltos)"
    assert loose_row["agentes"] == 1
    assert loose_row["estado"] == "ACTIVO"


def test_uptime_viene_de_la_linea_0_no_del_mtime(tmp_path, monkeypatch):
    """uptimeSeg mide desde el SPAWN; edadSeg, desde la ultima escritura.

    Es la distincion que la UI confundia: un agente 12 minutos adentro de su trabajo que
    acaba de escribir mostraba "hace 0s", que se lee como "recien arranco".
    """
    monkeypatch.setattr(fsread, "ROOT", tmp_path)
    wf = tmp_path / "c--Demo" / "sesion1234abcd" / "subagents" / "workflows" / "wf_up1"
    wf.mkdir(parents=True)
    (wf / "journal.jsonl").write_text('{"type":"started","agentId":"aa11","key":"k1"}\n',
                                      encoding="utf-8")
    # Linea 0 fechada 30 minutos atras, pero el archivo se acaba de escribir.
    spawn = time.time() - 1800
    line0 = {"type": "user",
             "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(spawn)),
             "message": {"content": "Sos el agente lento."}}
    (wf / "agent-aa11.jsonl").write_text(json.dumps(line0) + "\n", encoding="utf-8")

    d = api.api_run("wf_up1")
    assert d is not None
    a = d["agentes"][0]
    assert a["edadSeg"] < 60                    # escribio recien
    assert 1700 < a["uptimeSeg"] < 1900         # pero arranco hace media hora
    assert a["estado"] == "ACTIVO"


def test_uptime_es_none_sin_linea_0(tmp_path, monkeypatch):
    # Agente recien spawneado: el archivo existe pero la linea 0 todavia no esta.
    # Sin prompt no hay instante de spawn, y la UI tiene que poder omitir el dato.
    monkeypatch.setattr(fsread, "ROOT", tmp_path)
    wf = tmp_path / "c--Demo" / "sesion1234abcd" / "subagents" / "workflows" / "wf_up2"
    wf.mkdir(parents=True)
    (wf / "agent-bb22.jsonl").write_text("", encoding="utf-8")

    d = api.api_run("wf_up2")
    assert d is not None
    a = d["agentes"][0]
    assert a["uptimeSeg"] is None
    assert isinstance(a["edadSeg"], int)


def test_uptime_no_explota_con_timestamp_basura(tmp_path, monkeypatch):
    # El archivo lo escribe otro proceso: un timestamp no-ISO no puede tumbar el endpoint.
    monkeypatch.setattr(fsread, "ROOT", tmp_path)
    wf = tmp_path / "c--Demo" / "sesion1234abcd" / "subagents" / "workflows" / "wf_up3"
    wf.mkdir(parents=True)
    for i, ts in enumerate(["no-es-fecha", "", "0000-00-00T00:00:00Z"]):
        line0 = {"type": "user", "timestamp": ts, "message": {"content": "Sos un agente."}}
        (wf / f"agent-cc{i}.jsonl").write_text(json.dumps(line0) + "\n", encoding="utf-8")

    d = api.api_run("wf_up3")
    assert d is not None
    for a in d["agentes"]:
        assert a["uptimeSeg"] is None            # sin fecha usable, pero sin 500


def test_api_runs_missing_root(tmp_path, monkeypatch):
    # Freshly installed machine, no ~/.claude/projects: empty list, not a 500.
    monkeypatch.setattr(fsread, "ROOT", tmp_path / "no-existe")
    assert api.api_runs() == []


# ---------------------------------------------------------------- conversations

def write_chat(project_dir, session, titulo=None, ai_titulo=None, ultimo=None,
               primer="Hola, arranquemos con esto."):
    """Minimal <project>/<session>.jsonl. Mirrors what Claude Code appends: the opening
    user message plus the session records, each RE-APPENDED whenever it changes."""
    project_dir.mkdir(parents=True, exist_ok=True)
    lines = [{"type": "user", "timestamp": "2026-08-01T10:00:00Z",
              "message": {"content": primer}}]
    if ai_titulo:
        lines.append({"type": "ai-title", "aiTitle": ai_titulo, "sessionId": session})
    if titulo:
        lines.append({"type": "custom-title", "customTitle": titulo, "sessionId": session})
    if ultimo:
        lines.append({"type": "last-prompt", "lastPrompt": ultimo, "sessionId": session})
    f = project_dir / f"{session}.jsonl"
    f.write_text("".join(json.dumps(o) + "\n" for o in lines), encoding="utf-8")
    return f


def test_chat_aparece_en_runs_aunque_no_haya_subagentes(tmp_path, monkeypatch):
    # El caso que motivo la funcion: un proyecto donde SOLO se conversa. Antes no
    # producia ninguna fila y ni siquiera figuraba en el filtro de proyectos.
    monkeypatch.setattr(fsread, "ROOT", tmp_path)
    write_chat(tmp_path / "c--Demo", "aaaabbbbccccdddd",
               ai_titulo="Titulo puesto por la IA", ultimo="segui con el paso 3")

    rows = api.api_runs()
    assert len(rows) == 1
    r = rows[0]
    assert r["tipo"] == "chat"
    assert r["run"] == "chat_aaaabbbbccccdddd"
    assert r["proyecto"] == "Demo"
    assert r["workflow"] == "Titulo puesto por la IA"
    assert r["estado"] == "ACTIVO"           # recien escrito
    assert r["agentes"] == 0                 # un chat no tiene agentes que contar


def test_chat_prefiere_el_titulo_del_usuario(tmp_path, monkeypatch):
    monkeypatch.setattr(fsread, "ROOT", tmp_path)
    write_chat(tmp_path / "c--Demo", "1111222233334444",
               titulo="El que puse yo", ai_titulo="El que invento la IA")
    assert api.api_runs()[0]["workflow"] == "El que puse yo"


def test_chat_run_agente_y_prompt(tmp_path, monkeypatch):
    monkeypatch.setattr(fsread, "ROOT", tmp_path)
    write_chat(tmp_path / "c--Demo", "5555666677778888",
               titulo="Mi chat", ultimo="dale con eso", primer="Arranquemos por aca.")

    d = api.api_run("chat_5555666677778888")
    assert d is not None and len(d["agentes"]) == 1
    a = d["agentes"][0]
    assert a["id"] == "chat" and a["estado"] == "ACTIVO"
    assert a["mision"] == "dale con eso"     # el ultimo prompt, no el titulo

    p = api.api_prompt("chat_5555666677778888", "chat")
    assert p is not None and p["texto"] == "Arranquemos por aca."

    # El plan explica por que no hay plan, en vez de devolver 404 pelado.
    assert api.api_plan("chat_5555666677778888")["motivo"] == "chat"


def test_chat_id_invalido_no_toca_disco(tmp_path, monkeypatch):
    # Mismo criterio que find_run_dir: el id viene del cliente y termina en un glob.
    monkeypatch.setattr(fsread, "ROOT", tmp_path)
    write_chat(tmp_path / "c--Demo", "9999888877776666")
    for malo in ("chat_../../etc/passwd", "chat_*", "chat_", "chat_a/b", ""):
        assert api.find_conversation(malo) is None
        assert api.api_run(malo) is None


# ---------------------------------------------------------------- codex

def write_rollout(day_dir, session_id, *, cwd=r"C:\proy\demo", ciclo="complete",
                  tokens=1000, cache=800, ventana=250000, mensaje="Listo, verificado.",
                  primer="Revisa el modulo de pagos.", tool="exec"):
    """Rollout minimo con la forma real de Codex: session_meta en la linea 0 y despues
    event_msg / response_item, uno por linea."""
    day_dir.mkdir(parents=True, exist_ok=True)
    L = [
        {"timestamp": "2026-08-14T06:00:00.0Z", "type": "session_meta",
         "payload": {"session_id": session_id, "cwd": cwd, "cli_version": "0.147.0",
                     "model_provider": "openai"}},
        {"timestamp": "2026-08-14T06:00:01.0Z", "type": "event_msg",
         "payload": {"type": "task_started", "turn_id": "1"}},
        {"timestamp": "2026-08-14T06:00:02.0Z", "type": "event_msg",
         "payload": {"type": "user_message", "message": primer}},
        {"timestamp": "2026-08-14T06:00:03.0Z", "type": "response_item",
         "payload": {"type": "custom_tool_call", "name": tool, "input": "ls -la"}},
        {"timestamp": "2026-08-14T06:00:04.0Z", "type": "response_item",
         "payload": {"type": "custom_tool_call_output",
                     "output": [{"type": "input_text", "text": "ok"}]}},
        {"timestamp": "2026-08-14T06:00:05.0Z", "type": "event_msg",
         "payload": {"type": "agent_message", "message": mensaje}},
        {"timestamp": "2026-08-14T06:00:06.0Z", "type": "event_msg",
         "payload": {"type": "token_count", "info": {
             "total_token_usage": {"input_tokens": 900, "cached_input_tokens": cache,
                                   "output_tokens": 100, "reasoning_output_tokens": 20,
                                   "total_tokens": tokens},
             # el ULTIMO turno, no el acumulado: es lo que ocupa la ventana ahora
             "last_token_usage": {"input_tokens": 300, "cached_input_tokens": 250,
                                  "output_tokens": 40, "total_tokens": 340},
             "model_context_window": ventana}}},
    ]
    if ciclo == "complete":
        L.append({"timestamp": "2026-08-14T06:00:07.0Z", "type": "event_msg",
                  "payload": {"type": "task_complete", "turn_id": "1"}})
    f = day_dir / f"rollout-2026-08-14T06-00-00-{session_id}.jsonl"
    f.write_text("".join(json.dumps(o) + "\n" for o in L), encoding="utf-8")
    return f


def test_codex_aparece_en_runs_con_tokens(tmp_path, monkeypatch):
    monkeypatch.setattr(fsread, "ROOT", tmp_path / "claude")
    monkeypatch.setattr(fsread, "CODEX_ROOT", tmp_path / "codex")
    write_rollout(tmp_path / "codex" / "2026" / "08" / "14",
                  "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee", tokens=1300785)

    rows = api.api_runs()
    assert len(rows) == 1
    r = rows[0]
    assert r["tipo"] == "codex"
    assert r["run"] == "codex_aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
    assert r["estado"] == "TERMINADO"          # el ultimo evento de ciclo es task_complete
    assert r["tokens"] == 1300785              # el dato que Claude Code no escribe
    assert r["proyecto"] == "demo"             # del cwd de session_meta


def test_codex_estado_manda_el_ultimo_evento_no_la_frescura():
    # Igual que run_status con el journal: un turno abierto y quieto es ESTANCADO, no
    # TERMINADO; y sin ningun evento de ciclo en la cola se admite no saber.
    assert api._codex_estado("complete", 99999) == "TERMINADO"
    assert api._codex_estado("started", 5) == "ACTIVO"
    assert api._codex_estado("started", 99999) == "ESTANCADO"
    assert api._codex_estado(None, 5) == "?"


def test_codex_detalle_feed_y_prompt(tmp_path, monkeypatch):
    monkeypatch.setattr(fsread, "ROOT", tmp_path / "claude")
    monkeypatch.setattr(fsread, "CODEX_ROOT", tmp_path / "codex")
    sid = "11111111-2222-3333-4444-555555555555"
    write_rollout(tmp_path / "codex" / "2026" / "08" / "14", sid,
                  primer="Auditá el bridge.", mensaje="Encontré dos problemas.", tool="exec")
    rid = "codex_" + sid

    d = api.api_run(rid)
    assert d is not None and len(d["agentes"]) == 1
    a = d["agentes"][0]
    assert a["id"] == "codex" and a["estado"] == "TERMINADO"
    assert a["mision"] == "Encontré dos problemas."
    assert a["tokens"] == 1000 and a["tokensCache"] == 800 and a["ventana"] == 250000
    # La ocupacion de la ventana sale del ULTIMO turno (300), no del acumulado de entrada
    # (900). Confundirlos mostraba porcentajes de contexto por encima del 100%.
    assert a["contextoUso"] == 300

    f = api.api_agent(rid, "codex")
    assert f is not None
    tipos = [e["tipo"] for e in f["eventos"]]
    assert "TOOL" in tipos and "RES" in tipos and "DICE" in tipos

    p = api.api_prompt(rid, "codex")
    assert p is not None and p["texto"] == "Auditá el bridge."
    assert api.api_plan(rid)["motivo"] == "codex"


def test_codex_id_invalido_no_toca_disco(tmp_path, monkeypatch):
    monkeypatch.setattr(fsread, "CODEX_ROOT", tmp_path / "codex")
    write_rollout(tmp_path / "codex" / "2026" / "08" / "14",
                  "99999999-8888-7777-6666-555555555555")
    for malo in ("codex_../../etc/passwd", "codex_*", "codex_", "codex_a/b", ""):
        assert api.find_codex(malo) is None
        assert api.api_run(malo) is None


def test_codex_cache_no_relee_si_no_crecio(tmp_path, monkeypatch):
    # El cache es lo que hace viable barrer 277 MB cada 4 s: se keyea por TAMANO, porque
    # un rollout es append-only y uno terminado ya no crece nunca mas.
    monkeypatch.setattr(fsread, "CODEX_ROOT", tmp_path / "codex")
    f = write_rollout(tmp_path / "codex" / "2026" / "08" / "14",
                      "abcdabcd-0000-1111-2222-333333333333")
    llamadas = {"n": 0}
    real = fsread.codex_tail

    def contando(path):
        llamadas["n"] += 1
        return real(path)

    monkeypatch.setattr(fsread, "codex_tail", contando)
    size = f.stat().st_size
    prompts.codex_entry(f, size)
    prompts.codex_entry(f, size)
    prompts.codex_entry(f, size)
    assert llamadas["n"] == 1               # dos hits de cache
    prompts.codex_entry(f, size + 10)       # crecio -> hay que releer
    assert llamadas["n"] == 2


def test_codex_project_atribuye_al_padre_y_no_ensucia_el_filtro():
    p = fsread.codex_project
    # 1. cwd dentro del scratchpad de una sesion de Claude Code -> gana el proyecto padre.
    #    Es lo que hace visible que ese Codex lo lanzo esa sesion.
    assert p({"cwd": r"C:\Users\x\AppData\Local\Temp\claude\C--IA-Projects-whatsapp-mcp"
                     r"\c9a32750-1111\scratchpad\wt-a79326a"}) == "IA-Projects-whatsapp-mcp"
    # 2. si no, el repo de git que Codex anota.
    assert p({"cwd": r"C:\algo", "git": {"repository_url":
              "https://github.com/lharries/whatsapp-mcp.git"}}) == "whatsapp-mcp"
    # 3. el scratch propio de Codex no es un proyecto: una etiqueta, no un pedazo de prompt.
    assert p({"cwd": r"C:\Users\x\Documents\Codex\2026-07-05\let-s-set-up-a-scheduled"}) \
        == "(Codex sin proyecto)"
    # 4. un directorio de verdad se usa tal cual.
    assert p({"cwd": r"C:\Users\x\Documents\TunBridge"}) == "TunBridge"
    assert p({}) == "?"

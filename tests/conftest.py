"""Makes the workflow-monitor modules importable from the tests.

They live in workflow-monitor/ (hyphenated directory, not a package), so the
directory goes on sys.path and the modules are imported flat (fsread, prompts,
api, server) — the same layout server.py sees when run as a script. Tests that
touch ROOT monkeypatch it on fsread (the single owner): api reads it as
fsread.ROOT attribute, so one patch covers every consumer.
"""
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "workflow-monitor"))


@pytest.fixture(autouse=True)
def _codex_aislado(tmp_path, monkeypatch):
    """Ningun test puede leer el ~/.codex REAL de quien corre la suite.

    fsread tiene DOS raices —ROOT para Claude Code y CODEX_ROOT para Codex— y los tests
    que solo parcheaban la primera terminaban barriendo las sesiones de Codex de la
    maquina: /api/runs devolvia cientos de filas ajenas al fixture y las aserciones sobre
    cuantas filas hay pasaban a depender de quien corre los tests. Se apunta a un
    directorio que no existe; el que quiera probar Codex lo re-parchea.
    """
    import fsread
    monkeypatch.setattr(fsread, "CODEX_ROOT", tmp_path / "sin-codex")

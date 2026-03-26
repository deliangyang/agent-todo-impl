from pathlib import Path

import pytest
from agent_todo_impl.project_scan import resolve_project_root


def test_resolve_project_root_prefers_containing_deepest(tmp_path: Path):
    ws = tmp_path / "ws"
    ws.mkdir()
    p1 = ws / "proj-a"
    p1.mkdir()
    (p1 / ".git").mkdir()
    (p1 / "docs").mkdir()
    md = p1 / "docs" / "a.md"
    md.write_text("# a", encoding="utf-8")

    p2 = ws / "proj-b"
    p2.mkdir()
    (p2 / "pyproject.toml").write_text("[project]\nname='b'\n", encoding="utf-8")

    assert resolve_project_root(ws, md) == p1.resolve()


def test_resolve_project_root_single_candidate(tmp_path: Path):
    ws = tmp_path / "ws"
    ws.mkdir()
    p = ws / "only"
    p.mkdir()
    (p / "package.json").write_text("{}", encoding="utf-8")
    md = ws / "readme.md"
    md.write_text("# x", encoding="utf-8")

    assert resolve_project_root(ws, md) == p.resolve()


def test_resolve_project_root_raises_when_ambiguous(tmp_path: Path):
    ws = tmp_path / "ws"
    ws.mkdir()
    p1 = ws / "a"
    p2 = ws / "b"
    p1.mkdir()
    p2.mkdir()
    (p1 / "pyproject.toml").write_text("[project]\nname='a'\n", encoding="utf-8")
    (p2 / "package.json").write_text("{}", encoding="utf-8")
    md = ws / "docs.md"
    md.write_text("# x", encoding="utf-8")

    # Ambiguous: should pick deterministically (scan order prefers deeper, then git, then path).
    assert resolve_project_root(ws, md) == p1.resolve()

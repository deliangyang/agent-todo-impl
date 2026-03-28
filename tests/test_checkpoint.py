from __future__ import annotations

import json
from pathlib import Path

import pytest
from agent_todo_impl.checkpoint import (
    CHECKPOINT_VERSION,
    RunCheckpoint,
    in_progress_checkpoint_path,
    read_checkpoint,
    session_checkpoint_path,
    write_checkpoint,
)
from agent_todo_impl.orchestrator import Orchestrator, OrchestratorConfig


def test_checkpoint_roundtrip(tmp_path: Path) -> None:
    p = tmp_path / "ckpt.json"
    ckpt = RunCheckpoint(
        repo_root="/repo",
        md_path="/repo/docs",
        text_snippets=["a"],
        image_paths=[],
        image_urls=[],
        run_branch="run/1-test",
        plan_text='{"todos":[]}',
        todos=[{"id": "1", "content": "do"}],
        session_id="sess-1",
        implement_next_index=1,
        enable_review=False,
    )
    write_checkpoint(p, ckpt)
    loaded = read_checkpoint(p)
    assert loaded.repo_root == ckpt.repo_root
    assert loaded.todos == ckpt.todos
    assert loaded.implement_next_index == 1
    assert loaded.session_id == "sess-1"
    raw = json.loads(p.read_text(encoding="utf-8"))
    assert raw["version"] == CHECKPOINT_VERSION


def test_checkpoint_atomic_write(tmp_path: Path) -> None:
    p = tmp_path / "d" / "ckpt.json"
    write_checkpoint(p, RunCheckpoint(repo_root="/r"))
    assert p.is_file()
    assert not (tmp_path / "d" / "ckpt.json.tmp").exists()


def test_validate_checkpoint_mismatch_repo(tmp_path: Path) -> None:
    orch = Orchestrator(
        OrchestratorConfig(
            repo_root=tmp_path,
            md_path=None,
            model="gpt-4.1-mini",
            executor="cursor",
        )
    )
    bad = RunCheckpoint(repo_root="/other", todos=[{"id": "1", "content": "x"}])
    with pytest.raises(RuntimeError, match="repo_root"):
        orch._validate_checkpoint(bad)


def test_session_checkpoint_path_stable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENT_TODO_CHECKPOINT_DIR", str(tmp_path))
    sid = "cursor-sess-abc"
    p1 = session_checkpoint_path(tmp_path, sid)
    p2 = session_checkpoint_path(tmp_path, sid)
    assert p1 == p2
    assert p1.name.endswith(".json")
    assert "sessions" in p1.parts
    assert p1.is_relative_to(tmp_path)


def test_in_progress_scoped_by_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENT_TODO_CHECKPOINT_DIR", str(tmp_path))
    a = tmp_path / "repo_a"
    b = tmp_path / "repo_b"
    a.mkdir()
    b.mkdir()
    pa = in_progress_checkpoint_path(a)
    pb = in_progress_checkpoint_path(b)
    assert pa != pb
    assert "in-progress" in pa.parts

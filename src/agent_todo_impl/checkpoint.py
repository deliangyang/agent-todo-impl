from __future__ import annotations

import hashlib
import json
import os
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

# Default: ~/.agent-todo/checkpoint/{in-progress|sessions}/...
# Override root for tests: AGENT_TODO_CHECKPOINT_DIR

CHECKPOINT_VERSION = 1

Phase = Literal["implement", "review", "finished"]


@dataclass
class RunCheckpoint:
    """Persisted state for cursor executor resume after interrupt or failure."""

    version: int = CHECKPOINT_VERSION
    repo_root: str = ""
    md_path: str | None = None
    text_snippets: list[str] = field(default_factory=list)
    image_paths: list[str] = field(default_factory=list)
    image_urls: list[str] = field(default_factory=list)
    run_branch: str | None = None
    plan_text: str = ""
    todos: list[dict[str, str]] = field(default_factory=list)
    session_id: str | None = None
    phase: Phase = "implement"
    implement_next_index: int = 0
    review_round: int = 0
    review_fix_next_index: int = 0
    review_fix_todos: list[dict[str, str]] | None = None
    enable_review: bool = False

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json_dict(cls, data: dict[str, Any]) -> RunCheckpoint:
        v = int(data.get("version", 0))
        if v != CHECKPOINT_VERSION:
            raise ValueError(f"unsupported checkpoint version: {v}")
        phase = data.get("phase", "implement")
        if phase not in ("implement", "review", "finished"):
            raise ValueError(f"invalid phase: {phase}")
        todos = data.get("todos") or []
        if not isinstance(todos, list):
            raise ValueError("todos must be a list")
        rft = data.get("review_fix_todos")
        if rft is not None and not isinstance(rft, list):
            raise ValueError("review_fix_todos must be a list or null")
        return cls(
            version=CHECKPOINT_VERSION,
            repo_root=str(data.get("repo_root", "")),
            md_path=data.get("md_path"),
            text_snippets=list(data.get("text_snippets") or []),
            image_paths=[str(p) for p in (data.get("image_paths") or [])],
            image_urls=list(data.get("image_urls") or []),
            run_branch=data.get("run_branch"),
            plan_text=str(data.get("plan_text", "")),
            todos=[dict(x) for x in todos],
            session_id=data.get("session_id"),
            phase=phase,  # type: ignore[assignment]
            implement_next_index=int(data.get("implement_next_index", 0)),
            review_round=int(data.get("review_round", 0)),
            review_fix_next_index=int(data.get("review_fix_next_index", 0)),
            review_fix_todos=[dict(x) for x in rft] if rft else None,
            enable_review=bool(data.get("enable_review", False)),
        )


def write_checkpoint(path: Path, ckpt: RunCheckpoint) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    payload = json.dumps(ckpt.to_json_dict(), ensure_ascii=False, indent=2)
    tmp.write_text(payload, encoding="utf-8")
    tmp.replace(path)


def read_checkpoint(path: Path) -> RunCheckpoint:
    raw = path.read_text(encoding="utf-8")
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("checkpoint must be a JSON object")
    return RunCheckpoint.from_json_dict(data)


def checkpoint_root() -> Path:
    """User directory ~/.agent-todo/checkpoint (or AGENT_TODO_CHECKPOINT_DIR)."""
    override = os.environ.get("AGENT_TODO_CHECKPOINT_DIR")
    if override:
        return Path(override).expanduser().resolve()
    return (Path.home() / ".agent-todo" / "checkpoint").resolve()


def in_progress_checkpoint_path(repo_root: Path) -> Path:
    """Before first successful todo: one file per repo (keyed by repo path hash)."""
    repo_key = hashlib.sha256(str(repo_root.resolve()).encode("utf-8")).hexdigest()
    return checkpoint_root() / "in-progress" / f"{repo_key}.json"


def session_checkpoint_path(_repo_root: Path, session_id: str) -> Path:
    """After session_id is known; filename from session_id hash."""
    h = hashlib.sha256(session_id.encode("utf-8")).hexdigest()
    return checkpoint_root() / "sessions" / f"{h}.json"


# Set by orchestrator during run; SIGINT handler flushes last known state.
_active_lock = threading.Lock()
_active_path: Path | None = None
_active_snapshot: dict[str, Any] | None = None


def register_active_checkpoint(path: Path | None, snapshot: dict[str, Any] | None) -> None:
    global _active_path, _active_snapshot
    with _active_lock:
        _active_path = path.resolve() if path else None
        _active_snapshot = dict(snapshot) if snapshot else None


def flush_active_checkpoint() -> None:
    """Write registered snapshot to disk (used on SIGINT)."""
    with _active_lock:
        path = _active_path
        snap = _active_snapshot
    if path is None or snap is None:
        return
    try:
        ckpt = RunCheckpoint.from_json_dict(snap)
        write_checkpoint(path, ckpt)
    except Exception:
        pass


def clear_active_checkpoint() -> None:
    register_active_checkpoint(None, None)

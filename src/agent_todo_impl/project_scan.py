from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectCandidate:
    root: Path
    has_git: bool
    markers: tuple[str, ...]


_MARKER_FILES: tuple[str, ...] = (
    "pyproject.toml",
    "package.json",
    "go.mod",
    "Cargo.toml",
    "pom.xml",
    "build.gradle",
    "build.gradle.kts",
)


def _is_git_repo(dir_path: Path) -> bool:
    return (dir_path / ".git").exists()


def _marker_files_in(dir_path: Path) -> tuple[str, ...]:
    found: list[str] = []
    for name in _MARKER_FILES:
        if (dir_path / name).is_file():
            found.append(name)
    return tuple(found)


def scan_projects(workspace_root: Path, *, max_depth: int = 4) -> list[ProjectCandidate]:
    """
    Scan for possible project roots under workspace_root.
    A candidate is a directory with .git OR at least one known marker file.
    """
    workspace_root = workspace_root.resolve()
    if not workspace_root.is_dir():
        return []

    candidates: list[ProjectCandidate] = []
    # Use os.walk so we can prune heavy directories (like .git) early.
    for dirpath, dirnames, _filenames in os.walk(workspace_root):
        p = Path(dirpath)
        try:
            rel = p.relative_to(workspace_root)
        except ValueError:
            continue
        depth = len(rel.parts)
        if depth > max_depth:
            dirnames[:] = []
            continue

        # Prune common heavy dirs; never scan inside .git.
        dirnames[:] = [
            d
            for d in dirnames
            if d not in {".git", "node_modules", ".venv", "dist", "build", "__pycache__"}
        ]

        has_git = _is_git_repo(p)
        markers = _marker_files_in(p)
        if has_git or markers:
            candidates.append(ProjectCandidate(root=p, has_git=has_git, markers=markers))

    # Prefer deeper (more specific) roots first, then git repos.
    candidates.sort(
        key=lambda c: (
            -len(c.root.relative_to(workspace_root).parts),
            0 if c.has_git else 1,
            str(c.root),
        )
    )
    return candidates


def resolve_project_root(workspace_root: Path, md_path: Path) -> Path:
    """
    Resolve the target project root for a given md_path.

    Strategy:
    - If md_path is inside a candidate root, choose the deepest such root.
    - Else if exactly one candidate exists, choose it.
    - Else if workspace_root itself looks like a project, use workspace_root.
    - Else if multiple candidates exist, choose the best-ranked candidate (deterministic).
    - Else (no candidates), fall back to workspace_root.
    """
    workspace_root = workspace_root.resolve()
    md_path = md_path.resolve()

    candidates = scan_projects(workspace_root)
    containing: list[ProjectCandidate] = []
    for c in candidates:
        try:
            md_path.relative_to(c.root.resolve())
        except ValueError:
            continue
        containing.append(c)

    if containing:
        # candidates already sorted deepest-first
        return containing[0].root.resolve()

    if len(candidates) == 1:
        return candidates[0].root.resolve()

    if _is_git_repo(workspace_root) or _marker_files_in(workspace_root):
        return workspace_root

    # Deterministic fallback: pick the "best" candidate based on scan ordering,
    # or fall back to workspace_root if nothing was detected.
    if candidates:
        return candidates[0].root.resolve()
    return workspace_root

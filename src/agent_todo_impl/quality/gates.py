from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class GateResult:
    ok: bool
    output: str


def _run(cmd: list[str], *, cwd: Path) -> tuple[int, str]:
    env = os.environ.copy()
    # Avoid global pytest plugins pulling in optional native deps.
    env.setdefault("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
    p = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True, env=env)
    out = (p.stdout or "") + (("\n" + p.stderr) if p.stderr else "")
    return p.returncode, out.strip()


def _changed_paths(repo_root: Path) -> set[str]:
    """
    Best-effort list of changed files in the working tree.
    Includes unstaged/staged modifications and untracked files.
    """
    paths: set[str] = set()

    # Modified/added/deleted (unstaged)
    rc, out = _run(["git", "diff", "--name-only"], cwd=repo_root)
    if rc == 0 and out:
        paths.update(p for p in out.splitlines() if p.strip())

    # Staged changes (in case caller staged before running gates)
    rc, out = _run(["git", "diff", "--cached", "--name-only"], cwd=repo_root)
    if rc == 0 and out:
        paths.update(p for p in out.splitlines() if p.strip())

    # Untracked files
    rc, out = _run(["git", "status", "--porcelain"], cwd=repo_root)
    if rc == 0 and out:
        for line in out.splitlines():
            # Format: XY <path>  (or for renames: XY <from> -> <to>)
            if not line:
                continue
            status = line[:2]
            rest = line[3:] if len(line) >= 4 else ""
            if status == "??" and rest:
                # Keep the "to" path for renames-like output, otherwise the only path.
                if " -> " in rest:
                    rest = rest.split(" -> ", 1)[1]
                paths.add(rest.strip())

    return {p for p in paths if p}


def _needs_python_gates(changed_paths: set[str]) -> bool:
    if not changed_paths:
        # When we cannot detect changes, be conservative and run gates.
        return True

    python_trigger_files = {
        "pyproject.toml",
        "pytest.ini",
        "setup.cfg",
        "tox.ini",
        "requirements.txt",
        "requirements-dev.txt",
    }
    for p in changed_paths:
        name = Path(p).name
        if name in python_trigger_files:
            return True
        if p.endswith(".py") or p.endswith(".pyi"):
            return True
    return False


def run_quality_gates(repo_root: Path) -> GateResult:
    if not (repo_root / ".git").exists():
        return GateResult(
            ok=True,
            output=(
                "$ quality gates\n"
                "(skipped: repo_root is not a git repository; cannot detect changed files)\n"
                f"repo_root={repo_root}"
            ).strip(),
        )

    changed = _changed_paths(repo_root)
    if not _needs_python_gates(changed):
        files = "\n".join(sorted(changed)) if changed else "(none detected)"
        return GateResult(
            ok=True,
            output=(
                "$ quality gates\n"
                "(skipped: no Python-related files changed)\n\n"
                "changed files:\n"
                f"{files}"
            ).strip(),
        )

    codes: list[tuple[str, int, str]] = []

    rc, out = _run(["ruff", "format"], cwd=repo_root)
    codes.append(("ruff format", rc, out))

    rc, out = _run(["ruff", "check"], cwd=repo_root)
    codes.append(("ruff check", rc, out))

    rc, out = _run([sys.executable, "-m", "pytest"], cwd=repo_root)
    codes.append(("pytest", rc, out))

    ok = all(rc == 0 for _, rc, _ in codes)
    combined = "\n\n".join(
        [f"$ {name}\n(exit={rc})\n{out}".strip() for name, rc, out in codes]
    ).strip()
    return GateResult(ok=ok, output=combined)

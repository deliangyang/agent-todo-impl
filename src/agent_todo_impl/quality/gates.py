from __future__ import annotations

import os
import subprocess
import sys
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class GateResult:
    ok: bool
    output: str


@dataclass(frozen=True)
class SyntaxChecker:
    name: str
    commands: list[list[str]]  # List of commands to run
    check_availability: Callable[[Path], bool]  # Check if tool is available


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


def _classify_files_by_extension(changed_paths: set[str]) -> dict[str, list[str]]:
    """
    Classify changed files by their language/extension type.
    Returns a dict mapping language type to list of file paths.
    """
    classified: dict[str, list[str]] = defaultdict(list)
    
    extension_map = {
        # Python
        ".py": "python",
        ".pyi": "python",
        # JavaScript/TypeScript
        ".js": "javascript",
        ".jsx": "javascript",
        ".ts": "typescript",
        ".tsx": "typescript",
        # Go
        ".go": "go",
        # Java
        ".java": "java",
        # Rust
        ".rs": "rust",
        # Ruby
        ".rb": "ruby",
        # PHP
        ".php": "php",
        # Shell
        ".sh": "shell",
        ".bash": "shell",
        # C/C++
        ".c": "c",
        ".cpp": "cpp",
        ".cc": "cpp",
        ".cxx": "cpp",
        ".h": "c",
        ".hpp": "cpp",
    }
    
    for path in changed_paths:
        ext = Path(path).suffix.lower()
        if ext in extension_map:
            lang = extension_map[ext]
            classified[lang].append(path)
    
    return dict(classified)


def _command_exists(cmd: str, cwd: Path) -> bool:
    """Check if a command is available in PATH."""
    rc, _ = _run(["which", cmd], cwd=cwd)
    return rc == 0


def _get_syntax_checkers(lang: str, repo_root: Path) -> list[SyntaxChecker]:
    """
    Return syntax checkers for the given language.
    Only returns checkers whose tools are available.
    """
    checkers_map = {
        "python": [
            SyntaxChecker(
                name="python compile",
                commands=[[sys.executable, "-m", "py_compile"]],
                check_availability=lambda p: True,  # Python always available
            ),
            SyntaxChecker(
                name="ruff check",
                commands=[["ruff", "check"]],
                check_availability=lambda p: _command_exists("ruff", p),
            ),
            SyntaxChecker(
                name="mypy",
                commands=[["mypy", "."]],
                check_availability=lambda p: _command_exists("mypy", p),
            ),
        ],
        "javascript": [
            SyntaxChecker(
                name="eslint",
                commands=[["npx", "eslint", "."]],
                check_availability=lambda p: (p / "package.json").exists(),
            ),
        ],
        "typescript": [
            SyntaxChecker(
                name="tsc",
                commands=[["npx", "tsc", "--noEmit"]],
                check_availability=lambda p: (p / "tsconfig.json").exists(),
            ),
            SyntaxChecker(
                name="eslint",
                commands=[["npx", "eslint", "."]],
                check_availability=lambda p: (p / "package.json").exists(),
            ),
        ],
        "go": [
            SyntaxChecker(
                name="gofmt",
                commands=[["gofmt", "-l", "."]],
                check_availability=lambda p: _command_exists("gofmt", p),
            ),
            SyntaxChecker(
                name="go vet",
                commands=[["go", "vet", "./..."]],
                check_availability=lambda p: _command_exists("go", p),
            ),
        ],
        "java": [
            SyntaxChecker(
                name="javac",
                commands=[["javac", "-Xlint"]],
                check_availability=lambda p: _command_exists("javac", p),
            ),
        ],
        "rust": [
            SyntaxChecker(
                name="cargo check",
                commands=[["cargo", "check"]],
                check_availability=lambda p: (p / "Cargo.toml").exists(),
            ),
            SyntaxChecker(
                name="cargo clippy",
                commands=[["cargo", "clippy"]],
                check_availability=lambda p: (p / "Cargo.toml").exists(),
            ),
        ],
        "ruby": [
            SyntaxChecker(
                name="ruby syntax",
                commands=[["ruby", "-c"]],
                check_availability=lambda p: _command_exists("ruby", p),
            ),
        ],
        "php": [
            SyntaxChecker(
                name="php lint",
                commands=[["php", "-l"]],
                check_availability=lambda p: _command_exists("php", p),
            ),
        ],
        "shell": [
            SyntaxChecker(
                name="shellcheck",
                commands=[["shellcheck"]],
                check_availability=lambda p: _command_exists("shellcheck", p),
            ),
        ],
    }
    
    available_checkers = []
    for checker in checkers_map.get(lang, []):
        if checker.check_availability(repo_root):
            available_checkers.append(checker)
    
    return available_checkers


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
    if not changed:
        return GateResult(
            ok=True,
            output=(
                "$ quality gates\n"
                "(skipped: no changed files detected)"
            ).strip(),
        )
    
    # Classify files by language
    classified = _classify_files_by_extension(changed)
    
    if not classified:
        files = "\n".join(sorted(changed))
        return GateResult(
            ok=True,
            output=(
                "$ quality gates\n"
                "(skipped: no supported file types changed)\n\n"
                "changed files:\n"
                f"{files}"
            ).strip(),
        )
    
    # Build summary of what we found
    summary_lines = ["$ quality gates", ""]
    summary_lines.append("Changed files by type:")
    for lang, files in sorted(classified.items()):
        summary_lines.append(f"  {lang}: {len(files)} file(s)")
    summary_lines.append("")
    
    # Run syntax checkers for each language
    codes: list[tuple[str, int, str]] = []
    
    for lang, files in sorted(classified.items()):
        checkers = _get_syntax_checkers(lang, repo_root)
        
        if not checkers:
            codes.append((
                f"{lang} (no checkers available)",
                0,
                f"No syntax checkers found for {lang} files"
            ))
            continue
        
        for checker in checkers:
            for cmd in checker.commands:
                # For file-specific commands, append file paths
                if checker.name in ["python compile", "ruby syntax", "php lint", "shellcheck"]:
                    for file_path in files:
                        full_cmd = cmd + [file_path]
                        rc, out = _run(full_cmd, cwd=repo_root)
                        codes.append((f"{checker.name} ({Path(file_path).name})", rc, out))
                else:
                    rc, out = _run(cmd, cwd=repo_root)
                    codes.append((f"{checker.name} ({lang})", rc, out))
    
    ok = all(rc == 0 for _, rc, _ in codes)
    
    # Build output
    check_results = "\n\n".join(
        [f"$ {name}\n(exit={rc})\n{out}".strip() for name, rc, out in codes]
    ).strip()
    
    combined = "\n".join(summary_lines) + "\n\n" + check_results
    
    return GateResult(ok=ok, output=combined)

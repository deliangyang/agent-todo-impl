from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class GateResult:
    ok: bool
    output: str


def _run(cmd: list[str], *, cwd: Path) -> tuple[int, str]:
    p = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True)
    out = (p.stdout or "") + (("\n" + p.stderr) if p.stderr else "")
    return p.returncode, out.strip()


def run_quality_gates(repo_root: Path) -> GateResult:
    codes: list[tuple[str, int, str]] = []

    rc, out = _run(["ruff", "format"], cwd=repo_root)
    codes.append(("ruff format", rc, out))

    rc, out = _run(["ruff", "check"], cwd=repo_root)
    codes.append(("ruff check", rc, out))

    rc, out = _run(["pytest"], cwd=repo_root)
    codes.append(("pytest", rc, out))

    ok = all(rc == 0 for _, rc, _ in codes)
    combined = "\n\n".join(
        [f"$ {name}\n(exit={rc})\n{out}".strip() for name, rc, out in codes]
    ).strip()
    return GateResult(ok=ok, output=combined)

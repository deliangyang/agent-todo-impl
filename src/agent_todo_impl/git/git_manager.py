from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True)
class GitManager:
    repo_root: Path

    def _run(self, args: list[str]) -> str:
        if args[:2] == ["git", "merge"]:
            raise RuntimeError("git merge is forbidden")
        if args[:2] == ["git", "push"] and any(
            a in {"--force", "-f", "--force-with-lease"} for a in args[2:]
        ):
            raise RuntimeError("git push -f is forbidden")
        p = subprocess.run(args, cwd=self.repo_root, text=True, capture_output=True)
        if p.returncode != 0:
            raise RuntimeError(f"git failed: {' '.join(args)}\n{p.stderr}".strip())
        return p.stdout.strip()

    def is_repo(self) -> bool:
        return (self.repo_root / ".git").exists()

    def ensure_repo(self) -> None:
        if self.is_repo():
            return
        self._run(["git", "init", "-b", "main"])

    def current_branch(self) -> str:
        return self._run(["git", "rev-parse", "--abbrev-ref", "HEAD"])

    def create_run_branch(self, slug: str) -> str:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        slug = re.sub(r"[^a-zA-Z0-9-]+", "-", slug).strip("-").lower() or "run"
        name = f"run/{ts}-{slug}"
        self._run(["git", "checkout", "-b", name])
        return name

    def add_all(self) -> None:
        self._run(["git", "add", "-A"])

    def commit(self, message: str) -> str:
        self._run(["git", "commit", "-m", message])
        return self._run(["git", "rev-parse", "HEAD"])

    def diff(self) -> str:
        return self._run(["git", "diff", "--patch"])

    def status_porcelain(self) -> str:
        return self._run(["git", "status", "--porcelain"])

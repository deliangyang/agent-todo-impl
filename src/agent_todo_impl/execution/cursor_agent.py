from __future__ import annotations

import os
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path

from agent_todo_impl.planning.plan_parser import TodoItem


@dataclass(frozen=True)
class CursorAgentConfig:
    workspace: Path
    model: str = "auto"
    api_key: str | None = None
    trust: bool = True
    force: bool = True
    output_format: str = "stream-json"
    stream_partial_output: bool = True


@dataclass(frozen=True)
class CursorAgentRunResult:
    command: list[str]
    prompt: str
    exit_code: int
    stdout: str
    stderr: str

    def model_dump(self) -> dict:
        return {
            "command": self.command,
            "prompt": self.prompt,
            "exit_code": self.exit_code,
            "stdout": self.stdout,
            "stderr": self.stderr,
        }


def build_cursor_agent_prompt(todos: list[TodoItem], *, repo_snapshot_hint: str) -> str:
    todo_lines = "\n".join([f"- {t.id}: {t.content}" for t in todos])
    return (
        "你是 Cursor CLI agent（具备读写代码和运行命令能力）。\n"
        "请在当前 workspace 内完成以下 TODO，实现后确保质量门禁通过"
        "（ruff format/check + pytest）。\n"
        "约束：只修改仓库内文件；不要做 git push/force；不要引入无关依赖。\n\n"
        f"仓库提示：{repo_snapshot_hint}\n\n"
        "TODO：\n"
        f"{todo_lines}\n"
    )


def build_cursor_agent_command(cfg: CursorAgentConfig, *, prompt: str) -> list[str]:
    cmd: list[str] = [
        "cursor-agent",
        "agent",
        "--print",
        "--output-format",
        cfg.output_format,
    ]
    if cfg.stream_partial_output:
        cmd.append("--stream-partial-output")
    cmd += [
        "--workspace",
        str(cfg.workspace),
    ]
    if cfg.trust:
        cmd.append("--trust")
    if cfg.force:
        cmd.append("--force")
    cmd += ["--model", cfg.model]
    # Prefer explicit flag; fallback to env var handled by cursor-agent itself
    api_key = cfg.api_key or os.getenv("CURSOR_API_KEY")
    if api_key:
        cmd += ["--api-key", api_key]

    cmd.append(prompt)
    return cmd


def cursor_agent_command_string(cmd: list[str]) -> str:
    return " ".join(shlex.quote(x) for x in cmd)


def run_cursor_agent(cfg: CursorAgentConfig, *, prompt: str) -> CursorAgentRunResult:
    cmd = build_cursor_agent_command(cfg, prompt=prompt)
    p = subprocess.run(cmd, text=True, capture_output=True)
    return CursorAgentRunResult(
        command=cmd,
        prompt=prompt,
        exit_code=p.returncode,
        stdout=(p.stdout or "").strip(),
        stderr=(p.stderr or "").strip(),
    )

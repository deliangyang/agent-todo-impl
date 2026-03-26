from __future__ import annotations

import logging
import os
import shlex
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path

from agent_todo_impl.planning.plan_parser import TodoItem

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CursorAgentConfig:
    workspace: Path
    run_cwd: Path | None = None
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


def _stream_pipe_to_terminal(pipe, tee, parts: list[str]) -> None:
    """Read subprocess pipe in chunks; echo to tee and append to parts (for model_dump)."""
    try:
        while True:
            chunk = pipe.read(4096)
            if not chunk:
                break
            parts.append(chunk)
            tee.write(chunk)
            tee.flush()
    finally:
        pipe.close()


def _cursor_agent_command_for_log(cmd: list[str]) -> str:
    parts: list[str] = []
    i = 0
    while i < len(cmd):
        if cmd[i] == "--api-key" and i + 1 < len(cmd):
            parts.extend(["--api-key", "***REDACTED***"])
            i += 2
        else:
            parts.append(cmd[i])
            i += 1
    return cursor_agent_command_string(parts)


def run_cursor_agent(cfg: CursorAgentConfig, *, prompt: str) -> CursorAgentRunResult:
    cmd = build_cursor_agent_command(cfg, prompt=prompt)
    cwd = (cfg.run_cwd or Path.cwd()).resolve()
    logger.info(
        "cursor-agent 执行目录: %s 命令: %s",
        cwd,
        _cursor_agent_command_for_log(cmd),
    )
    out_parts: list[str] = []
    err_parts: list[str] = []
    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    t_out = threading.Thread(
        target=_stream_pipe_to_terminal,
        args=(proc.stdout, sys.stdout, out_parts),
    )
    t_err = threading.Thread(
        target=_stream_pipe_to_terminal,
        args=(proc.stderr, sys.stderr, err_parts),
    )
    t_out.start()
    t_err.start()
    exit_code = proc.wait()
    t_out.join()
    t_err.join()
    return CursorAgentRunResult(
        command=cmd,
        prompt=prompt,
        exit_code=exit_code,
        stdout="".join(out_parts).strip(),
        stderr="".join(err_parts).strip(),
    )

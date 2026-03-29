from __future__ import annotations

import logging
import shlex
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path

from agent_todo_impl.execution.cursor_agent import CursorAgentRunResult
from agent_todo_impl.planning.plan_parser import TodoItem

logger = logging.getLogger(__name__)


# Executor names accepted by Orchestrator / CLI
EXTERNAL_CLI_EXECUTORS = frozenset({"copilot", "codex", "gemini"})


@dataclass(frozen=True)
class ExternalCliConfig:
    """Workspace root (repo) and working directory for the agent process."""

    workspace: Path
    run_cwd: Path | None = None
    model: str = "auto"


def external_cli_role_name(executor: str) -> str:
    return {
        "copilot": "GitHub Copilot CLI",
        "codex": "OpenAI Codex CLI",
        "gemini": "Google Gemini CLI",
    }[executor]


def build_external_cli_prompt_for_todo(
    executor: str, todo: TodoItem, *, repo_snapshot_hint: str
) -> str:
    role = external_cli_role_name(executor)
    return (
        f"你是 {role} agent（具备读写代码和运行命令能力）。\n"
        "请在当前 workspace 内只完成下面这一条 TODO，实现后确保质量门禁可通过。\n"
        "约束：只修改仓库内文件；不要做 git push/force；不要引入无关依赖。\n\n"
        f"仓库提示：{repo_snapshot_hint}\n\n"
        f"TODO（仅完成这一条）：\n- {todo.id}: {todo.content}\n"
    )


def build_copilot_command(cfg: ExternalCliConfig, *, prompt: str) -> list[str]:
    ws = str(cfg.workspace.resolve())
    cmd: list[str] = [
        "copilot",
        "--add-dir",
        ws,
        "-p",
        prompt,
        "--no-ask-user",
        "-s",
        "--allow-all",
    ]
    if cfg.model and cfg.model != "auto":
        cmd.extend(["--model", cfg.model])
    return cmd


def build_codex_command(cfg: ExternalCliConfig, *, prompt: str) -> list[str]:
    ws = str(cfg.workspace.resolve())
    cmd: list[str] = [
        "codex",
        "exec",
        "-C",
        ws,
        "--sandbox",
        "workspace-write",
        "--ask-for-approval",
        "never",
    ]
    if cfg.model and cfg.model != "auto":
        cmd.extend(["-m", cfg.model])
    cmd.append(prompt)
    return cmd


def build_gemini_command(cfg: ExternalCliConfig, *, prompt: str) -> list[str]:
    cmd: list[str] = [
        "gemini",
        "-p",
        prompt,
        "--yolo",
        "--include-directories",
        str(cfg.workspace.resolve()),
    ]
    if cfg.model and cfg.model != "auto":
        cmd.extend(["-m", cfg.model])
    return cmd


def build_external_cli_command(executor: str, cfg: ExternalCliConfig, *, prompt: str) -> list[str]:
    if executor == "copilot":
        return build_copilot_command(cfg, prompt=prompt)
    if executor == "codex":
        return build_codex_command(cfg, prompt=prompt)
    if executor == "gemini":
        return build_gemini_command(cfg, prompt=prompt)
    raise ValueError(f"unknown external CLI executor: {executor!r}")


def external_cli_command_string(cmd: list[str]) -> str:
    return " ".join(shlex.quote(x) for x in cmd)


def _stream_pipe_to_terminal(pipe, tee, parts: list[str]) -> None:
    """Read subprocess pipe; echo to tee and append to parts.

    For stdout, keep a rolling window of the last 10 lines in dim gray,
    similar to cursor-agent CLI. Stderr is streamed directly.
    """
    max_lines = 10
    is_stdout = tee is sys.stdout
    buffer = ""
    lines: list[str] = []
    printed_lines = 0

    def _render_window() -> None:
        nonlocal printed_lines
        if printed_lines:
            sys.stdout.write(f"\x1b[{printed_lines}F")
            sys.stdout.write("\x1b[J")
        printed_lines = len(lines)
        gray_prefix = "\x1b[90m"
        reset = "\x1b[0m"
        for line in lines:
            sys.stdout.write(f"{gray_prefix}{line}{reset}\n")
        sys.stdout.flush()

    try:
        while True:
            chunk = pipe.read(4096)
            if not chunk:
                break
            parts.append(chunk)
            if not is_stdout:
                tee.write(chunk)
                tee.flush()
                continue

            buffer += chunk
            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                lines.append(line)
                if len(lines) > max_lines:
                    lines = lines[-max_lines:]
            _render_window()
    finally:
        pipe.close()


def run_external_cli(executor: str, cfg: ExternalCliConfig, *, prompt: str) -> CursorAgentRunResult:
    cmd = build_external_cli_command(executor, cfg, prompt=prompt)
    cwd = (cfg.run_cwd or Path.cwd()).resolve()
    logger.info(
        "%s 执行目录: %s 命令: %s",
        external_cli_role_name(executor),
        cwd,
        external_cli_command_string(cmd),
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
        session_id=None,
    )

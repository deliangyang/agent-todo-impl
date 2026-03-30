from __future__ import annotations

import json
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


def todo_run_cwd_for_md_path(md_path: Path) -> Path:
    """Use command invocation directory as the execution cwd.

    Keep md_path in the signature for backward compatibility with existing
    call sites, but do not derive runtime cwd from document location.
    """
    _ = md_path
    return Path.cwd().resolve()


@dataclass(frozen=True)
class CursorAgentConfig:
    workspace: Path
    run_cwd: Path | None = None
    model: str = "auto"
    api_key: str | None = None
    trust: bool = True
    force: bool = True
    # Continue the same chat as a previous successful JSON response (--resume <id>).
    resume_session_id: str | None = None
    # cursor-agent supports: text | json | stream-json
    # We default to stream-json for real-time streaming output.
    output_format: str = "stream-json"
    stream_partial_output: bool = True


@dataclass(frozen=True)
class CursorAgentRunResult:
    command: list[str]
    prompt: str
    exit_code: int
    stdout: str
    stderr: str
    # From --output-format json terminal object when present
    session_id: str | None = None

    def model_dump(self) -> dict:
        return {
            "command": self.command,
            "prompt": self.prompt,
            "exit_code": self.exit_code,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "session_id": self.session_id,
        }


def parse_cursor_agent_json_stdout(stdout: str) -> tuple[str | None, str]:
    """Parse cursor-agent `--output-format json` line. Returns (session_id, result text)."""
    raw = stdout.strip()
    if not raw:
        return None, ""
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None, raw
    if not isinstance(data, dict):
        return None, raw
    sid = data.get("session_id")
    result = data.get("result")
    sid_str = sid if isinstance(sid, str) else None
    res_str = result if isinstance(result, str) else ""
    return sid_str, res_str


def parse_cursor_agent_stream_json_stdout(stdout: str) -> tuple[str | None, str]:
    """Parse cursor-agent `--output-format stream-json` NDJSON output.

    Scans lines in reverse to find the last JSON object containing a ``result``
    field (the terminal summary line). Returns (session_id, result text).
    """
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(data, dict):
            continue
        if "result" in data:
            sid = data.get("session_id")
            result = data["result"]
            sid_str = sid if isinstance(sid, str) else None
            res_str = result if isinstance(result, str) else ""
            return sid_str, res_str
    return None, stdout


def extract_result_text(stdout: str, output_format: str) -> tuple[str | None, str]:
    """Unified result extractor for all cursor-agent output formats.

    Returns (session_id, result_text) regardless of ``output_format``.
    For ``text`` format, session_id is always None and result_text is the raw stdout.
    """
    if output_format == "json":
        return parse_cursor_agent_json_stdout(stdout)
    if output_format == "stream-json":
        return parse_cursor_agent_stream_json_stdout(stdout)
    return None, stdout


def build_cursor_agent_prompt_for_todo(todo: TodoItem, *, repo_snapshot_hint: str) -> str:
    return (
        "你是 Cursor CLI agent（具备读写代码和运行命令能力）。\n"
        "请在当前 workspace 内只完成下面这一条 TODO，实现后确保质量门禁可通过。\n"
        "约束：只修改仓库内文件；不要做 git push/force；不要引入无关依赖。\n\n"
        f"仓库提示：{repo_snapshot_hint}\n\n"
        f"TODO（仅完成这一条）：\n- {todo.id}: {todo.content}\n"
    )


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
    if cfg.output_format == "stream-json" and cfg.stream_partial_output:
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
    if cfg.resume_session_id:
        cmd += ["--resume", cfg.resume_session_id]

    cmd.append(prompt)
    return cmd


def cursor_agent_command_string(cmd: list[str]) -> str:
    return " ".join(shlex.quote(x) for x in cmd)


# ---------------------------------------------------------------------------
# ANSI styling constants (stream-json terminal renderer)
# ---------------------------------------------------------------------------
_R = "\x1b[0m"       # reset
_GRAY = "\x1b[90m"   # dim gray  — thinking text
_DIM = "\x1b[2m"     # dim       — thinking prefix / tool results
_ITALIC = "\x1b[3m"  # italic    — thinking
_CYAN = "\x1b[36m"   # cyan      — tool-use name
_BOLD = "\x1b[1m"    # bold      — tool-use name
_YELLOW = "\x1b[33m" # yellow    — tool result preview


def _render_stream_json_line(line: str, state: dict) -> str | None:
    """Render one NDJSON line from ``--output-format stream-json`` to a
    terminal string with ANSI styling.

    ``state`` is a mutable ``dict`` shared across successive calls::

        {"last": None}  # tracks last printed content-type for separator logic

    Returns a string to ``write()`` to stdout, or ``None`` to skip the line.

    Handled event / content types
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    * ``assistant / text``     — plain white, streamed token-by-token.
    * ``assistant / thinking`` — dim gray italic w/ ``💭`` prefix on entry.
    * ``assistant / tool_use`` — cyan bold ``● name(key: val …)`` on its own line.
    * ``tool_result``          — dim ``→ preview`` indented line.
    """
    try:
        data = json.loads(line)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(data, dict):
        return None

    event_type = data.get("type")

    # ------------------------------------------------------------------
    # assistant events: iterate over content blocks
    # ------------------------------------------------------------------
    if event_type == "assistant":
        msg = data.get("message") or {}
        content = msg.get("content") if isinstance(msg, dict) else None
        if not isinstance(content, list) or not content:
            return None

        parts: list[str] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            ctype = item.get("type")

            if ctype == "text":
                text = item.get("text") or ""
                if not text:
                    continue
                # Separate from a thinking / tool block with a newline.
                if state["last"] in ("thinking", "tool_use"):
                    parts.append("\n")
                parts.append(text)
                state["last"] = "text"

            elif ctype == "thinking":
                thinking = item.get("thinking") or ""
                if not thinking:
                    continue
                # Header on first thinking token of a new thinking block.
                if state["last"] != "thinking":
                    prefix = "\n" if state["last"] else ""
                    parts.append(f"{prefix}{_DIM}{_ITALIC}{_GRAY}💭 ")
                parts.append(f"{_DIM}{_ITALIC}{_GRAY}{thinking}{_R}")
                state["last"] = "thinking"

            elif ctype == "tool_use":
                name = item.get("name") or "unknown_tool"
                inp = item.get("input") or {}
                sep = "\n" if state["last"] else ""
                if isinstance(inp, dict) and inp:
                    inp_str = ", ".join(
                        f"{k}: {repr(v)[:50]}" for k, v in list(inp.items())[:4]
                    )
                    parts.append(f"{sep}{_CYAN}{_BOLD}● {name}({inp_str}){_R}\n")
                else:
                    parts.append(f"{sep}{_CYAN}{_BOLD}● {name}(){_R}\n")
                state["last"] = "tool_use"

        return "".join(parts) or None

    # ------------------------------------------------------------------
    # tool_result events: show a brief indented preview
    # ------------------------------------------------------------------
    if event_type == "tool_result":
        content = data.get("content") or []
        if isinstance(content, list):
            texts = [
                (c.get("text") or "")[:120]
                for c in content
                if isinstance(c, dict) and c.get("type") == "text"
            ]
            if texts:
                preview = texts[0].replace("\n", " ").strip()
                state["last"] = "tool_result"
                return f"  {_DIM}{_YELLOW}→ {preview}{_R}\n"
        return None

    return None


def _stream_pipe_to_terminal(pipe, tee, parts: list[str], *, stream_json: bool = False) -> None:
    """Read subprocess pipe; echo to tee and append to parts.

    For stdout with ``stream_json=True`` the pipe carries stream-json NDJSON.
    Each line is decoded and rendered with ANSI styling (see
    ``_render_stream_json_line``).  Raw JSON is never echoed.

    For plain stdout (``stream_json=False``) we keep a rolling window of the
    last 10 lines in dim gray.  Stderr is streamed directly without windowing.
    """
    is_stdout = tee is sys.stdout
    buffer = ""

    # ---- stream-json mode: styled real-time rendering ----
    if is_stdout and stream_json:
        render_state: dict = {"last": None}
        try:
            while True:
                chunk = pipe.read(4096)
                if not chunk:
                    break
                parts.append(chunk)
                buffer += chunk
                while "\n" in buffer:
                    ndjson_line, buffer = buffer.split("\n", 1)
                    ndjson_line = ndjson_line.strip()
                    if not ndjson_line:
                        continue
                    rendered = _render_stream_json_line(ndjson_line, render_state)
                    if rendered is not None:
                        sys.stdout.write(rendered)
                        sys.stdout.flush()
        finally:
            sys.stdout.write("\n")
            sys.stdout.flush()
            pipe.close()
        return

    # ---- default mode: rolling window for stdout, passthrough for stderr ----
    max_lines = 10
    lines: list[str] = []
    printed_lines = 0

    def _render_window() -> None:
        nonlocal printed_lines
        # Move cursor up to the beginning of the window and clear it.
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


def _cursor_agent_command_for_log(cmd: list[str]) -> str:
    parts: list[str] = []
    i = 0
    while i < len(cmd):
        if cmd[i] == "--api-key" and i + 1 < len(cmd):
            parts.extend(["--api-key", "***REDACTED***"])
            i += 2
        elif cmd[i] == "--resume" and i + 1 < len(cmd):
            parts.extend(["--resume", "***SESSION***"])
            i += 2
        else:
            parts.append(cmd[i])
            i += 1
    return cursor_agent_command_string(parts)


def run_cursor_agent(cfg: CursorAgentConfig, *, prompt: str) -> CursorAgentRunResult:
    cmd = build_cursor_agent_command(cfg, prompt=prompt)
    cwd = (cfg.run_cwd or Path.cwd()).resolve()
    logger.info(
        "cursor-agent 执行目录: %s ",
        cwd,
    )
    logger.info(
        'cursor-agent 命令: %s',
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
    use_stream_json = cfg.output_format == "stream-json"
    t_out = threading.Thread(
        target=_stream_pipe_to_terminal,
        args=(proc.stdout, sys.stdout, out_parts),
        kwargs={"stream_json": use_stream_json},
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
    stdout_full = "".join(out_parts).strip()
    stderr_full = "".join(err_parts).strip()
    session_id: str | None = None
    if cfg.output_format in ("json", "stream-json") and exit_code == 0:
        session_id, _ = extract_result_text(stdout_full, cfg.output_format)
    return CursorAgentRunResult(
        command=cmd,
        prompt=prompt,
        exit_code=exit_code,
        stdout=stdout_full,
        stderr=stderr_full,
        session_id=session_id,
    )

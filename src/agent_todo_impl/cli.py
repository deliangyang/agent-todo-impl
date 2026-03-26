from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import typer

from agent_todo_impl.execution.cursor_agent import (
    CursorAgentConfig,
    build_cursor_agent_command,
    build_cursor_agent_prompt,
    cursor_agent_command_string,
)
from agent_todo_impl.llm.openai_client import OpenAIClient, OpenAIClientConfig
from agent_todo_impl.mdscan import collect_markdown_context
from agent_todo_impl.orchestrator import Orchestrator, OrchestratorConfig
from agent_todo_impl.planning.plan_generator import build_plan_prompt
from agent_todo_impl.planning.plan_parser import parse_plan_text_to_todos
from agent_todo_impl.project_scan import resolve_project_root

app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.command()
def plan(
    md_path: Path = typer.Option(..., exists=True, readable=True, help="目录或单个 md 文件"),
    model: str = typer.Option("gpt-4.1-mini", envvar="AGENT_TODO_MODEL"),
    emit_cursor: bool = typer.Option(
        False, help="同时输出可直接交给 cursor-agent 执行的 prompt/command"
    ),
    cursor_model: Optional[str] = typer.Option(None, help="cursor-agent 使用的 model（默认 auto）"),
):
    """读取 md，调用 AI 生成 /plan todo（不改代码）。"""
    ctx = collect_markdown_context(md_path)
    prompt = build_plan_prompt(ctx)
    client = OpenAIClient(OpenAIClientConfig(model=model))
    plan_text = client.complete_text(prompt)
    todos = parse_plan_text_to_todos(plan_text)
    cursor_payload: dict | None = None
    if emit_cursor:
        repo_root = resolve_project_root(Path.cwd(), md_path)
        cursor_prompt = build_cursor_agent_prompt(
            todos,
            repo_snapshot_hint="python project with pyproject.toml, src/agent_todo_impl, tests/",
        )
        cmd = build_cursor_agent_command(
            CursorAgentConfig(workspace=repo_root, model=cursor_model or "auto"),
            prompt=cursor_prompt,
        )
        cursor_payload = {
            "prompt": cursor_prompt,
            "command": cmd,
            "command_str": cursor_agent_command_string(cmd),
        }
    typer.echo(
        json.dumps(
            {
                "plan_text": plan_text,
                "todos": [t.model_dump() for t in todos],
                "cursor": cursor_payload,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


@app.command()
def run(
    md_path: Path = typer.Option(..., exists=True, readable=True, help="目录或单个 md 文件"),
    model: str = typer.Option("gpt-4.1-mini", envvar="AGENT_TODO_MODEL"),
    executor: str = typer.Option("internal", help="执行器：internal | cursor"),
    cursor_model: Optional[str] = typer.Option(None, help="cursor-agent 使用的 model（默认 auto）"),
    cursor_force: bool = typer.Option(True, help="cursor-agent 允许强制执行（--force）"),
):
    """完整闭环：plan -> implement -> review(<=3) -> 自动提交（独立分支）。"""
    if executor == "cursor":
        logging.basicConfig(
            level=logging.INFO,
            format="%(levelname)s %(name)s: %(message)s",
        )
    repo_root = resolve_project_root(Path.cwd(), md_path)
    orchestrator = Orchestrator(
        OrchestratorConfig(
            repo_root=repo_root,
            md_path=md_path,
            model=model,
            executor=executor,
            cursor_model=cursor_model or "auto",
            cursor_force=cursor_force,
        )
    )
    result = orchestrator.run()
    typer.echo(json.dumps(result.model_dump(), ensure_ascii=False, indent=2))

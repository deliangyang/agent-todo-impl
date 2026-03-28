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
    run_cursor_agent,
    todo_run_cwd_for_md_path,
)
from agent_todo_impl.mdscan import EmptyRequirementContentError, collect_requirement_context
from agent_todo_impl.orchestrator import Orchestrator, OrchestratorConfig
from agent_todo_impl.planning.plan_generator import build_plan_prompt
from agent_todo_impl.planning.plan_parser import parse_plan_text_to_todos
from agent_todo_impl.project_scan import resolve_project_root

app = typer.Typer(add_completion=False, no_args_is_help=True)


def _require_at_least_one_source(
    md_path: Optional[Path],
    text: list[str],
    image: list[Path],
    image_url: list[str],
) -> None:
    if md_path is None and not text and not image and not image_url:
        typer.secho(
            "需要至少指定 --md-path、--text、--image 或 --image-url 之一",
            err=True,
        )
        raise typer.Exit(code=2)


def _repo_root_and_run_cwd(md_path: Optional[Path]) -> tuple[Path, Path]:
    anchor = (md_path or Path.cwd()).resolve()
    repo_root = resolve_project_root(Path.cwd(), anchor)
    run_cwd = todo_run_cwd_for_md_path(md_path) if md_path is not None else Path.cwd().resolve()
    return repo_root, run_cwd


@app.command()
def plan(
    md_path: Optional[Path] = typer.Option(
        None,
        exists=True,
        readable=True,
        help="目录、单个 .md、图片文件，或其它需求文件；可与下方选项组合",
    ),
    text: list[str] = typer.Option(
        [],
        "--text",
        "-t",
        help="内联文字（一行或多行由 shell 决定），可重复",
    ),
    image: list[Path] = typer.Option(
        [],
        "--image",
        "-i",
        exists=True,
        readable=True,
        help="本地图片路径，可重复",
    ),
    image_url: list[str] = typer.Option(
        [],
        "--image-url",
        help="图片 URL（http/https），可重复",
    ),
    emit_cursor: bool = typer.Option(
        False, help="同时输出可直接交给 cursor-agent 执行的 prompt/command"
    ),
    cursor_model: Optional[str] = typer.Option(None, help="cursor-agent 使用的 model（默认 auto）"),
):
    """读取需求（md、目录、内联文字、本地图片、图片 URL），用 cursor-agent 生成 /plan。"""
    _require_at_least_one_source(md_path, text, image, image_url)
    try:
        ctx = collect_requirement_context(
            md_path=md_path,
            text_snippets=text,
            image_paths=image,
            image_urls=image_url,
        )
    except EmptyRequirementContentError as e:
        typer.secho(str(e), err=True)
        raise typer.Exit(code=2) from e
    prompt = build_plan_prompt(ctx)
    repo_root, run_cwd = _repo_root_and_run_cwd(md_path)
    run = run_cursor_agent(
        CursorAgentConfig(
            workspace=repo_root,
            run_cwd=run_cwd,
            model=cursor_model or "auto",
            output_format="text",
        ),
        prompt=prompt,
    )
    if run.exit_code != 0:
        raise typer.Exit(code=run.exit_code)
    plan_text = run.stdout
    todos = parse_plan_text_to_todos(plan_text)
    cursor_payload: dict | None = None
    if emit_cursor:
        cursor_prompt = build_cursor_agent_prompt(
            todos,
            repo_snapshot_hint="python project with pyproject.toml, src/agent_todo_impl, tests/",
        )
        cmd = build_cursor_agent_command(
            CursorAgentConfig(workspace=repo_root, run_cwd=run_cwd, model=cursor_model or "auto"),
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
    md_path: Optional[Path] = typer.Option(
        None,
        exists=True,
        readable=True,
        help="目录、单个 .md、图片文件等；可与 --text / --image / --image-url 组合",
    ),
    text: list[str] = typer.Option(
        [],
        "--text",
        "-t",
        help="内联文字，可重复",
    ),
    image: list[Path] = typer.Option(
        [],
        "--image",
        "-i",
        exists=True,
        readable=True,
        help="本地图片路径，可重复",
    ),
    image_url: list[str] = typer.Option(
        [],
        "--image-url",
        help="图片 URL，可重复",
    ),
    model: str = typer.Option("gpt-4.1-mini", envvar="AGENT_TODO_MODEL"),
    executor: str = typer.Option("internal", help="执行器：internal | cursor"),
    cursor_model: Optional[str] = typer.Option(None, help="cursor-agent 使用的 model（默认 auto）"),
    cursor_force: bool = typer.Option(True, help="cursor-agent 允许强制执行（--force）"),
    cursor_output_format: str = typer.Option(
        "json",
        help="保留项；--executor cursor 时实现与 review 固定为 json（解析 session_id 并 --resume）",
    ),
    cursor_stream_partial_output: bool = typer.Option(
        False,
        help="保留项；当前 executor=cursor 实现未使用 stream-json",
    ),
    review: bool = typer.Option(
        False,
        "--review",
        help="启用 code review 与修复循环（最多 3 轮）；默认关闭",
    ),
):
    """完整闭环：plan -> implement -> 可选 review -> 自动提交（独立分支）。"""
    _require_at_least_one_source(md_path, text, image, image_url)
    if executor == "cursor":
        logging.basicConfig(
            level=logging.INFO,
            format="%(levelname)s %(name)s: %(message)s",
        )
    anchor = (md_path or Path.cwd()).resolve()
    repo_root = resolve_project_root(Path.cwd(), anchor)
    orchestrator = Orchestrator(
        OrchestratorConfig(
            repo_root=repo_root,
            md_path=md_path,
            model=model,
            enable_review=review,
            executor=executor,
            cursor_model=cursor_model or "auto",
            cursor_force=cursor_force,
            cursor_output_format=cursor_output_format,
            cursor_stream_partial_output=cursor_stream_partial_output,
            text_snippets=tuple(text),
            image_paths=tuple(image),
            image_urls=tuple(image_url),
        )
    )
    try:
        result = orchestrator.run()
    except EmptyRequirementContentError as e:
        typer.secho(str(e), err=True)
        raise typer.Exit(code=2) from e
    typer.echo(json.dumps(result.model_dump(), ensure_ascii=False, indent=2))

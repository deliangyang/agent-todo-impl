 from __future__ import annotations
 
 import json
 from pathlib import Path
 
 import typer
 
 from agent_todo_impl.llm.openai_client import OpenAIClient, OpenAIClientConfig
 from agent_todo_impl.mdscan import collect_markdown_context
 from agent_todo_impl.orchestrator import Orchestrator, OrchestratorConfig
 from agent_todo_impl.planning.plan_generator import build_plan_prompt
 from agent_todo_impl.planning.plan_parser import parse_plan_text_to_todos
 
 app = typer.Typer(add_completion=False, no_args_is_help=True)
 
 
 @app.command()
 def plan(
     md_path: Path = typer.Option(..., exists=True, readable=True, help="目录或单个 md 文件"),
     model: str = typer.Option("gpt-4.1-mini", envvar="AGENT_TODO_MODEL"),
 ):
     """读取 md，调用 AI 生成 /plan todo（不改代码）。"""
     ctx = collect_markdown_context(md_path)
     prompt = build_plan_prompt(ctx)
     client = OpenAIClient(OpenAIClientConfig(model=model))
     plan_text = client.complete_text(prompt)
     todos = parse_plan_text_to_todos(plan_text)
     typer.echo(json.dumps({"plan_text": plan_text, "todos": [t.model_dump() for t in todos]}, ensure_ascii=False, indent=2))
 
 
 @app.command()
 def run(
     md_path: Path = typer.Option(..., exists=True, readable=True, help="目录或单个 md 文件"),
     model: str = typer.Option("gpt-4.1-mini", envvar="AGENT_TODO_MODEL"),
 ):
     """完整闭环：plan -> implement -> review(<=3) -> 自动提交（独立分支）。"""
     orchestrator = Orchestrator(
         OrchestratorConfig(
             repo_root=Path.cwd(),
             md_path=md_path,
             model=model,
         )
     )
     result = orchestrator.run()
     typer.echo(json.dumps(result.model_dump(), ensure_ascii=False, indent=2))

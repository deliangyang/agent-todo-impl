 from __future__ import annotations
 
 import json
 from dataclasses import dataclass
 from pathlib import Path
 
 from agent_todo_impl.llm.base import LLMClient
 from agent_todo_impl.planning.plan_parser import TodoItem
 
 from .actions import ExecutionPlan, apply_execution_plan, parse_execution_json
 
 
 @dataclass(frozen=True)
 class ExecutorConfig:
     repo_root: Path
 
 
 def build_execution_prompt(todos: list[TodoItem], *, repo_snapshot_hint: str) -> str:
     return (
         "你是实现 agent。请根据 TODO 列表修改代码库（仅限仓库内文件）。\n"
         "输出严格的 JSON，不要额外文字：\n"
         "{\n"
         "  \"edits\": [\n"
         "    {\"path\": \"相对路径\", \"content\": \"该文件的完整新内容\"}\n"
         "  ]\n"
         "}\n\n"
         "约束：\n"
         "- 不要删除大量文件；优先小改动。\n"
         "- 让代码可运行、语法正确，尽量让 pytest 通过。\n\n"
        f"仓库提示：{repo_snapshot_hint}\n\n"
         "TODO：\n"
         + "\n".join([f"- {t.id}: {t.content}" for t in todos])
         + "\n"
     )
 
 
 class Executor:
     def __init__(self, config: ExecutorConfig, llm: LLMClient):
         self._config = config
         self._llm = llm
 
     def execute(self, todos: list[TodoItem], *, repo_snapshot_hint: str) -> ExecutionPlan:
         prompt = build_execution_prompt(todos, repo_snapshot_hint=repo_snapshot_hint)
         text = self._llm.complete_text(prompt).strip()
 
         # If model accidentally wraps JSON, try to extract first/last braces.
         if not text.startswith("{"):
             start = text.find("{")
             end = text.rfind("}")
             if start != -1 and end != -1 and end > start:
                 text = text[start : end + 1]
 
         plan = parse_execution_json(text)
         apply_execution_plan(self._config.repo_root, plan)
         return plan

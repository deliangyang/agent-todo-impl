 from __future__ import annotations
 
 from dataclasses import dataclass
 from pathlib import Path
 
 from agent_todo_impl.execution.executor import Executor, ExecutorConfig
 from agent_todo_impl.git.git_manager import GitManager
 from agent_todo_impl.llm.openai_client import OpenAIClient, OpenAIClientConfig
 from agent_todo_impl.mdscan import collect_markdown_context
 from agent_todo_impl.planning.plan_generator import build_plan_prompt
 from agent_todo_impl.planning.plan_parser import TodoItem, parse_plan_text_to_todos
 from agent_todo_impl.quality.gates import run_quality_gates
 from agent_todo_impl.review.reviewer import Reviewer
 
 
 @dataclass(frozen=True)
 class OrchestratorConfig:
     repo_root: Path
     md_path: Path
     model: str
     max_review_rounds: int = 3
 
 
 @dataclass(frozen=True)
 class OrchestratorResult:
     branch: str
     plan_text: str
     todos: list[TodoItem]
     review_rounds: int
     gates_ok: bool
     gates_output: str
     review: dict
 
     def model_dump(self) -> dict:
         return {
             "branch": self.branch,
             "plan_text": self.plan_text,
             "todos": [t.model_dump() for t in self.todos],
             "review_rounds": self.review_rounds,
             "gates_ok": self.gates_ok,
             "gates_output": self.gates_output,
             "review": self.review,
         }
 
 
 class Orchestrator:
     def __init__(self, config: OrchestratorConfig):
         self._config = config
         self._git = GitManager(config.repo_root)
         self._llm = OpenAIClient(OpenAIClientConfig(model=config.model))
         self._executor = Executor(ExecutorConfig(repo_root=config.repo_root), llm=self._llm)
         self._reviewer = Reviewer(llm=self._llm)
 
     def _snapshot_hint(self) -> str:
         # Lightweight hint; avoid reading entire repo.
         return "python project with pyproject.toml, src/agent_todo_impl, tests/"
 
     def run(self) -> OrchestratorResult:
         self._git.ensure_repo()
         branch = self._git.create_run_branch("agent-todo-run")
 
         docs = collect_markdown_context(self._config.md_path)
         plan_prompt = build_plan_prompt(docs)
         plan_text = self._llm.complete_text(plan_prompt)
         todos = parse_plan_text_to_todos(plan_text)
 
         # Implementation step
         self._executor.execute(todos, repo_snapshot_hint=self._snapshot_hint())
         gates = run_quality_gates(self._config.repo_root)
         self._git.add_all()
         self._git.commit("implement todos")
 
         # Review loop (<=3)
         rounds = 0
         review_dump: dict = {"findings": []}
         while rounds < self._config.max_review_rounds:
             rounds += 1
             diff = self._git.diff()
             review = self._reviewer.review(diff=diff, gates_output=gates.output)
             review_dump = review.model_dump()
             must_fix = [f for f in review.findings if f.level == "must_fix"]
             if gates.ok and not must_fix:
                 break
 
             # Ask LLM to produce edits to address review + gates output.
             fix_todos = [
                 TodoItem(id=f"review-{i+1}", content=f"{f.level}: {f.message}")
                 for i, f in enumerate(review.findings)
             ]
             self._executor.execute(fix_todos, repo_snapshot_hint=self._snapshot_hint())
             gates = run_quality_gates(self._config.repo_root)
             self._git.add_all()
             self._git.commit(f"review fix round {rounds}")
 
         return OrchestratorResult(
             branch=branch,
             plan_text=plan_text,
             todos=todos,
             review_rounds=rounds,
             gates_ok=gates.ok,
             gates_output=gates.output,
             review=review_dump,
         )

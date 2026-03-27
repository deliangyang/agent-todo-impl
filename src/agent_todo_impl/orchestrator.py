from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

from agent_todo_impl.execution.cursor_agent import (
    CursorAgentConfig,
    build_cursor_agent_prompt_for_todo,
    run_cursor_agent,
    todo_run_cwd_for_md_path,
)
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
    executor: str = "internal"  # internal | cursor
    cursor_model: str = "auto"
    cursor_force: bool = True
    cursor_output_format: str = "text"  # text | json | stream-json
    cursor_stream_partial_output: bool = False


@dataclass(frozen=True)
class OrchestratorResult:
    branch: str
    plan_text: str
    todos: list[TodoItem]
    review_rounds: int
    gates_ok: bool
    gates_output: str
    review: dict
    cursor: dict | None = None

    def model_dump(self) -> dict:
        return {
            "branch": self.branch,
            "plan_text": self.plan_text,
            "todos": [t.model_dump() for t in self.todos],
            "review_rounds": self.review_rounds,
            "gates_ok": self.gates_ok,
            "gates_output": self.gates_output,
            "review": self.review,
            "cursor": self.cursor,
        }


class Orchestrator:
    def __init__(self, config: OrchestratorConfig):
        self._config = config
        self._git = GitManager(config.repo_root)
        self._llm = OpenAIClient(OpenAIClientConfig(model=config.model))
        self._executor = Executor(ExecutorConfig(repo_root=config.repo_root), llm=self._llm)
        self._reviewer = Reviewer(llm=self._llm)

    def _snapshot_hint(self) -> str:
        cwd = Path.cwd().resolve()
        markers = [
            "composer.json",
            "pyproject.toml",
            "package.json",
            "go.mod",
            "Cargo.toml",
            "pom.xml",
        ]
        found_markers = [m for m in markers if (cwd / m).exists()]
        top_dirs = [
            p.name
            for p in sorted(cwd.iterdir(), key=lambda x: x.name)
            if p.is_dir() and not p.name.startswith(".")
        ][:8]
        marker_part = ",".join(found_markers) if found_markers else "none"
        dirs_part = ",".join(top_dirs) if top_dirs else "none"
        return f"cwd={cwd}; markers={marker_part}; top_dirs={dirs_part}"

    def _todo_run_cwd(self) -> Path:
        return todo_run_cwd_for_md_path(self._config.md_path)

    def run(self) -> OrchestratorResult:
        has_git = self._git.is_repo()
        branch = "(no-git)"
        if has_git:
            self._git.ensure_repo()
            branch = self._git.create_run_branch("agent-todo-run")

        docs = collect_markdown_context(self._config.md_path)
        plan_prompt = build_plan_prompt(docs)
        plan_run = run_cursor_agent(
            CursorAgentConfig(
                workspace=self._config.repo_root,
                run_cwd=self._todo_run_cwd(),
                model=self._config.cursor_model,
                force=self._config.cursor_force,
                output_format="text",
                stream_partial_output=False,
            ),
            prompt=plan_prompt,
        )
        if plan_run.exit_code != 0:
            raise RuntimeError(
                f"cursor-agent plan generation failed (exit={plan_run.exit_code}): "
                f"{plan_run.stderr or plan_run.stdout}"
            )
        plan_text = plan_run.stdout
        todos = parse_plan_text_to_todos(plan_text)

        cursor_runs: list[dict] = []
        session_id: str | None = None
        cursor_base = CursorAgentConfig(
            workspace=self._config.repo_root,
            run_cwd=self._todo_run_cwd(),
            model=self._config.cursor_model,
            force=self._config.cursor_force,
            output_format="json",
            stream_partial_output=False,
        )
        if self._config.executor == "cursor":
            for todo in todos:
                prompt = build_cursor_agent_prompt_for_todo(
                    todo, repo_snapshot_hint=self._snapshot_hint()
                )
                cfg = replace(cursor_base, resume_session_id=session_id)
                run = run_cursor_agent(cfg, prompt=prompt)
                cursor_runs.append(run.model_dump())
                if run.exit_code != 0:
                    raise RuntimeError(f"cursor-agent failed (exit={run.exit_code})")
                if session_id is None:
                    if not run.session_id:
                        raise RuntimeError(
                            "cursor-agent JSON missing session_id on first todo; "
                            "need --output-format json and a successful run"
                        )
                    session_id = run.session_id
                if has_git and self._git.has_worktree_changes():
                    self._git.add_all()
                    self._git.commit(f"implement todo {todo.id}")
        else:
            self._executor.execute(todos, repo_snapshot_hint=self._snapshot_hint())
            if has_git:
                self._git.add_all()
                self._git.commit("implement todos")
        gates = run_quality_gates(self._config.repo_root)

        rounds = 0
        review_dump: dict = {"findings": []}
        while has_git and rounds < self._config.max_review_rounds:
            rounds += 1
            diff = self._git.diff()
            review = self._reviewer.review(diff=diff, gates_output=gates.output)
            review_dump = review.model_dump()
            must_fix = [f for f in review.findings if f.level == "must_fix"]
            if gates.ok and not must_fix:
                break

            fix_todos = [
                TodoItem(id=f"review-{i+1}", content=f"{f.level}: {f.message}")
                for i, f in enumerate(review.findings)
            ]
            if self._config.executor == "cursor":
                for fix_todo in fix_todos:
                    prompt = build_cursor_agent_prompt_for_todo(
                        fix_todo, repo_snapshot_hint=self._snapshot_hint()
                    )
                    cfg = replace(cursor_base, resume_session_id=session_id)
                    run = run_cursor_agent(cfg, prompt=prompt)
                    cursor_runs.append(run.model_dump())
                    if run.exit_code != 0:
                        raise RuntimeError(f"cursor-agent failed (exit={run.exit_code})")
                    if session_id is None:
                        if not run.session_id:
                            raise RuntimeError(
                                "cursor-agent JSON missing session_id on first review fix"
                            )
                        session_id = run.session_id
                    gates = run_quality_gates(self._config.repo_root)
                    if self._git.has_worktree_changes():
                        self._git.add_all()
                        self._git.commit(f"review fix r{rounds} {fix_todo.id}")
            else:
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
            cursor={"runs": cursor_runs} if cursor_runs else None,
        )

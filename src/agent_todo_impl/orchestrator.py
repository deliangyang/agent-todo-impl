from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

from agent_todo_impl.checkpoint import (
    RunCheckpoint,
    clear_active_checkpoint,
    in_progress_checkpoint_path,
    read_checkpoint,
    register_active_checkpoint,
    session_checkpoint_path,
    write_checkpoint,
)
from agent_todo_impl.execution.cursor_agent import (
    CursorAgentConfig,
    build_cursor_agent_prompt_for_todo,
    run_cursor_agent,
    todo_run_cwd_for_md_path,
)
from agent_todo_impl.execution.external_cli import (
    EXTERNAL_CLI_EXECUTORS,
    ExternalCliConfig,
    build_external_cli_prompt_for_todo,
    run_external_cli,
)
from agent_todo_impl.execution.executor import Executor, ExecutorConfig
from agent_todo_impl.git.git_manager import GitManager, commit_changed_repos
from agent_todo_impl.llm.openai_client import OpenAIClient, OpenAIClientConfig
from agent_todo_impl.mdscan import collect_requirement_context
from agent_todo_impl.planning.plan_generator import build_plan_prompt
from agent_todo_impl.planning.plan_parser import TodoItem, parse_plan_text_to_todos
from agent_todo_impl.quality.gates import run_quality_gates
from agent_todo_impl.review.reviewer import Reviewer


@dataclass(frozen=True)
class OrchestratorConfig:
    repo_root: Path
    md_path: Path | None
    model: str
    enable_review: bool = False
    max_review_rounds: int = 3
    executor: str = "internal"  # internal | cursor | copilot | codex | gemini
    cursor_model: str = "auto"
    cursor_force: bool = True
    cursor_output_format: str = "text"  # text | json | stream-json
    cursor_stream_partial_output: bool = False
    text_snippets: tuple[str, ...] = ()
    image_paths: tuple[Path, ...] = ()
    image_urls: tuple[str, ...] = ()
    # cursor executor: persist under ~/.agent-todo/checkpoint/; resume with same Cursor session_id
    resume_session_id: str | None = None


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
    cli: dict | None = None

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
            "cli": self.cli,
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
        if self._config.md_path is not None:
            return todo_run_cwd_for_md_path(self._config.md_path)
        return Path.cwd().resolve()

    def _use_cursor_state(self) -> bool:
        return self._config.executor == "cursor"

    def _commit_changed_repositories(self, message: str) -> None:
        commit_changed_repos(self._todo_run_cwd(), message)

    def _persist_checkpoint(self, ckpt: RunCheckpoint) -> None:
        if not self._use_cursor_state():
            return
        root = self._config.repo_root.resolve()
        if ckpt.session_id:
            path = session_checkpoint_path(root, ckpt.session_id)
            in_progress_checkpoint_path(root).unlink(missing_ok=True)
        else:
            path = in_progress_checkpoint_path(root)
        write_checkpoint(path, ckpt)
        register_active_checkpoint(path, ckpt.to_json_dict())

    def _validate_checkpoint(self, ckpt: RunCheckpoint) -> None:
        cfg = self._config
        if str(cfg.repo_root.resolve()) != ckpt.repo_root:
            raise RuntimeError(
                "checkpoint repo_root mismatch; use the same project root as the original run"
            )
        md = str(cfg.md_path.resolve()) if cfg.md_path else None
        if md != ckpt.md_path:
            raise RuntimeError(
                "checkpoint md_path mismatch; pass the same --md-path as the original run"
            )
        if list(cfg.text_snippets) != list(ckpt.text_snippets):
            raise RuntimeError("checkpoint text_snippets mismatch")
        if [str(p.resolve()) for p in cfg.image_paths] != list(ckpt.image_paths):
            raise RuntimeError("checkpoint image_paths mismatch")
        if list(cfg.image_urls) != list(ckpt.image_urls):
            raise RuntimeError("checkpoint image_urls mismatch")
        if bool(ckpt.enable_review) != bool(cfg.enable_review):
            raise RuntimeError(
                "checkpoint enable_review mismatch; pass the same --review flag as the original run"
            )
        if not ckpt.todos:
            raise RuntimeError("checkpoint contains no todos")
        if ckpt.implement_next_index < 0 or ckpt.implement_next_index > len(ckpt.todos):
            raise RuntimeError("checkpoint implement_next_index is out of range")

    def run(self) -> OrchestratorResult:
        cfg = self._config
        sid = (cfg.resume_session_id or "").strip()
        if sid and cfg.executor != "cursor":
            raise RuntimeError("--resume SESSION_ID requires --executor cursor")
        if sid and not session_checkpoint_path(cfg.repo_root, sid).is_file():
            expect = session_checkpoint_path(cfg.repo_root, sid)
            raise RuntimeError(
                f"no saved state for session_id {sid!r}; expected {expect} "
                "(~/.agent-todo/checkpoint/sessions/; override with AGENT_TODO_CHECKPOINT_DIR)"
            )

        try:
            return self._run_inner(
                resume_session_id=sid or None,
                use_cursor_state=self._use_cursor_state(),
                has_git_initial=self._git.is_repo(),
            )
        finally:
            clear_active_checkpoint()

    def _run_inner(
        self,
        *,
        resume_session_id: str | None,
        use_cursor_state: bool,
        has_git_initial: bool,
    ) -> OrchestratorResult:
        cfg = self._config
        has_git = has_git_initial
        branch = "(no-git)"
        session_id: str | None = None
        plan_text = ""
        todos: list[TodoItem] = []
        ckpt: RunCheckpoint | None = None

        if use_cursor_state and resume_session_id:
            path = session_checkpoint_path(cfg.repo_root, resume_session_id)
            ckpt = read_checkpoint(path)
            self._validate_checkpoint(ckpt)
            if ckpt.session_id != resume_session_id:
                raise RuntimeError("checkpoint session_id does not match --resume value")
            plan_text = ckpt.plan_text
            todos = [
                TodoItem(id=str(t["id"]), content=str(t["content"]))
                for t in ckpt.todos
            ]
            session_id = ckpt.session_id
            branch = ckpt.run_branch or "(no-git)"
            if has_git:
                self._git.ensure_repo()
                if ckpt.run_branch:
                    self._git.checkout_branch(ckpt.run_branch)
        elif has_git:
            self._git.ensure_repo()
            branch = self._git.create_run_branch("agent-todo-run")

        if not (use_cursor_state and resume_session_id):
            docs = collect_requirement_context(
                md_path=cfg.md_path,
                text_snippets=list(cfg.text_snippets),
                image_paths=list(cfg.image_paths),
                image_urls=list(cfg.image_urls),
            )
            plan_prompt = build_plan_prompt(docs)
            plan_run = run_cursor_agent(
                CursorAgentConfig(
                    workspace=cfg.repo_root,
                    run_cwd=self._todo_run_cwd(),
                    model=cfg.cursor_model,
                    force=cfg.cursor_force,
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
            if use_cursor_state:
                ckpt = RunCheckpoint(
                    repo_root=str(cfg.repo_root.resolve()),
                    md_path=str(cfg.md_path.resolve()) if cfg.md_path else None,
                    text_snippets=list(cfg.text_snippets),
                    image_paths=[str(p.resolve()) for p in cfg.image_paths],
                    image_urls=list(cfg.image_urls),
                    run_branch=branch if branch != "(no-git)" else None,
                    plan_text=plan_text,
                    todos=[t.model_dump() for t in todos],
                    session_id=None,
                    phase="implement",
                    implement_next_index=0,
                    enable_review=cfg.enable_review,
                )
                self._persist_checkpoint(ckpt)

        cursor_runs: list[dict] = []
        cli_runs: list[dict] = []
        cursor_base = CursorAgentConfig(
            workspace=cfg.repo_root,
            run_cwd=self._todo_run_cwd(),
            model=cfg.cursor_model,
            force=cfg.cursor_force,
            output_format="json",
            stream_partial_output=False,
        )
        external_cli_base = ExternalCliConfig(
            workspace=cfg.repo_root,
            run_cwd=self._todo_run_cwd(),
            model=cfg.cursor_model,
        )
        if cfg.executor == "cursor":
            assert ckpt is not None or not use_cursor_state
            for idx, todo in enumerate(todos):
                if use_cursor_state:
                    assert ckpt is not None
                    if idx < ckpt.implement_next_index:
                        continue
                    ckpt.implement_next_index = idx
                    ckpt.session_id = session_id
                    self._persist_checkpoint(ckpt)
                prompt = build_cursor_agent_prompt_for_todo(
                    todo, repo_snapshot_hint=self._snapshot_hint()
                )
                cfg_run = replace(cursor_base, resume_session_id=session_id)
                run = run_cursor_agent(cfg_run, prompt=prompt)
                cursor_runs.append(run.model_dump())
                if run.exit_code != 0:
                    if use_cursor_state and ckpt is not None:
                        self._persist_checkpoint(ckpt)
                    raise RuntimeError(f"cursor-agent failed (exit={run.exit_code})")
                if session_id is None:
                    if not run.session_id:
                        raise RuntimeError(
                            "cursor-agent JSON missing session_id on first todo; "
                            "need --output-format json and a successful run"
                        )
                    session_id = run.session_id
                else:
                    session_id = run.session_id or session_id
                if use_cursor_state and ckpt is not None:
                    ckpt.session_id = session_id
                    ckpt.implement_next_index = idx + 1
                    self._persist_checkpoint(ckpt)
                self._commit_changed_repositories(f"implement todo {todo.id} {todo.content}")
        elif cfg.executor in EXTERNAL_CLI_EXECUTORS:
            for todo in todos:
                prompt = build_external_cli_prompt_for_todo(
                    cfg.executor,
                    todo,
                    repo_snapshot_hint=self._snapshot_hint(),
                )
                run = run_external_cli(cfg.executor, external_cli_base, prompt=prompt)
                cli_runs.append(run.model_dump())
                if run.exit_code != 0:
                    raise RuntimeError(
                        f"{cfg.executor} CLI failed (exit={run.exit_code}): "
                        f"{run.stderr or run.stdout}"
                    )
                self._commit_changed_repositories(f"implement todo {todo.id} {todo.content}")
        else:
            self._executor.execute(todos, repo_snapshot_hint=self._snapshot_hint())
            todo_summary = "; ".join(f"{t.id}:{t.content}" for t in todos)[:180]
            message = f"implement todos {todo_summary}" if todo_summary else "implement todos"
            self._commit_changed_repositories(message)
        gates = run_quality_gates(cfg.repo_root)

        rounds = 0
        review_dump: dict = {"findings": []}
        if cfg.enable_review:
            while has_git and rounds < cfg.max_review_rounds:
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
                if cfg.executor == "cursor":
                    for fix_todo in fix_todos:
                        prompt = build_cursor_agent_prompt_for_todo(
                            fix_todo, repo_snapshot_hint=self._snapshot_hint()
                        )
                        cfg_run = replace(cursor_base, resume_session_id=session_id)
                        run = run_cursor_agent(cfg_run, prompt=prompt)
                        cursor_runs.append(run.model_dump())
                        if run.exit_code != 0:
                            if use_cursor_state and ckpt is not None:
                                ckpt.session_id = session_id
                                self._persist_checkpoint(ckpt)
                            raise RuntimeError(f"cursor-agent failed (exit={run.exit_code})")
                        if session_id is None:
                            if not run.session_id:
                                raise RuntimeError(
                                    "cursor-agent JSON missing session_id on first review fix"
                                )
                            session_id = run.session_id
                        else:
                            session_id = run.session_id or session_id
                        if use_cursor_state and ckpt is not None:
                            ckpt.session_id = session_id
                            self._persist_checkpoint(ckpt)
                        gates = run_quality_gates(cfg.repo_root)
                        self._commit_changed_repositories(
                            f"review fix r{rounds} {fix_todo.id} {fix_todo.content}"
                        )
                elif cfg.executor in EXTERNAL_CLI_EXECUTORS:
                    for fix_todo in fix_todos:
                        prompt = build_external_cli_prompt_for_todo(
                            cfg.executor,
                            fix_todo,
                            repo_snapshot_hint=self._snapshot_hint(),
                        )
                        run = run_external_cli(cfg.executor, external_cli_base, prompt=prompt)
                        cli_runs.append(run.model_dump())
                        if run.exit_code != 0:
                            raise RuntimeError(
                                f"{cfg.executor} CLI failed (exit={run.exit_code}): "
                                f"{run.stderr or run.stdout}"
                            )
                        gates = run_quality_gates(cfg.repo_root)
                        self._commit_changed_repositories(
                            f"review fix r{rounds} {fix_todo.id} {fix_todo.content}"
                        )
                else:
                    self._executor.execute(fix_todos, repo_snapshot_hint=self._snapshot_hint())
                    gates = run_quality_gates(cfg.repo_root)
                    self._commit_changed_repositories(f"review fix round {rounds}")

        if use_cursor_state:
            root = cfg.repo_root.resolve()
            in_progress_checkpoint_path(root).unlink(missing_ok=True)
            if session_id:
                session_checkpoint_path(root, session_id).unlink(missing_ok=True)

        return OrchestratorResult(
            branch=branch,
            plan_text=plan_text,
            todos=todos,
            review_rounds=rounds,
            gates_ok=gates.ok,
            gates_output=gates.output,
            review=review_dump,
            cursor={"runs": cursor_runs} if cursor_runs else None,
            cli=(
                {"executor": cfg.executor, "runs": cli_runs}
                if cli_runs
                else None
            ),
        )

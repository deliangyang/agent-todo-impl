from pathlib import Path

from agent_todo_impl.execution.external_cli import (
    ExternalCliConfig,
    build_codex_command,
    build_copilot_command,
    build_external_cli_command,
    build_external_cli_prompt_for_todo,
    build_gemini_command,
)
from agent_todo_impl.planning.plan_parser import TodoItem


def test_build_copilot_command_includes_prompt_and_model(tmp_path: Path):
    cfg_workspace = tmp_path / "repo"
    cfg_workspace.mkdir()
    cfg = ExternalCliConfig(workspace=cfg_workspace, model="gpt-5.2")
    cmd = build_copilot_command(cfg, prompt="do the thing")
    assert cmd[0] == "copilot"
    assert "--add-dir" in cmd
    assert "-p" in cmd
    assert "do the thing" in cmd
    assert "--model" in cmd
    idx = cmd.index("--model")
    assert cmd[idx + 1] == "gpt-5.2"
    assert "--no-ask-user" in cmd
    assert "--allow-all" in cmd


def test_build_copilot_command_omits_model_when_auto(tmp_path: Path):
    cfg = ExternalCliConfig(workspace=tmp_path, model="auto")
    cmd = build_copilot_command(cfg, prompt="x")
    assert "--model" not in cmd


def test_build_codex_command_workspace_and_prompt(tmp_path: Path):
    ws = tmp_path.resolve()
    cfg = ExternalCliConfig(workspace=ws, model="auto")
    cmd = build_codex_command(cfg, prompt="task")
    assert cmd[:4] == ["codex", "exec", "-C", str(ws)]
    assert "--sandbox" in cmd
    assert "workspace-write" in cmd
    assert cmd[-1] == "task"


def test_build_gemini_command_includes_include_directories(tmp_path: Path):
    ws = tmp_path.resolve()
    cfg = ExternalCliConfig(workspace=ws)
    cmd = build_gemini_command(cfg, prompt="hi")
    assert cmd[0] == "gemini"
    assert "-p" in cmd
    assert "--include-directories" in cmd
    assert str(ws) in cmd
    assert "--yolo" in cmd


def test_build_external_cli_command_dispatch():
    cfg = ExternalCliConfig(workspace=Path("/tmp"))
    assert build_external_cli_command("copilot", cfg, prompt="a")[0] == "copilot"
    assert build_external_cli_command("codex", cfg, prompt="a")[0] == "codex"
    assert build_external_cli_command("gemini", cfg, prompt="a")[0] == "gemini"


def test_build_external_cli_prompt_mentions_executor_role():
    todo = TodoItem(id="1", content="fix bug")
    p = build_external_cli_prompt_for_todo("gemini", todo, repo_snapshot_hint="hint")
    assert "Gemini" in p or "Gemini CLI" in p
    assert "fix bug" in p
    assert "hint" in p

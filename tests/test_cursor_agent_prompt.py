from agent_todo_impl.execution.cursor_agent import (
    CursorAgentConfig,
    build_cursor_agent_command,
    build_cursor_agent_prompt,
)
from agent_todo_impl.planning.plan_parser import TodoItem


def test_build_cursor_agent_prompt_contains_todos():
    todos = [TodoItem(id="a-b", content="实现功能 X"), TodoItem(id="c-d", content="补测试")]
    prompt = build_cursor_agent_prompt(todos, repo_snapshot_hint="repo-hint")
    assert "repo-hint" in prompt
    assert "- a-b: 实现功能 X" in prompt
    assert "- c-d: 补测试" in prompt


def test_build_cursor_agent_command_includes_required_flags(tmp_path):
    cmd = build_cursor_agent_command(CursorAgentConfig(workspace=tmp_path), prompt="do it")
    s = " ".join(cmd)
    assert "cursor-agent" in cmd[0]
    assert "--print" in cmd
    assert "--trust" in cmd
    assert "--force" in cmd
    assert "--workspace" in cmd
    assert str(tmp_path) in s
    assert "--model" in cmd
    assert "auto" in cmd
    assert "--output-format" in cmd
    assert "text" in cmd
    assert "--stream-partial-output" not in cmd

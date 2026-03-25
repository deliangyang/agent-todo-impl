from agent_todo_impl.execution.cursor_agent import build_cursor_agent_prompt
from agent_todo_impl.planning.plan_parser import TodoItem


def test_build_cursor_agent_prompt_contains_todos():
    todos = [TodoItem(id="a-b", content="实现功能 X"), TodoItem(id="c-d", content="补测试")]
    prompt = build_cursor_agent_prompt(todos, repo_snapshot_hint="repo-hint")
    assert "repo-hint" in prompt
    assert "- a-b: 实现功能 X" in prompt
    assert "- c-d: 补测试" in prompt

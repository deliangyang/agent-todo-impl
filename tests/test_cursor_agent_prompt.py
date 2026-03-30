from agent_todo_impl.execution.cursor_agent import (
    CursorAgentConfig,
    build_cursor_agent_command,
    build_cursor_agent_prompt,
    build_cursor_agent_prompt_for_todo,
    parse_cursor_agent_json_stdout,
)
from agent_todo_impl.planning.plan_parser import TodoItem


def test_build_cursor_agent_prompt_for_todo_single_item():
    t = TodoItem(id="a-b", content="只做这一件")
    prompt = build_cursor_agent_prompt_for_todo(t, repo_snapshot_hint="hint")
    assert "hint" in prompt
    assert "仅完成这一条" in prompt
    assert "- a-b: 只做这一件" in prompt


def test_parse_cursor_agent_json_stdout():
    raw = '{"type":"result","session_id":"sid-uuid","result":"done"}'
    sid, res = parse_cursor_agent_json_stdout(raw)
    assert sid == "sid-uuid"
    assert res == "done"


def test_build_cursor_agent_command_includes_resume(tmp_path):
    cfg = CursorAgentConfig(
        workspace=tmp_path,
        resume_session_id="abc-session",
        output_format="json",
    )
    cmd = build_cursor_agent_command(cfg, prompt="next")
    assert "--resume" in cmd
    idx = cmd.index("--resume")
    assert cmd[idx + 1] == "abc-session"


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
    assert "stream-json" in cmd
    assert "--stream-partial-output" in cmd

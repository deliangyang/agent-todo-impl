from pathlib import Path

from agent_todo_impl.execution.cursor_agent import todo_run_cwd_for_md_path


def test_todo_run_cwd_uses_command_cwd_for_file_input(tmp_path: Path, monkeypatch):
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    doc_dir = tmp_path / "nested" / "vendor" / "laravel" / "telescope"
    doc_dir.mkdir(parents=True)
    md_file = doc_dir / "todo.md"
    md_file.write_text("# todo\n", encoding="utf-8")

    monkeypatch.chdir(run_dir)

    assert todo_run_cwd_for_md_path(md_file) == run_dir.resolve()


def test_todo_run_cwd_uses_command_cwd_for_dir_input(tmp_path: Path, monkeypatch):
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    md_dir = tmp_path / "docs"
    md_dir.mkdir()

    monkeypatch.chdir(run_dir)

    assert todo_run_cwd_for_md_path(md_dir) == run_dir.resolve()

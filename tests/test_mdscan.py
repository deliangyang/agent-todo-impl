from pathlib import Path

from agent_todo_impl.mdscan import collect_markdown_context, format_markdown_context_for_prompt


def test_collect_markdown_context_file(tmp_path: Path):
    md = tmp_path / "a.md"
    md.write_text("# hi\n", encoding="utf-8")
    docs = collect_markdown_context(md)
    assert len(docs) == 1
    assert docs[0].path == md


def test_collect_markdown_context_dir(tmp_path: Path):
    (tmp_path / "x").mkdir()
    (tmp_path / "x" / "a.md").write_text("A", encoding="utf-8")
    (tmp_path / "b.md").write_text("B", encoding="utf-8")
    docs = collect_markdown_context(tmp_path)
    assert [d.path.name for d in docs] == ["b.md", "a.md"]
    s = format_markdown_context_for_prompt(docs)
    assert "=== FILE:" in s

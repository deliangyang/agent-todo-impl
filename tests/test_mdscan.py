from pathlib import Path

import pytest
from agent_todo_impl.mdscan import (
    EmptyRequirementContentError,
    collect_markdown_context,
    collect_requirement_context,
    format_markdown_context_for_prompt,
)


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


def test_collect_requirement_context_text_and_url():
    docs = collect_requirement_context(
        md_path=None,
        text_snippets=["一行需求"],
        image_urls=["https://example.com/x.png"],
    )
    assert len(docs) == 2
    s = format_markdown_context_for_prompt(docs)
    assert "INLINE TEXT" in s
    assert "一行需求" in s
    assert "https://example.com/x.png" in s


def test_collect_requirement_context_image_file(tmp_path: Path):
    img = tmp_path / "s.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n")
    docs = collect_requirement_context(md_path=img)
    assert len(docs) == 1
    assert "IMAGE FILE" in format_markdown_context_for_prompt(docs)


def test_collect_requirement_context_empty_dir_fails(tmp_path: Path):
    empty = tmp_path / "e"
    empty.mkdir()
    with pytest.raises(EmptyRequirementContentError):
        collect_requirement_context(md_path=empty)

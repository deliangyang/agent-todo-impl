from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MarkdownDocument:
    path: Path
    text: str


def _iter_md_files(md_path: Path) -> list[Path]:
    if md_path.is_file():
        return [md_path]
    if not md_path.is_dir():
        raise ValueError(f"md_path must be a file or directory: {md_path}")
    return sorted([p for p in md_path.rglob("*.md") if p.is_file()])


def collect_markdown_context(
    md_path: Path, *, max_chars_per_file: int = 50_000
) -> list[MarkdownDocument]:
    docs: list[MarkdownDocument] = []
    for p in _iter_md_files(md_path):
        text = p.read_text(encoding="utf-8", errors="replace")
        if len(text) > max_chars_per_file:
            text = text[:max_chars_per_file] + "\n\n[TRUNCATED]\n"
        docs.append(MarkdownDocument(path=p, text=text))
    return docs


def format_markdown_context_for_prompt(docs: list[MarkdownDocument]) -> str:
    parts: list[str] = []
    for d in docs:
        parts.append(f"=== FILE: {d.path.as_posix()} ===\n{d.text}")
    return "\n\n".join(parts).strip() + "\n"

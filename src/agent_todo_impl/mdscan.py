from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


class EmptyRequirementContentError(ValueError):
    """Raised when the given sources yield no documents (e.g. empty directory)."""


_MD_SUFFIXES = {".md", ".markdown"}
_IMAGE_SUFFIXES = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".bmp",
    ".svg",
    ".ico",
}


@dataclass(frozen=True)
class MarkdownDocument:
    """A chunk of requirement context (markdown file, inline text, or image reference)."""

    path: Path
    text: str
    # If set, used as the section header instead of "FILE: path"
    section_title: str | None = None


def _is_image_path(path: Path) -> bool:
    return path.suffix.lower() in _IMAGE_SUFFIXES


def _read_md_file(p: Path, *, max_chars: int) -> str:
    text = p.read_text(encoding="utf-8", errors="replace")
    if len(text) > max_chars:
        text = text[:max_chars] + "\n\n[TRUNCATED]\n"
    return text


def _image_file_block(path: Path) -> str:
    resolved = path.resolve()
    size = resolved.stat().st_size
    uri = resolved.as_uri()
    return (
        f"本地图片文件（绝对路径）: {resolved}\n"
        f"文件大小: {size} bytes\n"
        f"Markdown 引用: ![image]({uri})"
    )


def _image_url_block(url: str) -> str:
    u = url.strip()
    return f"图片链接（URL）:\n{u}\n\nMarkdown: ![image]({u})"


def _iter_md_files_under_dir(md_path: Path) -> list[Path]:
    return sorted([p for p in md_path.rglob("*.md") if p.is_file()])


def _append_md_path_file(path: Path, docs: list[MarkdownDocument], *, max_chars: int) -> None:
    if path.suffix.lower() in _MD_SUFFIXES:
        docs.append(
            MarkdownDocument(path=path, text=_read_md_file(path, max_chars=max_chars))
        )
        return
    if _is_image_path(path):
        docs.append(
            MarkdownDocument(
                path=path,
                text=_image_file_block(path),
                section_title=f"IMAGE FILE: {path.name}",
            )
        )
        return
    # Other single files: best-effort as UTF-8 text (e.g. .txt)
    docs.append(
        MarkdownDocument(
            path=path,
            text=_read_md_file(path, max_chars=max_chars),
            section_title=f"FILE: {path.name}",
        )
    )


def collect_requirement_context(
    *,
    md_path: Path | None = None,
    text_snippets: list[str] | None = None,
    image_paths: list[Path] | None = None,
    image_urls: list[str] | None = None,
    max_chars_per_file: int = 50_000,
) -> list[MarkdownDocument]:
    """
    Build requirement documents in fixed order:
    1) md_path: directory (*.md), or a single file (.md / image / other as text)
    2) each --text snippet
    3) each --image path
    4) each --image-url
    """
    docs: list[MarkdownDocument] = []
    texts = [t for t in (text_snippets or []) if t.strip()]
    images = list(image_paths or [])
    urls = [u for u in (image_urls or []) if u.strip()]

    if md_path is not None:
        p = md_path
        if p.is_file():
            _append_md_path_file(p, docs, max_chars=max_chars_per_file)
        elif p.is_dir():
            for f in _iter_md_files_under_dir(p):
                docs.append(
                    MarkdownDocument(
                        path=f,
                        text=_read_md_file(f, max_chars=max_chars_per_file),
                    )
                )
        else:
            raise ValueError(f"md_path is not a file or directory: {md_path}")

    for i, t in enumerate(texts, start=1):
        label = Path(f"_inline_text_{i}.txt")
        docs.append(
            MarkdownDocument(
                path=label,
                text=t.strip(),
                section_title=f"INLINE TEXT #{i}",
            )
        )

    for j, img in enumerate(images, start=1):
        if not img.is_file():
            raise ValueError(f"image path is not a file: {img}")
        docs.append(
            MarkdownDocument(
                path=img,
                text=_image_file_block(img),
                section_title=f"IMAGE FILE #{j}: {img.name}",
            )
        )

    for k, u in enumerate(urls, start=1):
        docs.append(
            MarkdownDocument(
                path=Path(f"_image_url_{k}.txt"),
                text=_image_url_block(u),
                section_title=f"IMAGE URL #{k}",
            )
        )

    if not docs:
        raise EmptyRequirementContentError(
            "未能从给定来源收集到任何需求内容（例如目录下没有 .md）"
        )
    return docs


def collect_markdown_context(
    md_path: Path, *, max_chars_per_file: int = 50_000
) -> list[MarkdownDocument]:
    """Backward-compatible: only md_path (file, directory, or image file)."""
    return collect_requirement_context(
        md_path=md_path,
        max_chars_per_file=max_chars_per_file,
    )


def format_markdown_context_for_prompt(docs: list[MarkdownDocument]) -> str:
    parts: list[str] = []
    for d in docs:
        if d.section_title:
            header = f"=== {d.section_title} ==="
        else:
            header = f"=== FILE: {d.path.as_posix()} ==="
        parts.append(f"{header}\n{d.text}")
    return "\n\n".join(parts).strip() + "\n"

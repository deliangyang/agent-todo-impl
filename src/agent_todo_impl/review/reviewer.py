from __future__ import annotations

import json
from dataclasses import dataclass

from agent_todo_impl.llm.base import LLMClient


@dataclass(frozen=True)
class ReviewFinding:
    level: str  # must_fix | should_fix | nit
    message: str


@dataclass(frozen=True)
class ReviewResult:
    findings: list[ReviewFinding]
    raw_text: str

    def model_dump(self) -> dict:
        return {
            "findings": [{"level": f.level, "message": f.message} for f in self.findings],
            "raw_text": self.raw_text,
        }


def build_review_prompt(*, diff: str, gates_output: str) -> str:
    return (
        "你是 code review agent。请针对下面 git diff 与质量门禁输出给出改进建议。\n"
        "输出严格 JSON，不要额外文字：\n"
        "{\n"
        '  "findings": [\n'
        '    {"level": "must_fix|should_fix|nit", "message": "..."}\n'
        "  ]\n"
        "}\n\n"
        "git diff:\n"
        f"{diff}\n\n"
        "quality gates output:\n"
        f"{gates_output}\n"
    )


class Reviewer:
    def __init__(self, llm: LLMClient):
        self._llm = llm

    def review(self, *, diff: str, gates_output: str) -> ReviewResult:
        prompt = build_review_prompt(diff=diff, gates_output=gates_output)
        text = self._llm.complete_text(prompt).strip()

        if not text.startswith("{"):
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                text = text[start : end + 1]

        findings: list[ReviewFinding] = []
        try:
            data = json.loads(text)
            for f in data.get("findings", []):
                level = str(f.get("level", "")).strip()
                msg = str(f.get("message", "")).strip()
                if level and msg:
                    findings.append(ReviewFinding(level=level, message=msg))
        except json.JSONDecodeError:
            pass

        return ReviewResult(findings=findings, raw_text=text)

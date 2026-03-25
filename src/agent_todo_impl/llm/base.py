from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LLMResponse:
    text: str


class LLMClient:
    def complete_text(self, prompt: str) -> str:  # pragma: no cover (interface)
        raise NotImplementedError

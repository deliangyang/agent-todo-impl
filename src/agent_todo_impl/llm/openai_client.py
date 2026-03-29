from __future__ import annotations

import os
import sys
from dataclasses import dataclass

from openai import OpenAI

from agent_todo_impl.llm.base import LLMClient


@dataclass(frozen=True)
class OpenAIClientConfig:
    model: str
    timeout_s: float = 60.0
    max_output_tokens: int = 1500


class OpenAIClient(LLMClient):
    def __init__(self, config: OpenAIClientConfig):
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required")
        self._client = OpenAI(api_key=api_key, timeout=config.timeout_s)
        self._config = config

    def complete_text(self, prompt: str) -> str:
        """Stream agent output to terminal while accumulating final text.

        The terminal view keeps a rolling window of the last 10 lines in a dim
        gray color, similar风格 to cursor-agent CLI。
        """
        stream = self._client.responses.stream(
            model=self._config.model,
            input=prompt,
            max_output_tokens=self._config.max_output_tokens,
        )
        parts: list[str] = []
        buffer = ""
        lines: list[str] = []
        printed_lines = 0
        max_lines = 10

        def _render_window() -> None:
            nonlocal printed_lines
            if printed_lines:
                sys.stdout.write(f"\x1b[{printed_lines}F")
                sys.stdout.write("\x1b[J")
            printed_lines = len(lines)
            gray_prefix = "\x1b[90m"
            reset = "\x1b[0m"
            for line in lines:
                sys.stdout.write(f"{gray_prefix}{line}{reset}\n")
            sys.stdout.flush()

        with stream as s:
            for event in s:
                delta = getattr(event, "output_text_delta", None)
                if not delta:
                    continue
                parts.append(delta)
                buffer += delta
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    lines.append(line)
                    if len(lines) > max_lines:
                        lines = lines[-max_lines:]
                _render_window()

        if buffer:
            lines.append(buffer)
            if len(lines) > max_lines:
                lines = lines[-max_lines:]
            _render_window()

        return "".join(parts)

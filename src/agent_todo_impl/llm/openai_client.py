 from __future__ import annotations
 
 import os
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
         resp = self._client.responses.create(
             model=self._config.model,
             input=prompt,
             max_output_tokens=self._config.max_output_tokens,
         )
         return resp.output_text

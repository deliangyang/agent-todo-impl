 from __future__ import annotations
 
 import json
 import re
 from dataclasses import dataclass
 
 
 @dataclass(frozen=True)
 class TodoItem:
     id: str
     content: str
 
     def model_dump(self) -> dict[str, str]:
         return {"id": self.id, "content": self.content}
 
 
 _JSON_BLOCK_RE = re.compile(r"\{[\s\S]*\}\s*$")
 
 
 def _coerce_id(raw: str, *, fallback: str) -> str:
     s = re.sub(r"[^a-zA-Z0-9-]+", "-", raw.strip()).strip("-").lower()
     return s or fallback
 
 
 def parse_plan_text_to_todos(plan_text: str) -> list[TodoItem]:
     """
     Accept either:
     - A JSON tail: {"todos":[{"id":"..","content":".."}]}
     - Or plain /plan lines, e.g. "- xxx" / "1. xxx"
     """
     plan_text = (plan_text or "").strip()
     if not plan_text:
         return []
 
     m = _JSON_BLOCK_RE.search(plan_text)
     if m:
         try:
             data = json.loads(m.group(0))
             todos = data.get("todos", [])
             out: list[TodoItem] = []
             for i, t in enumerate(todos):
                 content = str(t.get("content", "")).strip()
                 tid = _coerce_id(str(t.get("id", "")).strip(), fallback=f"todo-{i+1}")
                 if content:
                     out.append(TodoItem(id=tid, content=content))
             if out:
                 return out
         except json.JSONDecodeError:
             pass
 
     lines = [ln.strip() for ln in plan_text.splitlines() if ln.strip()]
     out: list[TodoItem] = []
     for i, ln in enumerate(lines):
         if ln.startswith("/plan"):
             continue
         ln = re.sub(r"^[-*]\s+", "", ln)
         ln = re.sub(r"^\d+[.)]\s+", "", ln)
         if not ln:
             continue
         out.append(TodoItem(id=f"todo-{i+1}", content=ln))
     return out

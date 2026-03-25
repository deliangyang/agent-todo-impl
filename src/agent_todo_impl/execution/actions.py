 from __future__ import annotations
 
 import json
 from dataclasses import dataclass
 from pathlib import Path
 
 
 @dataclass(frozen=True)
 class FileEditAction:
     path: str
     content: str
 
 
 @dataclass(frozen=True)
 class ExecutionPlan:
     edits: list[FileEditAction]
 
 
 def parse_execution_json(text: str) -> ExecutionPlan:
     """
     Expected JSON shape:
     {
       "edits": [{"path": "src/x.py", "content": "...full file content..."}]
     }
     """
     data = json.loads(text)
     edits_raw = data.get("edits", [])
     edits: list[FileEditAction] = []
     for e in edits_raw:
         p = str(e.get("path", "")).strip()
         c = str(e.get("content", ""))
         if p:
             edits.append(FileEditAction(path=p, content=c))
     return ExecutionPlan(edits=edits)
 
 
 def apply_execution_plan(repo_root: Path, plan: ExecutionPlan) -> list[Path]:
     changed: list[Path] = []
     for e in plan.edits:
         p = (repo_root / e.path).resolve()
         if repo_root.resolve() not in p.parents and p != repo_root.resolve():
             raise RuntimeError(f"Refuse to write outside repo_root: {p}")
         p.parent.mkdir(parents=True, exist_ok=True)
         p.write_text(e.content, encoding="utf-8")
         changed.append(p)
     return changed

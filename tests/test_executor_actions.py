 import json
 from pathlib import Path
 
 from agent_todo_impl.execution.actions import apply_execution_plan, parse_execution_json
 
 
 def test_apply_execution_plan_writes_files(tmp_path: Path):
     data = {"edits": [{"path": "src/x.txt", "content": "hello"}]}
     plan = parse_execution_json(json.dumps(data))
     changed = apply_execution_plan(tmp_path, plan)
     assert (tmp_path / "src" / "x.txt").read_text(encoding="utf-8") == "hello"
     assert changed

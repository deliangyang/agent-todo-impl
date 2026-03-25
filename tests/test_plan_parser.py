from agent_todo_impl.planning.plan_parser import parse_plan_text_to_todos


def test_parse_plan_json_tail():
    text = '/plan\n- x\n\n{"todos":[{"id":"a-b","content":"Do it"}]}'
    todos = parse_plan_text_to_todos(text)
    assert [(t.id, t.content) for t in todos] == [("a-b", "Do it")]


def test_parse_plan_lines():
    text = "/plan\n1. 第一件事\n- 第二件事"
    todos = parse_plan_text_to_todos(text)
    assert len(todos) == 2
    assert todos[0].content == "第一件事"


def test_parse_plan_json_fenced_code_block():
    text = """/plan
- something

```json
{"todos":[{"id":"a-b","content":"Do it"}]}
```
"""
    todos = parse_plan_text_to_todos(text)
    assert [(t.id, t.content) for t in todos] == [("a-b", "Do it")]

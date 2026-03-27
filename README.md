 ## agent-todo-impl
 
 一个用 Python 实现的“任务编排 agent”CLI：读取 `*.md` → 让 AI 生成 `/plan` todo → 自动实现并跑质量门禁 → code review agent 最多 3 轮修正 → 每次 run 都在独立分支自动提交（仅本地）。
 
 ### 安装
 
 ```bash
 python -m venv .venv
 source .venv/bin/activate
 pip install -e ".[dev]"
 ```
 
 ### 环境变量
 
 - `cursor-agent`：需在 PATH 中（`plan` 与 `run` 中的 plan 步骤均通过其生成 `/plan`）
 - `CURSOR_API_KEY`：可选，传给 `cursor-agent --api-key`（亦可由 cursor-agent 自行从环境读取）
 - `OPENAI_API_KEY`：`agent-todo run` 在 **internal 执行器** 或 **code review** 时必填（与 OpenAI API 通信）
 - `AGENT_TODO_MODEL`：可选，默认 `gpt-4.1-mini`（仅影响上述 OpenAI 调用，不影响 plan）
 
 ### 使用
 
 - 生成 plan（仅输出到 stdout，不改代码）
 
 ```bash
 agent-todo plan --md-path docs/
 ```
 
- 生成 plan，并输出可交付给 Cursor CLI agent 的执行载荷（prompt/command）

```bash
agent-todo plan --md-path docs/ --emit-cursor
```

 - 执行完整闭环（建分支、改代码、review 循环、自动提交）
 
 ```bash
 agent-todo run --md-path docs/
 ```
 
- 用 Cursor CLI agent 执行 todo（实现交给 `cursor-agent`，其余闭环不变）

```bash
agent-todo run --md-path docs/ --executor cursor
```

 ### 约束
 - 不做 `git merge`。\n+ - 不做任何 `git push`，也不会使用 `push --force`。

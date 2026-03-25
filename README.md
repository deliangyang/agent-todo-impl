 ## agent-todo-impl
 
 一个用 Python 实现的“任务编排 agent”CLI：读取 `*.md` → 让 AI 生成 `/plan` todo → 自动实现并跑质量门禁 → code review agent 最多 3 轮修正 → 每次 run 都在独立分支自动提交（仅本地）。
 
 ### 安装
 
 ```bash
 python -m venv .venv
 source .venv/bin/activate
 pip install -e ".[dev]"
 ```
 
 ### 环境变量
 
 - `OPENAI_API_KEY`: 必填
 - `AGENT_TODO_MODEL`: 可选，默认 `gpt-4.1-mini`
 
 ### 使用
 
 - 生成 plan（仅输出到 stdout，不改代码）
 
 ```bash
 agent-todo plan --md-path docs/
 ```
 
 - 执行完整闭环（建分支、改代码、review 循环、自动提交）
 
 ```bash
 agent-todo run --md-path docs/
 ```
 
 ### 约束
 - 不做 `git merge`。\n+ - 不做任何 `git push`，也不会使用 `push --force`。

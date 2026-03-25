 from __future__ import annotations
 
 from agent_todo_impl.mdscan import MarkdownDocument, format_markdown_context_for_prompt
 
 
 def build_plan_prompt(docs: list[MarkdownDocument]) -> str:
     md = format_markdown_context_for_prompt(docs)
     return (
         "你是一个资深软件工程师。请基于下面的 Markdown 需求文档生成一个 /plan。\n"
         "要求：\n"
         "1) 先给出 /plan（每行一个 TODO，尽量可执行）\n"
         "2) 再输出一个 JSON：{\"todos\": [{\"id\": \"...\", \"content\": \"...\"}]}\n"
         "3) id 用短横线风格，content 用中文。\n\n"
         "文档如下：\n\n"
         f"{md}\n"
     )

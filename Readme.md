# Demonstration: Building an AI Agent


## Prompt example

- 使用 langgraph 构建文献检索 Agent。
- 明确区分“Node（流程控制）”与“Tool（功能实现）”。

Node 设计（必须对应 langgraph 节点）

1. Node1：classify_query
输入：用户问题
判断是否为“文献检索问题”
否：直接输出回答并结束（END）
是：进入 Node2
1. Node2：search_pubmed
调用 tool1
输出：检索词、文献数量、top2 文献信息
1. Node3：translate_top2
调用 tool2
将 top2 的 title/abstract 翻译为中文
1. Node4：build_report
调用 tool3 生成 HTML
返回报告路径
1. Node5：save_log
追加保存到 bak.jsonl
输出最终汇总信息（检索词、文献数、报告路径）

Tool 设计（功能实现，供节点调用）

- tool1：生成 PubMed 检索词并用 e-utilities 检索
    - 返回条目数量 + top2 title/abstract
    - 若数量 < 5，改进检索词并重试，最多 3 次
- tool2：LLM 翻译为中文（title/abstract）
- tool3：生成 HTML 报告，展示检索词 + 中英文 title/abstract，保存到 report/

其他硬性要求

- 使用 TypedDict 传递与保存所有信息
- 结果追加写入 bak.jsonl
- LLM 调用方式固定如下：
```python
import os
from langchain_openai import ChatOpenAI

deepseek_chat = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com",
    temperature=0,
)
response = deepseek_chat.invoke("Hello, how are you?")
print(response.content)
```

- 新建 notebook：
    - 导入 AIAgent
    - 示例输入：AML treatment
    - 包含绘图代码：
```python
from paper_query.ai_agent import AIAgent
from IPython.display import Image, display

agent = AIAgent()
display(Image(agent.get_graph().draw_mermaid_png()))
```

交付物
- paper_query/ 下的 Agent 脚本
- 新 notebook（含测试与 graph 绘图）
- report/ 下的 HTML 报告
- 追加的 bak.jsonl
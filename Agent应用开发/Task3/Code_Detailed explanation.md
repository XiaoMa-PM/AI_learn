# Code_Detailed explanation
---
拆解代码：
[agent](Example_Code/agent.py)：Agent的“大脑”，进行封装与调用。
[llm](Example_Code/llm.py)：Agent的“手臂”，工具是如何被定义和供Agent使用的。
[tool](Example_Code/tool.py)：ReAct模式的“心脏”，Agent是如何协调“大脑”和“手臂”完成任务的。
安装pip
```txt
openai
requests
json5
```

## llm.py
---
提供Agent的思考和推理引擎

### Motivation
Agent的智能来源于LLM，需要构建一种标准化方式去调用不同的LLM，向其发出prompt，接收其响应。
该代码为了实现封装与LLM通信的复杂细节，让主程序`agent.py`可以用一种简单、统一的方式使用LLM的能力，而不用更新底层API具体实现。

### Definition
`llm.py=BaseModel+Siliconflow`
由一个基础模型类与具体的实现类组成。

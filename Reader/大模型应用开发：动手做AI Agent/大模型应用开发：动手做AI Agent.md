# 大模型应用开发：动手做AI Agent
---
🧑‍🔧作者：黄佳
异步图书系列
版本：第一版本
出版日期：2024年5月

## 前言
---
2023年GAI&AIGC&LLM元年
![](inbox/Pasted%20image%2020250930162208.png)
人们对AI的期望转变：从生成工作需求-->复杂任务的关键纽带时代

ZhenFund（真格基金）提出GAI的5个层级

|层级|AI 应用|描述|示例|
|---|---|---|---|
|L1|Tool（工具）|人类完成所有工作，没有任何明显的 AI 辅助|Excel、Photoshop、MATLAB 和 AutoCAD 等绝大多数应用|
|L2|Chatbot（聊天机器人）|人类直接完成绝大部分工作。人类向 AI 询问，了解信息。AI 提供信息和建议，但不直接处理工作|初代 ChatGPT|
|L3|Copilot（协同）|人类和 AI 共同工作，工作量相当。AI 根据人类要求完成工作初稿，人类进行后期校正、修改和调整，并最终确认|GitHub Copilot、Microsoft Copilot|
|L4|Agent|AI 完成绝大部分工作，人类负责设定目标、提供资源和监督结果，以及最终决策。AI 进行任务拆分、工具选择、进度控制，实现目标后自主结束工作|AutoGPT、BabyAGI、MetaGPT|
|L5|Intelligence（智能）|完全无须人类监督，AI 自主拆解目标、寻找资源，选择并使用工具，完成全部工作，人类只须给出初始目标|冯・诺伊曼机器人或者…… 人？|
其中的L3-->L4是进入了一个自主决策的阶段，Agent成为了驱动力。
Agent：
- 不仅仅是内容生成
- 整合了LLM、Data、Tools
- 跨越单纯内容生成界限，涉略决策制定、行动实施等
- 解读复杂指令、规划策略、拆解任务、执行实现目标具体步骤
- 具备自主性、适应性

未来的Agent不止于是提供建议、输出文本工作，而是涉略到使用工具进行生活服务项工作：订酒店、航班、餐厅等

构建Agent的基石：
- AIGC模型
- LLM
- Agent开发框架and工具
- 丰富的数据资源
目前缺乏的是整合经验与技术
>此处理解为是业务的理解力与技术的整合力

Agent的探索需要深入探讨一下几个**关键问题**：
- Agent如何在各行各业提升效率以及创作机会和更多可能性？
- 在众多的Agent框架中，如何选择适合自己的需求框架？
- 在解决现实世界的问题时，如何实施Agent最有效？
- 自主Agent如何改变我们对人工智能驱动的任务管理的认知和实践？

本书旨在于从技术和工具层面阐述Agent设计框架、功能和方法，具体涉及及如下技术或工具
- OpenAI Assistans API：用于调用包含GPT-4模型等
- LangChain：开源框架，简化构建基于语言的人工智能应用过程，包含ReAct框架的封装和实现
- LlamaIndex：开源框架，用于帮助管理和检索非结构化数据，利用大模型的能力和Agent框架提高文本检索的准确性、效率和智能程度。

此外，1还通过7个实战案例去学习前沿的Agent实现技术：
- Agent 1：自动化办公的实现
  通过Assistans API和DALL·E3模型制作PPT
- Agent 2：多功能选择引擎
  通过Function Calling调用函数
- Agent 3：推理与行动的协同
  通过LangChain中的ReAct框架实现自动定价
- Agent 4：计划和执行的解耦
  通过LangChain中的Play-and-Execute实现智能调度库存
- Agent 5：知识的提取与整合
  通过Llamalndex实现检索增强生成
- Agent 6：Github的网红聚落
  AutoGPT、BabyAGI、CAMEL
- Agent 7：多个Agent框架
  AutoGen&Meta GPT

博学--海量数据
审问--有效提示词工程
慎思--配置CoT、ToT、ReAct等思维框架
明辨--指令微调大模型的规范
笃行--Tool Calls&Function Calling等技术

![](inbox/Pasted%20image%2020250930172054.png)
## 目录
---

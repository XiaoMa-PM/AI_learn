# Agent应用开发与落地全景
---
毕昇 产品总监专注于企业大模型应用开发 鲁力
开源社区Datawhale成员
和君商学十三届毕业生
北京大学计算机技术硕士
公众号芥子观徐弥
## 智能体（agent）定义
---
### 引子
思考问题：机器学习算法的定义是什么？
线性回归属于机器学习算法吗？

- 最小二乘法用于线性回归思想
- 计算机诞生比线性回归晚
世界上本没有定义，共识的声音多了，从而出现定义

现状：从学术界到产业界，大模型领域仍然处于快速发展的阶段，还没有明确的具体共识。
任何一个非单次大模型调用的系统，都有可能被其开发者成为“Agent”

### 了解不同定义角度
OpenAI的AGI五级分类：
  

| Level | Description |
| :---- | :---------- |
| 1 | Chatbots, AI with conversational language |
| 2 | Reasoners, human-level problem solving |
| 3 | Agents, systems that can take actions |
| 4 | Innovators, AI that can aid in invention |
| 5 | Organizations, AI that can do the work of an organization |

*Source: Bloomberg reporting*

前OpenAI研究副总裁 翁荔（Lilian Weng）的定义：
Agent=大模型+记忆+主动规划+根据使用
![](inbox/Pasted%20image%2020250916093305.png)

 
 LangChain作者Horrison的定义：
 Agent是一个使用LLM决定应用程序控制流的系统

吴恩达：与其讨论agent，不如讨论agentic：渐进的智能属性

Langchain
类似自动驾驶汽车的L1-L4分级，一个agentic system的智能程度上可以有不同等级的，取决于LLm对系统行为的决策权重

| Mode                   | Category       | Decide Step Output | Decide Which Steps to Take | Decide Which Steps Are Available |
| ---------------------- | -------------- | ------------------ | -------------------------- | -------------------------------- |
| Code                   | 👨-Driven   | 👨              | 👨                      | 👨                            |
| LLM Call               | 👨-Driven   | ⚙️              | 👨                      | 👨                            |
| Chain (multiple steps) | 👨-Driven   | ⚙️              | ⚙️                      | 👨                            |
| Router (no cycles)     | 👨-Driven   | ⚙️              | ⚙️                      | 👨                            |
| State Machine (cycles) | ⚙️-Executed | ⚙️              | ⚙️                      | 👨                            |
| Autonomous (cycles)    | ⚙️-Executed | ⚙️              | ⚙️                      | ⚙️                            |

## 智能体系统（Agentic System）的划分
---
参考SOP公司的划分
从架构上看，agentic system可以分为两大类系统
- 工作流（Workflow）：通过预定义代码路径编排LLM和工具
- 自主智能体（Autonomous Agent）：LLM动态控制决策的工具使用，自主规划任务

特点对比：
- 工作流侧重流程固定和可预测性
- 自主智能体侧重灵活性和自我决策

适用场景：
一般场景可能不需要用到agentic system，当确定性、复杂性需求方案时，才会考虑工作流或者自主智能体：
- 工作流适用于任务明确、步骤可预定义的场景
- 自主智能体适用于任务步骤难以预知、需长期自主规划的场景
一般场景使用RAG+Prompt优化可能已经满足，增加系统复杂度伴随延迟和成本，需要权衡利弊。

## 智能体系统（Agentic System）组成模块
---
### 1. 基础构建模块：增强型LLM
智能体系统的基础上**检索、工具使用和记忆能力**的增强型LLM
- 复杂应用使用的模型能力都是基于API，建议一定要搞清楚大模型接口逻辑
- 开发时有限直接使用API，只在必要时借助高级框架
![](inbox/Pasted%20image%2020250916095503.png)

参考OpenAI的api接口：[OpenAI Platform](https://platform.openai.com/docs/api-reference/responses/create)

### 2. 工作流（Workflow）的常见模式：
**提示链Prompt Chaining**
定义：
- 顺序拆分任务，每一步由LLM生成内容
- 可以在任意中间步骤添加程序检查（如下图的“Gate”）以确保整个过程依然按计划执行
适用场景示例：
- 先生成市场营销文案，再将其翻译成另一种语言
- 编写文档提纲、检验提纲是否符合某些标准，然后再根据提纲书写出完整文档
![](inbox/Pasted%20image%2020250916101514.png)

**2.1 路由Routing**
定义：
- 根据输入分类，分配给专门的后续任务
适用场景示例：
- 不同类型的客户服务请求（常见问题、退款请求、技术支持）进入不同的后续流程、提示词及工具。
- 将**简单/常见问题**路由给较**小模型**，**将困难/罕见问题**路由给**更强大的模型**，以优化成本与速度。
![](inbox/Pasted%20image%2020250916124131.png)

**2.2 并行Parallelization** 
同时执行多个任务，然后将它们的输出聚合在一起，有两个主要变体：
- 分段（Sectioning）：将任务划分为可以并行运行的独立子任务
- 投票（Voting）：对同一任务就行多次执行，从而获得多元化的输出，以进行对比或投票
适用场景示例：
- （分段）并行内容审核与朱任务处理：一个模型实例负责处理用户查询，另一个模型实例同时对查询进行不恰当或非法请求筛查。「现有大部分的Chat-LLM生成的工作流」
- （投票）评估某段内容是否不当：多个提示从不同角度评估，或设定不同的投票阈值以平衡误报与漏报
![](inbox/Pasted%20image%2020250916130653.png)

**2.3 协调者-工作流Orchestrator-Workers**
定义：
- 协调者拆解任务
- 工作者专注子任务
- 区别并行，其子任务不可预知
场景示例：
- 多文件代码修改：每次都需要对多个文件进行复杂改动的编程产品
- 多源信息搜索分析：在多源信息中搜索并分析可能相关的信息来完成搜索任务
![](inbox/Pasted%20image%2020250916132122.png)

**2.4 评估-优化循环 Evaluator-Optimizer**
定义：
- 一个LLM生成输出
- 另外一个LLM进行反馈和优化
- 反复循环
适用场景示例：
- 文字翻译润色：一些细微的语言差异可能在初稿中并未充分体现，而评估者LLM可以指出这些不足并给出改进意见
- 复杂搜索任务的多轮优化：需要多轮搜索和分析以收集全面信息，由评估判断是否需要进一步搜索
![](inbox/Pasted%20image%2020250916140412.png)

### 3. 自主智能体（Autonomous Agent）
定义：
- 执行过程中获取环境真实反馈（例如工具调用的结果或代码执行情况）
- 支持人工检查点干预（可在检查点或遇到阻碍时暂停，等待人类反馈）
- 设置终止条件（任务通常在完成时终止，但也常常设置停止条件（如最大迭代次数）以保证不会无休止运行）

适用场景：适用于开放性问题，即任务步骤数量难以预知或无法预先固定时。
![](inbox/Pasted%20image%2020250916222631.png)

在基于LLM的自主智能体系统中，LLM充当智能体的大脑，需结合以下关键组件：
- 规划模块
	- 子目标拆解：将复杂任务分解为可管理的子目标
	- 反思优化：通过自我评估改进执行策略
- 记忆系统
	- 短期记忆（上下文）
	- 长期记忆（外部储存）
- 工具使用：获取预训练模型外的实时信息与功能扩展
![](inbox/Pasted%20image%2020250916223205.png)

## 国内外有哪些Agent平台、框架、产品
---
### 1. 构建智能体系统的框架
- 全代码框架
	- LangChain&LangGraph（开源）
	- Llamalndex（开源）
- 低代码平台
	- 毕昇（Apache 2.0 开源，专注企业级场景）
	- Dify（开源）
	- Coze（Apache2.0开源）
	- FastGPT（开源）

### 2. 低代码workflow产品设计的思考
- 独立、**完备**的流程编排框架
	- 不简单是一个被bot调用的工具
	- 不需要划分出chatflow和workflow

- Human in the loop 特性：中间过程支持灵活的输入/输出和多样化的人机交互
	- 复杂业务场景需要人类与AI进行协作：人类也能参与执行过程的判断和决策

![](inbox/Pasted%20image%2020250916224027.png)

- 节点之间是否支持成环？
	- 核心在于workflow执行引擎抽象出来后，到底是一个DAG（有向无环图）还是状态机？
	- 工具形态语言与自动机理论：支持成环能勾更多范围的场景
![](inbox/Pasted%20image%2020250916225305.png)
![](inbox/Pasted%20image%2020250916225310.png)


### 3. Agent产品
ChatGPT DeepResearch
Manus
扣子空间
毕昇灵思
**AutoGLM沉思**


## 总结
---
### 1. 智能体定义
agentic

### 2. agentic system划分
从架构上划分为：
- 自主智能体（Autonomous Agent）
- 工作流（Workflow）

### 3. 自主智能体
三大模块

### 4. 工作流模块
- 包含提示链、路由、并行、协调者——工作者、评估——优化循环等
- 从0手搓还是先用框架？
	- 设计上应避免盲目追求过度复杂性
	- 针对具体任务，从简单方案出发，在根据性能与需求逐步优化

### 5. 推荐阅读&参考资料

- **OpenAI API 文档**：<https://platform.openai.com/docs/guides/function-calling>
- **lilian weng**：《LLM Powered Autonomous Agents》- <https://lilianweng.github.io/posts/2023-06-23-agent/>
- **Langchain**：《What is an AI agent?》- <https://blog.langchain.dev/what-is-an-agent/>
- **Anthropic**：《Building effective agents》- <https://www.anthropic.com/research/building-effective-agents>

**相关框架 & 平台**
- **langchain**：<https://www.langchain.com/>
- **llamaindex**：<https://www.llamaindex.ai/>
- **毕昇 BISHENG**：<https://bisheng.ai.com/>
- **Dify**：<https://cloud.dify.ai/>
- **Coze**：<https://www.coze.cn/>

## 扩散
Agent方向有两个：（ToB视角）
- 工作流（业务流程、固定流程）
- 智能体（通用智能体为目标）

低代码的未来发展：
- 用于做agent快速迭代，做demo
- 给业务人员快速上手使用自主搭建简单Agent

纯代码推荐学习：
- LangGraph

商业化问题：
- 做Agent给Sass生态还不成熟
- 央企国企需要私有化的Sass系统
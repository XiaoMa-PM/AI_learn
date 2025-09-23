# Agent设计模式与工程化
---
**吕昭波**
Datawhale成员，AI架构师，MumuLab创始人
直播时间：2025/09/16
学习时间：2025/09/23
**Abstract**
提供Agent/AI的工程化要点
目的是实现“规模化”+“标准化”展开工程工作

AI智能客服案例（用案例切入今天主题），说明此类流程的全封装成模版进行不同业务套用的作用性。
可以基于一个智能客服--拓展-->迎新Ai小助手
核心逻辑就是把一个落地场景agent进行logic-Abstract成一个可复现的东西
如同常见的开发23模式套用。
现在大概可以抽象的要点：
- 安全护栏法则
- 对话类
- RAG类
![](inbox/Pasted%20image%2020250923223704.png)

提出框架设计框架需要从下面三个点切入去看待

## 一、3大核心能力：
---
Planning·Tool-use·Miming
模型的三个能力
![](inbox/Pasted%20image%2020250923235915.png)
- Planning：解决批量、todo list、项目计划管理
- Miming：个性化推荐（类比数据挖掘）、RAG
- Tool-Use：工具想象（拓展边界）

三个维度后，我们如何重构体系去定位应用场景和批量化场景？
![](inbox/Pasted%20image%2020250924000131.png)
选择“传统场景”+“新技术”，更容易实现“模版化”和“规模化”
模版：
- 批量生成图文
- chabot/知识库->智能助手
- AI新交互突破创新和能力限制

## 二、6资源维度：
---
数据·模型·工具·算力·场景·组织
有了规则--3个Ai能力-->模版-->资源层面了

评估成本适度思考资源，这涉及到企业资产层面

Module1:模型服务
- 选用本地部署、还是云调用
- 要配置多少种类型模型与服务
- 选、调、评、用

Module2:数据
- 数据集--微调、精调、评估
- 数据库--传统数据库概念
- 知识库--用于RAG，由于文档组成--用于扩大LLM组成

Module3:工具tools
插件、工具流、API/MCP、代码片段
Tool如何编排？采用串联or？

Module4:Prompt/上下文context
如何团队内部高度共享使用

Module5:场景（业务流程理解与疏离）
Chatbot
-->飞书/企业微信/顶顶
-->OA/CRM

Module6:算力
- 云/本地的选择
- 场景影响/成本影响

## 三、6架构维度：
---
开发·评估·部署·安全·观测·治理
目的是可控、可解释、可改进的卓越架构，去思考着几个点部署
Pillar1:开发
![](inbox/Pasted%20image%2020250924001218.png)

Pillar2:评估
LLM、API、插件、设施、数据、服务等评估
确保全线高度可用
![](inbox/Pasted%20image%2020250924001227.png)

Pillar3:部署
![](inbox/Pasted%20image%2020250924001239.png)

Pillar4:安全、合规
生成内容的拦截Safe
![](inbox/Pasted%20image%2020250924001300.png)

Pillar5:可观测
可感知
![](inbox/Pasted%20image%2020250924001312.png)

Pillar6:治理
理解为工具维度和模型维护更新
![](inbox/Pasted%20image%2020250924001321.png)

## 四、Agentic思维升级
---
通过架构画布进阶Agentic架构师
![](inbox/Pasted%20image%2020250924001333.png)
![](inbox/Pasted%20image%2020250924001512.png)

可以考虑选几种目前常见的Agent去尝试将其架构画布填写，或者做自己的目标Agent填写：

![](inbox/Pasted%20image%2020250924001524.png)
![](inbox/Pasted%20image%2020250924001533.png)

几个示例
![](inbox/Pasted%20image%2020250924001545.png)
![](inbox/Pasted%20image%2020250924001603.png)
![](inbox/Pasted%20image%2020250924001626.png)
![](inbox/Pasted%20image%2020250924001634.png)
![](inbox/Pasted%20image%2020250924001706.png)
学习路径--学会去做分享，自己错误的经验和成功的经验都是值得分享

## Q&A

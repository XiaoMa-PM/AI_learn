# Agent+MCP开发范式与最佳实践
---
**高大伟**
阿里巴巴通义实验室
AgentScope技术负责人
gaodawei.gdw@alibaba-inc.com
直播时间：2025/09/18
学习时间：2025/09/29

## MCP改变了什么？
---
工具构建过程变化
- 将工具逻辑封装为函数
- 面向智能体构件提示（Prompt）
	- 内容：介绍工具用处，如何使用，以及注意事项
MCP
- 分发逻辑
	- 对现有工具（地图API、模型服务），提供新的发布渠道
	- 使用，构建新工具（复杂人间，专业工具）的难度降低
- 发展速度
MCP作用：
- 用于减轻工具上手难度，加快开发agent速度
- MCP也是现成新的tool/app生态，类比apple/安卓的app市场

## MCP实践-思考-展望
---
Talk is cheap，show me your code
### 实践：使用MCP构建智能体应用
**AgentScope 1.0 with MCP出现的原因**
举例：
“搜索云谷园区附近的咖啡厅”案例
原本的agent流程：
- 链接高德MCP Server，添加工具
- 创建ReAct和user智能体
- 通过信息传递，显式构建对话


## 思考：如何解决实践中出现的问题
**这是好的agent吗？**
NO
样例：高德MCP Server的函数

|函数名称|函数介绍|输入参数|输出结果|
|---|---|---|---|
|`maps_text_search`|根据用户传入关键词，搜索相关的 POI 地点信息|keywords, city|POIs|
|`maps_geo`|将详细的结构化地址转换为经纬度坐标|address, city|location|
|`maps_around_search`|根据用户传入关键词以及坐标，搜索出 radius 半径范围的 POI 地点信息|keywords, location, radius|POIs|
执行过程：
出现了识别地点错误，原因如下：
- AI幻觉进行地点假设（用禁止假设Prompt去控制）
- 结构化地址输入去解决
- 运用MCP自带的结构化，生成新的workflow节点去解决

**这是好的agent吗？**
No
这个MCP中暗含了一个隐藏的SOP（Standard Operation Process，类似RAG）：
用户的非结构化文本 => maps_text_search 改写 => 结构化文本 => maps_geo搜索 => 得到准确坐标

**这是好的agent吗？**
No
出现传回的经纬度不准确问题，并且会根据不同的LLM而不同，这种就涉及到了MCP的质量问题了

## 展望：如何更好地构建智能体应用
**MCP质量控制**
MCP质量控制缺位
- 智能体开发者：面对不同的MCP Server，如何进行挑选？
- MCP的开发者：我开发了一个MCP Server，如何评价好坏？
两者之间存在一个灰色地带
![](inbox/Pasted%20image%2020250930022211.png)

这种对于Agent开发者而言MCP的情况：
- 简单的MCP：我直接实现就好了，透明、可控、可编辑
- 复杂的MCP：我需要谨慎对待，去详细搞清楚文档与逻辑（不亚于去开发新的tools）

该问题区别于软件开发SDK的问题
- 不同点
	- 对于LLM，人有期待和简化需求
		- 自动、自主去完成工具理解与学习，探索并且了解工具边界
		- 长期，更少的认为干预
	- 伴随MCP兴起，更多复杂工具出现，加重了MCP试错成本（不利于MCP生态发展）
提出区分：无状态、有状态的MCP
- 例子：浏览器检索MCP长期后台挂着（有状态）
允许开发者进行函数级别的MCP操作

**MCP解决一切了吗？**
多个MCP Server
- 每个智能体负责一个MCPserver
	- 导致智能体之间交互复杂化，也存在一些断点不可控

**Agent Scope构建的是一个生态思路**
- 增设了Toolskit（对工具进行归类，设置新的工具组，需要使用的时候由元工具去激活（布尔值））
- 元工具是一个对工具的提炼，防止跨类型工具的调用去导致调用断点，该工具的另外一层意义是去解决工具上下文过长
这个问题用了浏览器检索到财报，直接调用了财报分析工具，出现检索断电问题。

agent的MCP工具不是一个孤立的组成部分
- 让LLM训练理解MCP
- 自我学习/演化（今天的例子是一个程序员去试错，可以让LLM自己去试错迭代，形成新的LLM版本）
- 长期记忆

## 拓展
他这个案例解决agent问题的思路是是什么？
#Question 

Function calling与ReAct区别？为什么使用ReAct？
#Question 

Research的工具任务，Agent不应该进行keyword相似度匹配计算，应该直接禁止这种情况（检索AI幻觉，或者进行权重再定义）


# Agent原理与最简实践
---
锦恢
- Datawhale成员
- OPenMCP负责人，SlidevAI作者
- 深度参与ColossaAI大模型训练架构开发
直播日期：
学习日期：
Link：

## 一、MCP的由来和原理
---
再次之前安装uv和编译器的OpenMCP插件

### 1.1 Function Calling基础概念
OpenAI 的API下的一环

**Motivation**
之前的LLM解决的text交互问题，但是为了解决更多的交互问题，出现了function calling去实现
![](inbox/Pasted%20image%2020250923004139.png)


**Elements**
XML指令包裹、引导解码器
[🔖 两种基本方法与原理-博客 \| 汇尘轩 - 锦恢的博客](https://kirigaya.cn/blog/article?seq=325)
[🔖 如何约束大模型严格输出制定内容？引导解码器Guided Decoder博客 \| 汇尘轩 - 锦恢的博客](https://kirigaya.cn/blog/article?seq=345)

### 1.2 一次Function-Calling的案例
注：建议的Function-Calling，典型的`get_weather`案例

- 定义LLM可用工具函数Function`get_weather`
- `tools`字段注入上下文，对Funtion进行封装，对齐包装+说明。此处为数组数据类型。
- 返回`tools_calls`工具的元素，如0--`get_weather`。此处同上为数组数据类型。


```python
from openai import OpenAI

client = OpenAI(
    api_key="<your api key>",
    base_url="https://api.deepseek.com",
)

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather of a location, the user should supply a location first.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    }
                },
                "required": ["location"]
            },
        }
    },
]

def send_messages(messages):
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        tools=tools
    )
    return response.choices[0].message

messages = [{"role": "user", "content": "How's the weather in Hangzhou?"}]# prompt
message = send_messages(messages)
print(f"User>\t {messages[0]['content']}")

tool = message.tool_calls[0]
messages.append(message)# resource

messages.append({"role": "tool", "tool_call_id": tool.id, "content": "24℃"})# resource
message = send_messages(messages)
print(f"Model>\t {message.content}")
```

### 1.3 Motivation of MCP
开发痛点：
- 业务规模扩大带来的复杂性
- 代码维护性下降
- 系统不确定性挑战
- 开发心智负担加重
如后端开发过程的业务很多是SOP硬编码，而Function Calling大部分都需要用到维护和调整。
而正常点击按钮的业务线上post请求/sse请求->java去走业务逻辑。

出现了很多优化空间，可以把工程化内容SOP内容进行==Abstract==

很多Agent框架如LangGraph、auto gpt等都是把代码结构、工程框架封装去解决该痛点。等于**封装了SOP部分**。

而实际上还有很多内容可以**Abstract**。
于是做了：==tools、prompt、resource==部分的耦合工作，组成MCP服务（如简易示例中的过程message，实际上的resoure不至于此，还有web的访问痕迹等）

### 1.4 Definiton MCP协议
- 工具、提示词、资源->`MCP服务器`
- 准入MCP服务器的接口协议->`MCP协议`
作为了agent后端服务的一部分，MCP协议实现了纯粹的后端Agent业务逻辑的剥离与解耦合

统一的MCP协议用于让不同客户端直接准入访问MCP服务器

**对比有无MCP协议--无MCP协议**
耦合度高，开发和维护成本大
![](inbox/Pasted%20image%2020250923011602.png)
组装并生成提示词=传参（字段）+prompt
这块地方的“组装并生成提示词”是内粘在后端的代码中的，导致后端的工作内容复杂化。

编写工具调用规则+API也会增加工作复杂性

**对比有无MCP协议--有MCP协议**
后端服务与Agent解耦，扩展灵活
![](inbox/Pasted%20image%2020250923012034.png)
“根据参数”与“Prompt”需要双方约定好传参名与Prompt名称

## 二、MCP开发最简案例
---
### 2.1 开发环境配置
- VS code最新版本
- OpenMCP插件安装[OpenMCP](https://openmcp.kirigaya.cn/)
- Python3.10+运行
- 大模型API Token准备

OpenMCP项目：[GitHub - LSTM-Kirigaya/openmcp-client: All in one vscode plugin for mcp developer](https://github.com/LSTM-Kirigaya/openmcp-client)

### 2.2 开发案例Word MCP简介
Word MCP--根据文案生成Word文档
解决痛点：
- LLM输出的text复制入word存在格式问题（如md转doxc的文本问题）
- 需要自行使用插件调整问题

### 2.3 MCP调试&优质资源
**常用MCP Server框架介绍**
![](inbox/Pasted%20image%2020250923012627.png)
fastmcp的步骤与内容最少，可以完成简易的mcp
>这些框架抉择没有那么重要，MCP开发概念大于框架技术

**MCP调试&资源**
>容易存在误区：API服务包装成MCP直接调用
>MCP完成后必须进行调试

Agent开发需要理解业务知识，去判断Tool怎么做，==Tool、Resource、Prompt与业务关联性==强，很重要
![](inbox/Pasted%20image%2020250923013026.png)
MCP Inspector针对Web服务，在Web端使用，做tool、prompt、resource
MCP-CLI属于MCP Inspector的命令行，用于命令调用测试
Cherry Studio原本是最早做LLM API的客户端，这里是做完MCP上传上去接入该平台，注册接入LLM模型进行调试
OpenMCP Client统一成编译器插件，专用于MCP服务调试

**优质资源荟萃**
![](inbox/Pasted%20image%2020250923013358.png)
首推：[GitHub - modelcontextprotocol/servers: Model Context Protocol Servers](https://github.com/modelcontextprotocol/servers)
可关注独立开发者：艾逗笔

### 2.4 本次技术选型
fastmcp（开发服务器）+OpenMCP（调试验证MCP服务器）

**fastmcp安装**
```python
pip install uv
uv add mcp "mcp[cli]"
```
拓展uv：
- 管理环境工具
- 使用uv去隔离环境，符合mcp隔离特征
- `uv init`创建环境管理
- `uv add mcp"mc[cli]"`运行环境下安装包
- `uv run mcp --help`

**OpenMCP安装**
在对应编辑器的插件商城搜索openmcp下载即可

### 2.5 fastmcp快速入门


## 三、从大模型到AI Agent
---
了解技术的历史很重要
- 明白技术的发展
- 技术的边界

### 3.1 技术积累阶段2015-2021
最早一批公司：百度、微软、谷歌
特点：搜索引擎公司
数据库的数据变相->数据挖掘（规模更大）->知识图谱、拓扑分析、大模型（续写文章，无法回答内容）
数据挖掘e.g. 通过QQ加好友，各类型属性去推断其真实性别

堆参数做vc，vc越强表达的语义空间越丰富，承载信息量越大。
转折点在于以下四项技术：
- NVIDA CUDA生态&NVLink技术->硬件
- Transformer架构演进->算法
- DeepSpeed训练框架->Infra
- 大规模语料积累->数据

CUDA生态：让Ai的编译、训练都可以用python使用，降低了Ai的门槛
NVLink技术：训练大模型需要多张卡，通过NVLink去做连接多显卡传输数据

Transformer：处理自然语言，以前很多用RNN，RNN串形态，但是在做数据参数增大会导致其层数变大，时间变长。把自然语言表征改为并行，可以让模型无限变大。

DeepSpeed：embedding在训练中做的是索引工作，然而参数量有很大。而deepspeed分割数据后在多个卡上负载均衡。

这三者解决了：
- 不断累积数据
- 不断加卡无工程成本
- 不断加大模型尺寸

大规模预料积：大模型的养料，也是大模型最终发展边界。

### 3.2 大规模验证时期2020-2023
大模型竞争步入进程

- GPT-3：参数规模与智能提升
- InstructGPT：RLHF训练范式
- ChatGPT爆火
- GPT-R1：深度思考
- 百模大战（预估今年年底结束）

快速消耗算力、算法、infra和数据红利

### 3.3 迈向Agent时代2024-至今
![](inbox/Pasted%20image%2020250923180228.png)
- 目前仅应用于output-chat很局限，缺乏商业价值
- 目前还做到code

通用大模型发展缓慢，大家都在agent上发力了

### 3.4 MCP与AI Agent的未来
**Agent全生命周期管理**
- 开发、验证、迭代工作
- Agent会成为一个系统开发
用户使用MCP产生的数据，回流给LLM，去做post-traning。（操作流数据）

**待解决的核心问题**
- 如何验证真实业务场景下的结果？（评估）
	- 示例：Gpt-5的一大改进就是做了check list。打分。
- 如何通过验证的反馈去迭代系统？（自迭代）
	- 示例：代码的debug，是否此处流程已经达成目的？是修改tool还是prompt？
- 如何将这些步骤进行标准化和统一？（工业化、标准化）

Agent到真正应用之间的GAP

## Q&A
---

1. 很多Tools接入同一个agent，考虑到调用准确性，有没有什么好的解决方法？
现在LLM默认要求Tools智能只有100个
- 进行工具整合
- MCPzero论文：做tool分类，让大模型工具问题的对工具选择，先选择类别然后选择具体工具。（类似聚类）缺点：是损失语义信息，也会在调用时有性能损失。

2. 能否对RAG对Tool做初筛，再放到tool的MCP定义中？或者在Tool上封装一层，提供一个查询所有Tool和调用Tool的MCP，不暴露所有的Tool Schema。
- 和MCPzero相似

3. Agent在科研上的未来趋势会怎么样？
- 做agent生命周期，如何实现一个规范的流程。存在“五个轮子马车”问题。如何将最优的模块组合成最有效的系统。
- 验证环节：目前是很难的点，用什么方法？用于什么领域效果与分数会更高？这些都需要去尝试。
- 看MCPzero，看阿发论文社区

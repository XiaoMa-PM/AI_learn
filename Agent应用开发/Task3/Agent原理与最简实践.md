# Agent原理与最简实践
---
宋志学
- Datawhale成员
- happy-llm、self-llm等万星系开源项目作者
- 其他项目：
	- tiny-universe
	- huanhuan-chat
	- Amchat
	- d2l-ai-solution-manual
	- prompt-engineering-for-developers

## 一、Agent原理深度解析
---
### 1.1 Agent核心概念
#### Agent的本质公式：
`大模型+记忆+工具=Agent`
Agent（智能体）是能够感知环境、做出决策并采取行动以实现特定目标的自主实体。与传统程序相比，Agent具备以下核心特性：
- **自主性**：无需人工干预即可独立运行
- **反应性**：能对环境变化做出实时响应
- **主动性**：主动追求目标而非被动响应
- **社会性**：能与其他Agent或人类进行交互

当前主流的Agent架构：
1. ReAct（推理+行动）
	- 将思考和行动融合在每个步骤中
	- 通过观察-思考-行动的循环实现决策
	- 适合需要实时响应的动态任务
2. Plan-and-Solve（规划-求解）
	- 先规划再执行的解耦式架构
	- 制定详细计划后严格按照步骤执行
	- 适合需要长远规划的复杂任务
3. Reflection（反思优化）
	- 执行->反思->优化的三步循环
	- 通过自我评估和迭代改进提升质量
	- 适合追求高精度的关键任务

#### ReAct 架构详解
ReAct（Reasoning+Acting），其核心思想是：
`观察环境->思考推理->拆取行动->观察结果->循环`
**ReAct决策循环**：
1. Thought：基于当前观察进行推理
2. Action：选择并执行具体行动
3. Observation：观察行动结果
4. 循环：根据新观察继续思考

示例
天气查询：用户输入--Thought（思考是否需要走入工具判断）--需要工具？--调用天气MCP--选择执行--Observation--Thought（思考是否需要走入工具判断）--需要工具？（已有结果，否）--输出结果
```
用户："北京天气如何？"
Thought：用户询问天气，需要获取北京当前天气信息
Action：weather_query(location="北京")
Observation：{"temperature": 25, "condition": "晴"}
Thought：已获得天气数据，可以回复用户
Action：回复"北京今天25度，晴天"
```

![](inbox/Pasted%20image%2020250921000503.png)
#### 其他主流Agent架构
**Plan-and-Solve Agent**：
- 工作原理：将整个流程解耦为规划阶段和执行阶段
- 规划阶段：接受完整问题，分解任务并制定分步骤的行动计划
- 执行阶段：严格按照计划执行，保持目标一致性，避免**中间步骤迷失方向**
- 优势：在处理多步骤复杂任务时，能够保持更高的目标一致性

**Reflection Agent**：
- 核心思想：灵感来源于人类的学习过程，通过执行->反思->优化的循环提升质量
- 执行阶段：使用ReAct或者Plan-and-Solve生成初步解决方案
- 反思阶段：调用独立的LLM实例担任“评审员”，评估实时性、逻辑性、效率等维度
- 优化阶段：基于反馈内容对初稿进行修正，生成更完善的修订稿

**LangChain Agent**：
- 基于链式调用的Agent框架
- 支持多种提示模板
- 丰富的工具集成生态
- 适合复杂工作流

**AutoGPT**：
- 完全自主的目标追求
- 长期记忆系统
- 自我提示生成
- 适合开放式任务

**MetaGPT**：
- 软件开发的Multi-Agent协作框架
- 模拟真实软件团队角色分工
- 产品经理、架构师、工程师等角色扮演
- 适合自动化软件开发任务

**GAMELAI**：
- 基于角色扮演的对话式Agent框架
- 多Agent协作完成复杂任务
- 强调角色定义和通信协议
- 适合创意写作、教育培训等场景

**架构对比：**

| **架构**         | **复杂度** | **控制力** | **适用场景** |
| -------------- | ------- | ------- | -------- |
| ReAct          | 低       | 高       | 简单决策任务   |
| Plan-and-Solve | 中       | 高       | 多步骤复杂任务  |
| Reflection     | 高       | 中       | 高精度关键任务  |
| LangChain      | 高       | 中       | 复杂工作流    |
| AutoGPT        | 高       | 低       | 自主探索任务   |
| MetaGPT        | 高       | 中       | 软件开发自动化  |
| CAMELAI        | 中       | 高       | 角色扮演对话任务 |

## 二、从零实现最简 React Agent
---
演示项目的GitHub：[blog/Blog/react-agent/code at master · KMnO4-zx/blog · GitHub](https://github.com/KMnO4-zx/blog/tree/master/Blog/react-agent/code)
### Agent 内部数据流：
![](inbox/Pasted%20image%2020250921003459.png)

- 个人认为此处的Agent有点误导，写成**上下文记忆/文本**可能更好一些

### Setp 1:构建大模型
阶段Task：
- 建立模型对话方法
- 建立llm：使用api接口/本地部署
- 命名`llm.py`

使用Qwen3-30B-A3B-Instruct-2507模型

>为什么使用Instruct，学习优先此模型，推理模型容易出现工具调用错误。

创建一个**BaseModel类**，在该类中定义一些基本的方法，比如chta方法，方便以后扩展使用其他模型。

```python
class BaseModel:
    def __init__(self, api_key: str = '') -> None:
        self.api_key = api_key

    def chat(self, prompt: str, history: List[Dict[str, str]], system_prompt: str = "") -> Tuple[str, List[Dict[str, str]]]:
        """
        基础聊天接口
        
        Args:
            prompt: 用户输入
            history: 对话历史
            system_prompt: 系统提示
            
        Returns:
            (模型响应, 更新后的对话历史)
        """
        pass
```

创建一个Siliconflow类，这个类继承自BaseModel类，在这个类中实现chat方法。
硅基流动平台--提供--开源模型的API服务，可以很方便调试和开发。**区别上一个类是把模型放进去了。**

命名llm.py
```Python
class Siliconflow(BaseModel):
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.client = OpenAI(api_key=self.api_key, base_url="https://api.siliconflow.cn/v1")#此处的base_url是调用api服务的link

    def chat(self, prompt: str, history: List[Dict[str, str]] = [], system_prompt: str = "") -> Tuple[str, List[Dict[str, str]]]:
        """
        与 Siliconflow API 进行聊天
        
        Args:
            prompt: 用户输入
            history: 对话历史
            system_prompt: 系统提示
            
        Returns:
            (模型响应, 更新后的对话历史)
        """
        # 构建消息列表
        messages = [
            {"role": "system", "content": system_prompt or "You are a helpful assistant."}
        ]
        
        # 添加历史消息
        if history:
            messages.extend(history)
        
        # 添加当前用户消息
        messages.append({"role": "user", "content": prompt})

        # 调用 API
        response = self.client.chat.completions.create(
            model="Qwen/Qwen3-30B-A3B-Instruct-2507",
            messages=messages,
            temperature=0.6,# 模型回答创意程度
            max_tokens=2000,# 一次回答输出的tokens数量
        )
		
		# 获取模型回答
        model_response = response.choices[0].message.content
        
        # 更新对话历史
        updated_history = messages.copy()
        updated_history.append({"role": "assistant", "content": model_response})

        return model_response, updated_history
```

```python
if __name__ == "__main__":
	llm = Siliconflow(api_key="")
	prompt = "Hello"
	response, history = llm.chat(prompt)
	print("Response:",response)
	print("History:",history)
```

安装的pip：
```txt
openai
requests
json5# 解析字符串比较灵活
```

### Step 2：构建工具
阶段Task：
- 命名：`Tools.py`
- 添加工具的描述信息
- 添加工具的具体实现
- 服务于`system_prompt`去实现模型知道可以调用哪些工具，工具的描述信息和参数

>使用Google搜索功能的话需要去`serper`官网申请一下`token`: [https://serper.dev/dashboard，](https://serper.dev/dashboard%EF%BC%8C) 然后在tools.py文件中填写你的key，这个key每人可以免费申请一个，且有2500次的免费调用额度

省略演示代码：
```python
class ReactTools:
    """
    React Agent 工具类
    为 ReAct Agent 提供标准化的工具接口
    """
    
    def __init__(self) -> None:
        self.toolConfig = self._build_tool_config()# 工具清单
    
    def _build_tool_config(self) -> List[Dict[str, Any]]:
        """构建工具配置信息"""
        return [
            {
                'name_for_human': '谷歌搜索',
                'name_for_model': 'google_search', # 模型调用使用名称
                'description_for_model': '谷歌搜索是一个通用搜索引擎，可用于访问互联网、查询百科知识、了解时事新闻等。',
                'parameters': [
                    {
                        'name': 'search_query',
                        'description': '搜索关键词或短语',
                        'required': True,
                        'schema': {'type': 'string'},# schema数据格式要求
                    }
                ],
            }
        ]

    def google_search(self, search_query: str) -> str:
        """执行谷歌搜索

        可在 https://serper.dev/dashboard 申请 api key

        Args: # Args表示函数需要传入的参数用于做什么的，传入参数说明
            search_query: 搜索关键词
            
        Returns:
            格式化的搜索结果字符串
        """
        url = "https://google.serper.dev/search"

        payload = json.dumps({"q": search_query})# 发送服务器数据，创建了Python字典，json.dumps把其转化为json格式
        headers = {
            'X-API-KEY': 'your serper api key',
            'Content-Type': 'application/json'
        }

        pass
    
    def get_available_tools(self) -> List[str]:
        pass

    def get_tool_description(self, tool_name: str) -> str:
        pass
```

### Step 3:构建React Agent
整个系统的协调者，负责**管理LLM调用、工具执行和状态维护**。

#### 核心组件设计
**Why？**
React Agent的设计采用经典的分层架构模式：
- 接口层：对外暴露简单的`run()`方法
- 协调层：管理思考-行动-观察的循环
- 解析层：从模型输出中提取结构化信息（解析工具函数、参数）
- 执行层：调用具体工具完成任务
优点：结构清晰，扩展性维护性强

**ReactAgent类结构**：

```python
class ReactAgent:
    def __init__(self, api_key: str = '') -> None:
        """初始化 React Agent"""
	        self.api_key = api_key# 模型Api
        self.tools = ReactTools()                    # 工具管理
        self.model = Siliconflow(api_key)      # LLM 客户端
        self.system_prompt = self._build_system_prompt()
    
    def _build_system_prompt(self) -> str:
        """构建 ReAct 系统提示"""
        # 组合工具描述和 ReAct 模式指导
        return prompt_template
    
    def _parse_action(self, text: str) -> tuple[str, dict]:
        """解析模型输出中的行动和参数"""
        # 使用正则表达式提取行动和参数
        action_pattern = r"行动[:：]\s*(\w+)"
        action_input_pattern = r"行动输入[:：]\s*({.*?}|[^\n]*)"
        return action, action_input_dict
    
    def _execute_action(self, action: str, action_input: dict) -> str:
        """执行指定的工具行动"""
        # 调用对应工具并返回结果
        
    def run(self, query: str, max_iterations: int = 3, verbose: bool = True) -> str: # 循环次数为3
        """运行 ReAct Agent 主循环"""
        # 实现思考-行动-观察循环
```

#编程知识
正则化表达式里的“元字符”
`\s `(匹配空白)
- 任何一个空白字符
`\w `(匹配单词字符)
- 代表任何一个 字母、数字或下划线 ( `_` )。
`.` (匹配任意字符)
- 代表 除了换行符以外的任何单个字符 。
`* `和 `?` (控制匹配次数)“贪婪度”
- .`*` (贪婪模式) : `. `匹配任意字符， `*` 匹配零次或多次。组合起来， `.*` 会尽可能 多地 匹配字符。
- `.*?` (非贪婪/懒惰模式) : 在 `*` 后面加上 ? ，就会让匹配变得“懒惰”，也就是尽可能 少地 匹配字符。
代码` ({.*?}|\{.*?\}|[^\n]*) `中，` {.*?} `就是用了非贪婪模式，以确保它只匹配到最近的那个 `}` 就结束，这对于处理嵌套或多个JSON对象的情况非常重要。
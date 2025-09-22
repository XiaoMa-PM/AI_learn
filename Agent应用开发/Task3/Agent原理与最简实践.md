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
### 2.1 Agent 内部数据流：
![](inbox/Pasted%20image%2020250921003459.png)

- 个人认为此处的Agent有点误导，写成**上下文记忆/文本**可能更好一些

### 2.2 系统实现流程
#### Setp 1:构建大模型
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

#### Step 2：构建工具
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
                        'schema': {'type': 'string'},## Schema数据格式要求
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

#### Step 3:构建React Agent
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

### 2.3 系统提示构建
**为什么系统提示如此重要？**
系统提示是React Agent的“大脑”，他直接决定了Agent的行为模式。一个好的系统提示应该包含：
1. 时间信息：让Agent知道道歉时间，避免过时信息
2. 工具清单：明确告诉Agent有哪些工具可用
3. 行为模式：详细的ReAct流程指导
4. 输出格式：规范化的思考-行动-观察格式

**构建思路**：
我们使用f-string动态生成系统提示，这样可以：
- 自动包含当前时间
- 动态加载可用工具列表
- 保持提示的时效性和准确性

**系统提示词模版解析**：
```
prompt = f"""现在时间是 {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}。
你是一位智能助手，可以使用以下工具来回答问题：

{tool_descriptions}

请遵循以下 ReAct 模式：

思考：分析问题和需要使用的工具
行动：选择工具 [google_search] 中的一个
行动输入：提供工具的参数
观察：工具返回的结果

你可以重复以上循环，直到获得足够的信息来回答问题。

最终答案：基于所有信息给出最终答案

开始！"""
```

设计要点：
- 中文提示：更符合国内用户习惯
- 具体工具名：明确告诉模型可用工具
- 循环指导：说明可以多次使用工具
- 最终答案：明确结束条件

### 2.4行动解析机制
**为什么解析这么复杂？**
在实际应用中，大模型的输出格式往往不够规范，可能出现：
- 中英文混合的冒号
- JSON格式的不规范
- 参数缺失和格式错误
- 多余的空格或换行

因此我们需要一个**鲁棒的解析机制**。
**解析思路**：
1. 多层匹配：先用正则提取，再用JSON解析
2. 容错设计：解析失败时提供降级方案
3. 格式兼容：支持JSON字符串和纯文本参数

**解析流程详解**：
```
模型输出：
思考：用户询问特朗普生日，需要搜索
行动：google_search
行动输入：{"search_query": "特朗普生日"}

解析步骤：
1. 正则提取行动 → google_search
2. 正则提取参数 → {"search_query": "特朗普生日"}
3. JSON解析 → {'search_query': '特朗普生日'}
4. 返回结构化数据 → ('google_search', {'search_query': '特朗普生日'})
```

**常见错误场景**：
- ❌ `行动输入：特朗普生日` → 自动转为 `{"search_query": "特朗普生日"}`
- ❌ `行动输入：{"search_query":"特朗普生日"` → 补全JSON格式
- ❌ `行动输入："特朗普生日"` → 去除多余引号

**代码实现**：
```python
def _parse_action(self, text: str, verbose: bool = False) -> tuple[str, dict]:
    """从文本中解析行动和行动输入"""
    # 更灵活的正则表达式模式
    action_pattern = r"行动[:：]\s*(\w+)"
    action_input_pattern = r"行动输入[:：]\s*({.*?}|\{.*?\}|[^\n]*)"
    
    action_match = re.search(action_pattern, text, re.IGNORECASE)
    action_input_match = re.search(action_input_pattern, text, re.DOTALL)
    
    action = action_match.group(1).strip() if action_match else ""
    action_input_str = action_input_match.group(1).strip() if action_input_match else ""
    
    # 清理和解析JSON
    action_input_dict = {}
    if action_input_str:
        try:
            action_input_str = action_input_str.strip()
            if action_input_str.startswith('{') and action_input_str.endswith('}'):
                action_input_dict = json5.loads(action_input_str)
            else:
                # 如果不是JSON格式，尝试解析为简单字符串参数
                action_input_dict = {"search_query": action_input_str.strip('"\'')}
        except Exception as e:
            action_input_dict = {"search_query": action_input_str.strip('"\'')}
    
    return action, action_input_dict
```

### 2.5 ReAct主循环
**什么是ReAct循环？**
ReAct循环是Agent的“心跳”，它让Agent能够：
- 持续思考：基于新信息不断调整策略
- 工具调用：在需要时主动获取外部信息
- 结果整合：将工具结果与已有知识结合
循环的四个阶段：
1. 思考阶段（Thought）：模型分析问题，决定是否需要工具
2. 行动阶段（Action）：选择合适的工具提供参数
3. 观察阶段（Observation）：执行工具并获取结果
4. 整合阶段（Integration）：将新信息整合到上下文中

**为什么需要`max_iterations` ？**
- 防止无限循环：避免模型陷入死循环
- 控制成本：限制API调用次数
- 用户体验：避免过长的响应时间

**状态管理的重要性：**
每次循环都需要：
- 保持上下文：将观察结果加入对话历史
- 更新输入：为下一轮循环准备新的提示
- 历史记录：确保模型指导之前做了什么

**循环逻辑详解**：
```python
def run(self, query: str, max_iterations: int = 3, verbose: bool = True) -> str:
    conversation_history = []
    current_text = f"问题：{query}"
    
    for iteration in range(max_iterations):
        # 获取模型响应
        response, history = self.model.chat(current_text, conversation_history, self.system_prompt)
        
        # 解析行动
        action, action_input = self._parse_action(response, verbose=verbose)
        
        if not action or action == "最终答案":
            return self._format_response(response)
        
        # 执行行动
        observation = self._execute_action(action, action_input)
        
        # 更新上下文继续对话
        current_text = f"{response}\n观察结果:{observation}\n"
        conversation_history = history
    
    return self._format_response(response)
```


## Q&A
---
1. Agent框架、强化学习、优化算法三者是什么关系呢？
三者比较独立
agent是让llm解决问题用的
强化学习是让llm去对齐用户偏好：规范大模型的回答、提升agent能力
优化算法：框架、循环、状态
模型训练优化：k2等的优化器

2. 请问目前最值得学习的agent开发？langgraph怎么样
使用：哪个都行
学习框架原理：openai的swam框架，代码精简。

3. 使用langgragh做很简单吗这个框架和kanggragh相比有哪些优势？
	a. 什么场景下需要自己搭建实现agent而不是使用成熟的agent框架？
	b. 理解agenr原理对后续企业级agent应用开发有哪些帮助？
ReAct用于学习的，langgragh可以适用于生产环境了
在特定化领域针对性修改，需要自己做agent框架。
工作可以使用成熟的框架去实现即刻。
企业级agent应用都是封装特别好了，如dify等，了解原理后，有利于快速优化和参数理解。

4. 怎么管理长期记忆？
	a. 记忆的持久化保持一般用什么方法？
	b. 如何对agent的记忆进行管理
上下文太长导致记忆不清楚之前的内容，设置超出窗口（超过最大输入尺寸）总结前文内容。持久化保存和本地文件以RAG支持大模型。
克劳德的方法：使用命令搜索内容，用RAG管理记忆。

5. 调用工具失败，反思纠错逻辑如何？
首先工具必须写清楚，其次是系统提示词
换个模型

6. codex和cc的agent设计有什么特别？
看官方文档

7. 请问什么agent框架适合入门开发使用呢？看到阿里有一个agentscope？
都适合，对着文档看去使用，并且尝试理解它的底层逻辑怎么做的。

8. 实际工程中使用agent还需要哪些优化或注意的地方？
明确agent需要解决的问题。

9. 目前agent框架很多，一般怎么选择agent？
agentscope和msagent

10. chatgpt里的gpt5thinking一个问题可以回答==好几分钟==，==多次搜索==，这种agent是怎么实现的？
ReAct的原理即为本体答案，现在的llm的chatbox都是有这个基本框架的。
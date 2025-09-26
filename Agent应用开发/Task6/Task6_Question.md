打卡内容:动手开发一个工作流agent

## 建筑案例检索器
---
类型：chatflow
用途：建筑设计前期调研，快速检索目标建筑类型的建筑案例，并且进行分类输出
使用模型：dlm-4-air（联网检索）、Qwen3-235B-A22B（案例分析归类，输出html）
参考于：[5. 进阶-chatflow：小红书读书卡片 - 飞书云文档](https://datawhaler.feishu.cn/wiki/ZB3ZwD2foi2EWukNypecpVtrnoY)

### flow
![](inbox/Pasted%20image%2020250927022040.png)

### 开始
![](inbox/Pasted%20image%2020250927022108.png)

### LLM（glm-4-air）
系统提示词
```
# 角色

你是一名专业的建筑信息研究员。

  

# 任务

根据用户提供的建筑类型，在全球范围内检索相关的优秀建筑设计案例。你需要输出每个案例的名称、可以访问的介绍网址、**案例的详细地址**以及案例的简要介绍。

  

# 输入

建筑类型：{{building_type}}

  

# 输出要求

1. 请最少检索 5 个相关的建筑案例。

2. 你的输出**必须**是一个严格的 JSON 数组格式，不要在 JSON 内容前后添加任何解释性文字或 ```json 标记。

3. 每个 JSON 对象包含以下**四个键 (key)**：

- "name": 案例的完整名称。

- "url": 介绍该案例的有效网址。

- "address": 案例所在的详细地理位置（国家、城市等）。如果找不到详细地址，请至少填写国家和城市。

- "description": 对该案例的简要介绍（100-200字）。

  

# 示例输出格式

[

{

"name": "宁波博物馆",

"url": "[https://www.archdaily.cn/cn/02-1407/ning-bo-bo-wu-guan-wang-shu](https://www.archdaily.cn/cn/02-1407/ning-bo-bo-wu-guan-wang-shu)",

"address": "中国，浙江省，宁波市，鄞州区首南中路1000号",

"description": "由普利兹克奖得主王澍设计，建筑形态以山、水、海洋为设计理念，外墙采用了大量宁波旧城改造中收集的旧砖瓦，体现了地域文化和循环建造的理念。"

},

{

"name": "新世纪福音战士主题美术馆",

"url": "[https://www.archdaily.com/964601/fuji-eva-a-themed-attraction-in-an-amusement-park-suppose-design-office](https://www.archdaily.com/964601/fuji-eva-a-themed-attraction-in-an-amusement-park-suppose-design-office)",

"address": "日本，山梨县，富士吉田市",

"description": "位于富士急高原乐园内，该建筑以其独特的几何形态和与动漫主题的深度结合而闻名，为游客提供了沉浸式的体验。"

}

]
```

### LLM（Qwen3-235B-A22B）
```
# 角色

你是一名建筑设计助理，擅长整理资料并以清晰的网页格式呈现。

  

# 任务

接收一个包含建筑案例信息的 JSON 数组，并根据以下规则进行分类和整理，最终生成一份 HTML 格式的报告。

  

# 输入数据

以下是待处理的建筑案例 JSON 数据：

{{case_data}}

  

# 处理规则

1. **一级分类**: 将所有案例分为“国内案例”和“国外案例”。根据案例的地理位置进行判断。

2. **二级分类**: 在“国内案例”中，根据其地理位置和气候特点，进一步划分为“南方建筑”和“北方建筑”。

3. **格式化输出**:

- 生成一个完整的 HTML 页面代码。

- 使用 `<h2>` 标签作为一级分类标题。

- 使用 `<h3>` 标签作为二级分类标题。

- 在每个分类下，使用无序列表 `<ul>` 和 `<li>` 来展示案例。

- 每个案例的名称应该是一个可以点击的链接（`<a>` 标签）。

- **在链接下方，新增一行用于展示地址，格式为 `<p><strong>地址:</strong> {{案例地址}}</p>`**。

- 在地址下方附上案例的简介。

  

# 输出要求

- 你的输出**必须**是纯粹的 HTML 代码，不要包含任何额外的解释或 ```html 标记。

- 如果某个分类下没有案例，则不显示该分类的标题。
```

### 效果
```html
<!DOCTYPE html> <html lang="zh-CN"> <head> <meta charset="UTF-8"> <title>建筑案例分析报告</title> <style> body {font-family: Arial, sans-serif; line-height: 1.6; max-width: 800px; margin: 20px auto; padding: 0 20px;} h2 {color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 5px;} h3 {color: #34495e;} ul {padding-left: 20px;} li {margin-bottom: 20px;} .address {color: #7f8c8d; margin: 5px 0;} </style> </head> <body> <h2>国内案例</h2> <h3>南方建筑</h3> <ul> <li> <a href="https://www.library.sh.cn/" target="_blank">上海图书馆</a> <p class="address"><strong>地址:</strong> 中国，上海市，徐汇区，淮海中路1555号</p> <p>上海图书馆是中国最大的公共图书馆之一，它的设计结合了中国传统元素和现代建筑风格，是一个集学习、研究和社区活动于一体的服务中心。</p> </li> </ul> <h2>国外案例</h2> <ul> <li> <a href="https://www.nypl.org/" target="_blank">纽约公共图书馆</a> <p class="address"><strong>地址:</strong> 美国，纽约州，纽约市，第五大道与42街交叉口</p> <p>纽约公共图书馆是世界上最著名的公共图书馆之一，它的主楼是一座古典主义建筑，由约翰·梅森·鲍德温设计，是19世纪末20世纪初纽约市的象征之一。</p> </li> <li> <a href="https://www.spl.org/" target="_blank">西雅图中央图书馆</a> <p class="address"><strong>地址:</strong> 美国，华盛顿州，西雅图市，1000 Fourth Ave</p> <p>西雅图中央图书馆由著名建筑师诺曼·福斯特设计，是一个现代化的社区服务中心，以其独特的菱形玻璃外观和内部空间的灵活布局而闻名。</p> </li> <li> <a href="https://www.oba.nl/" target="_blank">阿姆斯特丹公共图书馆</a> <p class="address"><strong>地址:</strong> 荷兰，阿姆斯特丹市，阿姆斯特丹中心区</p> <p>阿姆斯特丹公共图书馆是一个结合了传统图书馆与现代社区服务中心功能的建筑，它的设计鼓励社交互动，提供了多种学习与活动空间。</p> </li> <li> <a href="https://www.torontopubliclibrary.ca/" target="_blank">多伦多公共图书馆</a> <p class="address"><strong>地址:</strong> 加拿大，安大略省，多伦多市，550 Spadina Road</p> <p>多伦多公共图书馆系统遍布全市，提供了各种教育、文化和社区活动，其中某些分馆的建筑设计反映了现代与传统的融合。</p> </li> </ul> </body> </html>
```

![](inbox/Pasted%20image%2020250927022620.png)

### 当前问题
实测可用性不高：
- 检索无法直接输出html结果
- 检索的链接点击跳转页面大部分输出案例丢失页面
- ![](inbox/Pasted%20image%2020250927022932.png)
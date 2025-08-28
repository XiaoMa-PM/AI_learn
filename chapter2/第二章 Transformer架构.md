# 第二章 Transformer架构
---

## 一、注意力机制
---

### 1. 什么是注意力机制
Word2Vec单层神经网络-->神经网络
计算机视觉（Computer Vision）CV发展起源**神经网络核心三种架构：**
- 全连接神经网络（Feedforward Neural Network，FNN）
- 卷积神经网络（Convolutional Neural Network，CNN）
- 循环神经网络（Recurrent Neural Network，RNN），能够使用历史信息作为输入、包含环和自重复的网络：![](inbox/Pasted%20image%2020250823024029.png)

**RNN及LSTM具有捕捉时序信息，但是存在两个缺陷**：
- 长记忆丢失：序列按顺序读入，距离变长、内存有限
- 串行计算，效率低下：无法很好利用GPU

利用了CV的注意力机制（Attention），在NLP领域做出了基于注意力机制的神经网络--Transformer，成为LLM、深度学习最核心的架构之一。

**注意力机制（Attention）：**
大脑🧠未来理解当前目标，对所有输入信息进行扫描，分配不同注意力分数的能力，就是**注意力机制。**

**简化的工作流程（Q，K，V模型）**
Query，Key，Value

Query：查询目标
Key：vector（““电脑”的索引/或者概括”），索引标签。**计算相关性，吸引注意力，与Q计算“注意力分数”。** 
Value：vector（“电脑”），内容本身。**被提取使用，提供内容，与“注意力分数”就行加权求和得出结果。** 

真值（Ground Truth）：模型训练的目标答案

在**训练**一个翻译模型时，模型内部的**注意力机制**会利用 **Key** 和 **Value** 去处理输入的句子，得出一个翻译结果（预测）。然后，模型会把这个结果和数据集里的**真值（Ground Truth）进行比较。如果发现有差距，模型就会调整自己内部的所有参数，包括那些用来生成  Key** 和 **Value** 的网络层，以便下次能做出更接近**真值**的预测。

### 2. 深入理解注意力机制
如何计算得处注意力分数？

 **Step1:测量相似度，计算单个词Score**
词向量表示语义信息，距离远近表示词义接近。通过欧式距离来衡量词向量的相似性，此处使用点积计算。
$$
v·w = \sum_{i}v_iw_i
$$
语义相似：点积>0，语义不相似：点积<0。

**Step2:计算一系列的原始Score**
计算Query与所有词的Key向量进行矩阵点积计算，得到x为Q与所有词Key（包含自己）的相关性。此处是一一对应，而非求和结果。
$$
x = qK^T
$$
```
**举例：** 假设我们要理解句子 "The cat sat" 中 "cat" 这个词。
Q_cat 会分别和 K_The、K_cat、K_sat 进行点积。
得到三个原始分数：
e1 = Q_cat \cdot K_The (比如得到 12.5)  
e2 = Q_cat \cdot K_cat (比如得到 30.8)
e3 = Q_cat \cdot K_sat (比如得到 25.1)

这些分数 `[12.5, 30.8, 25.1]` 直观地反映了相关性，分数越高，相关性越强。
```

**Step3:稳定训练技巧——缩放**
原因：所有向量的d维度很高，点积计算结果会特别大，导致过大树枝对于后续的（Softmax）带来困难，导致梯度（模型学习修正信号）变得极小，不利于模型训练
解决方案：增加缩放因子
$$ \sqrt{d_k} = \sqrt{64} = 8 $$
dk为k的维度
把上一部计算的e1、e2、e3进行scaled。
```
e1_scaled = 12.5 / 8 = 1.56
e2_scaled = 30.8 / 8 = 3.85
e3_scaled = 25.1 / 8 = 3.14
```
为了利于模型稳定学习

**Step4:转换为最终权重--Softmax函数/归一化处理**
最终结果目的是“注意力权重”，必须满足**正数**和**所有权加起来为1**。百分比计算使用。
此处使用了自然常数e的指数函数去解决这两个问题。
$$ \text{softmax}(x)_i = \frac{e^{xi}}{\sum_{j}e^{x_j}} $$
**Step5:最终的attention公式**
$$
attention(Q,K,V) = softmax(\frac{QK^T}{√d_k})v
$$

### 3. 注意力机制的实现
```python
'''注意力计算函数'''
def attention(query, key, value, dropout=None):
    '''
    args:
    query: 查询值矩阵
    key: 键值矩阵
    value: 真值矩阵
    '''
    # 获取键向量的维度，键向量的维度和值向量的维度相同
    d_k = query.size(-1) 
    # 计算Q与K的内积并除以根号dk
    # transpose——相当于转置
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    # Softmax
    p_attn = scores.softmax(dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
        # 采样
     # 根据计算结果对value进行加权求和
    return torch.matmul(p_attn, value), p_attn

```
其中，dropout是做了p_attn的矩阵权重暂时归零，防止”过拟合“（Overfitting）和保证好的鲁棒性

return：结果、矩阵
结果中的“目标词”不仅仅存在value结果，还存在一个包含了上下文学习的（矩阵信息）。

### 4. 自注意力
Self-Attention

注意力完全发生在输入的序列内部，一个词在自己句子内部的注意力。而不是输入一个Q，在另外提供的KV中计算注意力。

Self-Attention会定义每个词的权重矩阵（Weight Matrix）；
$$
W^Q,W^K,W^V
$$
这三者会各自负责把词向量x转化为适配自己对应Query，Key，Value的向量。由一个输入，得出QKV。

作用：
- 真正的上下文感知：句子中任意两个词可以直接计算关系，不受距离限制，解决了LSTM的问题。
- 高并行运算：能够同时让所有词进行注意力计算，而不是仅计算一个词
- Transformer基石：由多个Self-Attention+前反馈网络组成

意义：Attention是基于LSTM串行架构为核心+注意力模块为辅助的上下文理解；Self-Attention是注意力模块为核心构建的上下文理解。

>计算机对一个句子的阅读能力
>**Word2Vec**
> 静态翻译，一个词一个向量值，无法理解上下文解决多义词问题
>**ELMo**
>实现句子内部感知上下文，通过上下文去生成向量，**动态向量**。局限性是超长LSTM理解
>**Attention**
>Encoder-Decoder 架构
>翻译工作，目标词去聚焦源头的那一块内容实现翻译
>**Self-Attention**
>不使用LSTM架构，该用并行架构。

### 5. 掩码自注意力
Mask Self-Attention

掩码作用：遮蔽掉一些特定位置的token，模型在学习过程中，会忽略掉遮蔽的token。核心在于不让模型看到未来信息。只能通过之前的token去预测下一个token。

掩码注意力机制的实现：

### 6. 多头注意力

## 二、Encoder-Decoder
---
### 1. Seq2Seq模型

### 2. 前馈神经网络

### 3. 层归一化

### 4. 残差连接

### 5. Encoder

### 6. Decoder

## 三、搭建一个Transformer
---

### 1. Embedding层

### 2. 位置编码

### 3. 一个完整的Transformer
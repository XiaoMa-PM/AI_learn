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

**掩码作用**：遮蔽掉一些特定位置的token，模型在学习过程中，会忽略掉遮蔽的token。核心在于不让模型看到未来信息。只能通过之前的token去预测下一个token。

**掩码注意力机制的实现：**
计算完QK^T 后，送入Softmax之前会得到“原始注意力分数”矩阵`Scores。`

```
我 爱 你 (Key) 
我 [10, 8, 5] 
爱 [ 9, 12, 11] 
你 [ 4, 7, 9] 
(Query)
```

生成一个掩码矩阵`mask`
```
我 爱 你 
我 [ 0, -∞, -∞] 
爱 [ 0, 0, -∞] 
你 [ 0, 0, 0]
```

实现方式：送入Softmax之前进行分数矩与掩码矩阵相加。
`Scores_masked=Scores+mask`

最后通过Softmax，把`-∞`处理为0则得到了：
```
我 爱 你 我 [1.0, 0.0, 0.0] // “我”只能100%关注自己 
爱 [0.2, 0.8, 0.0] // “爱”的注意力分配给了“我”和“爱” 
你 [0.1, 0.4, 0.5] // “你”的注意力分配给了所有人
```

**该方法应用于Decoder部分，其关键在于实现并行计算序列预测任务**

### 6. 多头注意力
Multi-Head Attention

之前的的自注意力缩放点积、QK^T、Mask、Score的完整流程属于单头注意力，而多头注意力可以理解为“团队协作”机制。

自注意力机制解决了并行问题，但是其中还存在单头只能拟合**一种相关关系**。（如语法一类，无法拟合其他关系）这块理解为进行**自注意力点积**的时候，只能**允许一种关系的自注意力点积**。

![](inbox/Pasted%20image%2020250828104226.png)
>图源：论文：Attention Is All Need
>图中两层为两个注意力头对同一段语句序列进行自注意力计算的结果。可以看到，对于不同的注意力头，能够拟合不同层次的相关信息。

“假设模型维度的`d_model=512`，头数为`h=8`，准备8组独立的W^qW^kW^v，输入词向量X分别乘以这8组举证，得到矩阵后，每一组降维为64。

**模型维度**：`d_model`由人工确定，根据性能与成本去衡量。
在Input Embedding中，一个单词（Token）输入后会转化为一个向量，其长度（维度）即为`d_model`。

代码如下：
```python
import torch.nn as nn
import torch

'''
该项目先采用了PyTorch的nn.Module去改造。
准备工作：__init__函数（构造函数）构建方法，只运行一次
工作阶段：forward函数 定义数据流过运行，多次调用

super().__init__()调用父类nn.Module的方法
后续会调用其中的数据类型作为使用

args.dim 模型的总维度
args.n_heads 模型的头数
self.head_dim 每个头的维度
args.n_embd ？

nn.Linear()PyTorch创建全连接层命令，即权重矩Wqkv,Wo用于容纳所有头的结果
nn.Linear(input_tensor,output_tensor,bias=True)输入张量、输出张量、偏置
'''

'''多头自注意力计算模块'''
class MultiHeadAttention(nn.Module): 

    def __init__(self, args: ModelArgs, is_causal=False):
        # 构造函数
        # args: 配置对象
        super().__init__()
        # assert 安全检查整除。隐藏层维度必须是头数的整数倍，因为后面我们会将输入拆成头数个矩阵
        assert args.dim % args.n_heads == 0
        # 每个头的维度，等于模型维度除以头的总数。
        self.head_dim = args.dim // args.n_heads
        self.n_heads = args.n_heads

        # Wq, Wk, Wv 参数矩阵，每个参数矩阵为 n_embd x dim
        # 这里通过三个组合矩阵来代替了n个参数矩阵的组合，其逻辑在于矩阵内积再拼接其实等同于拼接矩阵再内积，
        # 不理解的读者可以自行模拟一下，每一个线性层其实相当于n个参数矩阵的拼接
        self.wq = nn.Linear(args.n_embd, self.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.n_embd, self.n_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.n_embd, self.n_heads * self.head_dim, bias=False)
        # 输出权重矩阵，维度为 dim x dim（head_dim = dim / n_heads）
        self.wo = nn.Linear(self.n_heads * self.head_dim, args.dim, bias=False)
        # 注意力的 dropout，用于注意力权重
        self.attn_dropout = nn.Dropout(args.dropout)
        # 残差连接的 dropout，用于最终输出用的
        self.resid_dropout = nn.Dropout(args.dropout)
        self.is_causal = is_causal

        # 创建一个上三角矩阵，用于遮蔽未来信息
        # 注意，因为是多头注意力，Mask 矩阵比之前我们定义的多一个维度。原因：利用广播机制。
        if is_causal:
            mask = torch.full((1, 1, args.max_seq_len, args.max_seq_len), float("-inf"))
            mask = torch.triu(mask, diagonal=1)
            # 注册为模型的缓冲区
            self.register_buffer("mask", mask)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):

        # 获取批次大小和序列长度，[batch_size, seq_len, dim]
        bsz, seqlen, _ = q.shape

        # 计算查询（Q）、键（K）、值（V）,输入通过参数矩阵层，维度为 (B, T, n_embed) x (n_embed, dim) -> (B, T, dim)
        xq, xk, xv = self.wq(q), self.wk(k), self.wv(v)

        # 将 Q、K、V 拆分成多头，维度为 (B, T, n_head, dim // n_head)，然后交换维度，变成 (B, n_head, T, dim // n_head)
        # 因为在注意力计算中我们是取了后两个维度参与计算
        # 为什么要先按B*T*n_head*C//n_head展开再互换1、2维度而不是直接按注意力输入展开，是因为view的展开方式是直接把输入全部排开，
        # 然后按要求构造，可以发现只有上述操作能够实现我们将每个头对应部分取出来的目标
        '''
        xq.view()类似于numpy的.reshape(),[bsz,seqlen,512]-->[bsz,sqlen,8,64]
        xq.transpose(1,2)交换维度，把第1和第2维交换，[bsz,sqlen,8,64]-->[bsz,8,sqlen,64]
        此处转换是因为后续的matmul处理张量使用，B为批次、nh为头数、T为句子长度、hs头的维度）
        “我们有 `bsz` 批句子，每句话我们都派了 `8` 个专家（头）去分析，每个专家都把每个词 `T` 转换成了一个 `64` 维的向量。”
        '''
        xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_heads, self.head_dim)
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)
        # 注意力计算
        # 计算 QK^T / sqrt(d_k)，维度为 (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        '''
        transpose（2，3）是对每个头的[seqlen,64]进行矩阵转置，变成[64，seplen]
        而计算是针对整个[bsz，8,...]展开的，一次性计算了8个头的分数
        torch.matmul（a，b），a=[A,B,C,D],计算中A，B归属批次，而C，D才是计算的矩阵内容
        计算最后输出的scores就为[T，T]
        /math.sqrt(self.head_dim)即为降温处理（即前文提到的缩放8）,防止Softmax之后，传回学习导致梯度过小无法进一步优化
        '''
        scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)
        # 掩码自注意力必须有注意力掩码
        if self.is_causal:
            assert hasattr(self, 'mask')
            # 这里截取到序列长度，因为有些序列可能比 max_seq_len 短
            scores = scores + self.mask[:, :, :seqlen, :seqlen]
        # 计算 softmax，维度为 (B, nh, T, T)
        '''
        F.softmax()的F.属于计算方式引用，从F中调出sofmax()，F归属于torch里的，这里区分nn.Module，nn.Module归属于零件，调用需要对内部参数进行定义，而F属于直接的function工具，原本为import torch.nn.functional as F，而在torch中其本被as F了，不需要进行引入定义。
        scores.float()将张量数据类型转换为更加精确、范围更大的float32，而原有为float16会缺乏精度面，并且e^x后会出现无穷大结果inf。此处是转化float32后进行softmax计算。
        .type_as(xq)转换为原有xq的数据类型float16.
        以上注释的float16仅为假设，float32的根本目的是更加精确与计算稳定。
        '''
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        # 做 Dropout
        '''
        随机遮住了一部分的scores（vq*vk）
        最后由scores*wv去获取内容匹配上
        '''
        scores = self.attn_dropout(scores)
        # V * Score，维度为(B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        output = torch.matmul(scores, xv)
        # 恢复时间维度并合并头。
        # 将多头的结果拼接起来, 先交换维度为 (B, T, n_head, dim // n_head)，再拼接成 (B, T, n_head * dim // n_head)
        # contiguous 函数用于重新开辟一块新内存存储，因为Pytorch设置先transpose再view会报错，
        # 因为view直接基于底层存储得到，然而transpose并不会改变底层存储，因此需要额外存储
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)

        # 最终投影回残差流。
        '''
        scores*wv的结果output是[B，T，512]，里面容纳了8个头的（专家）结果，无法识别哪一个的权重更加重要，故在此加入了wo去分配不同专家的权重。
        wo是[512,512,bias=False]，其计算结果是[B，T，512]
        '''
        output = self.wo(output)
        output = self.resid_dropout(output)
        return output

```
自问自答：
`=F.softmax`和`args.dim`这类型的差别，为什么会有`F.`。
`= F.softmax`属于pytorch的固有函数公式，`__init__`是直接调用pytorch里的内容进行内部参数定义，与前者区别是后者需要进行配置后才能使用的工具。

为什么Wv是在Softmax和dropout之后才去乘以scores？
因为scores是权重，来源于Wq与Wk的对应计算，算完了相对于是每个词对应的权重，但是没有获取到对应的向量，与其Wv相乘后就是一个组合动作。整个Self-Attention目的是解决“如何组合配置Value”。

为什么最后的output还要放Wo里面，这一步是如何计算的？
Wo之前是output`[B,T,512]`，其内容虽然在计算过程是把8头分开后并行运算，但是结果是粗暴拼凑，没有解决每个头的权重问题，可以理解为8个专家解析语言的角度不同，解析后，还需要Wo对8个专家的结果进行配重后整理成结果。

完整流程梳理：
`__init__`是前期的类准备，准备好运行所需的工具/方法等。
1. 核心⚙️：WqWkWvWo四个`nn.Linear`层，是四个科学系权重矩阵。
2. 辅助🔧：Dropout、Causal Mask
3. 参数设置：计算运行所用的通道，此处维度受到算力影响，分配X头X维
`forward`数据处理流水线
4. 核心在于把一批句子的（q,k,v）拉入流向wq,wk,wv，之后把（q,k,v）512维，拆成8\*64；
5. 继之让每一个64的（q,k,v）进行注意力计算，得出原始注意力分数矩阵（在此之前进行了维度切换，在此输出的矩阵中还做了降温处理，防止后续softmax与梯度优化出现问题），这个步骤选择加入Masking可以实现并行处理（GPT）；
6. Softmax把分数矩阵归一化为概率权重，获得了“关注度”分布；Dropout解决泛化问题，为了传回后学习使用；
7. 最后，进行了Wo去实现对每个头的权重分配，求和，得出结果。
最后结果是：一批新的、每个词都包含了来自8个视角去分析上下文的信息张量

**以上学习内容为MultiHeadAttention，正常处理的下一步就是前馈网络层**，此处插入补充学习Encoder-Decoder
## 二、Encoder-Decoder
---
### 1. Seq2Seq模型
Sequence-to-Sequence，序列到序列，一种经典的NLP任务。
**定义了解决一类核心NLP问题的通用范式**，注意力机制使用的地方

**Seq2Seq要解决什么问题？**
其涵盖任务：
- **机器翻译（Machine Translation）**：输入一个英文句子序列->输出一个中文句子序列
- 文本摘要（Text Summarization）：输入一篇长文章序列->输出一个短摘要序列
- 对话机器人（Chatbots）：输入一个问题序列->输出一个回答序列
- 语言识别（Speech Recognition）：输入一段音频的特征序列->输出一段文字序列
这些任务共同点：输入序列和输出序列**长度不一的相等**，这种特性不能使用一对一的分类模型去解决

**S2S的核心架构--Encoder-Decoder**
Encoder：
- 任务：读取并理解整个输入序列，捕捉器核心内容
- 工作模式：RNN（循环神经网络），LSTM、GRU
- 产出：对整句话最终理解后，压缩成固定长度的向量，即上下文向量（Context Vector）
Decoder：
- 任务：对Encoder传入的“理解”，一个词一个词生成输出序列
- 工作模式：RNN，LSTM
	- 会先根据“理解”生成一个词a1，再根据a1+“理解”生成a2，以此类推，直到结束
- 产出：序列不同之前的句子
![](inbox/Pasted%20image%2020250830015353.png)

**Seq2Seq模型--信息瓶颈**
Information Bottleneck

这里经典的S2S使用RNN、LSTM，就出现了超长句子，相对于短的句子，其输出的唯一、固定长度的上下文向量，容易遗忘前文内容。也就是提到过的LSTM的缺点。[第一章 NLP基础概念LSTM](../chapter1/第一章%20NLP基础概念.md#^8d17f8)

**Attention的引入**
核心思路：Decoder为什么要去看被Encoder压缩到最后的“理解”（最后的Encoder Layer），可以生成每个词的同时，回顾“理解”+a1...+每一层Encoder Layer
新的工作模式（Seq2Seq with Attention）：
1. Encoder：提供输出的最后的隐藏层h_n同时，也把每一次的隐藏层全部汇总一起为`[h_1,h_2,...,h_n]`，形成笔记集合
2. Decoder：生成每一个词的时候，执行一次Attention计算：
	- 当前内部状态设置为Q
	- 把Encoder的“笔记集合”设置为K、V
	- 计算后，会得到注意力权重分布，告诉它应该把“注意力”放在原文中哪几个词上
	- 再将权重加权后的上下文信息与自己当前记忆结合起来，再去生成下一个词。
举例：生成“建筑”的词，不仅看了输入的句子的摘要（Encoder最终结果），还用Attention看了一下输入句子中和“建筑”相关的词，进行Wq\*Wk得到权重后，获取output结果，再去与自己生的上一个词走LSTM的路程。

作用：打破了处理很长的输入序列，实现质的飞跃。
总结：把Seq2Seq模型中原有的RNN/LSTM换成了Self-Attention发动机

### 2. 前馈神经网络
Feed Forward Neural Network，FNN

### 3. 层归一化

### 4. 残差连接

### 5. Encoder

### 6. Decoder

## 三、搭建一个Transformer
---

### 1. Embedding层

### 2. 位置编码

### 3. 一个完整的Transformer
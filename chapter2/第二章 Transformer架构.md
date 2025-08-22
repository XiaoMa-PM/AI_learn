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
大脑🧠未来理解当前目标，对所有输入信息进行扫描，分配不同权重的能力，就是**注意力机制。**


### 2. 深入理解注意力机制

### 3. 注意力机制的实现

### 4. 自注意力

### 5. 掩码自注意力

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
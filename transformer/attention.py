"""
1、Attention 到底在干什么？
假设有一句话 I love machine learning   
经过 embedding 后:
I       → x1
love    → x2
machine → x3
learning→ x4
可以表示 X
shape = [B, N, D]  
B = 1      batch size
N = 4      token 数
D = 8      embedding dimension
Attention 要解决的问题是：每一个 token 应该关注其他哪些 token?
比如machine和learning  可能比machine和I 更相关  所以 Attention 本质上是在计算token 与 token 之间的相关性

2、Q / K / V 是什么？【下面先不讲batch】
输入x 经过三个不同的 Linear  (q,k,v是下标的意思  wq就是权重)
Q = XWq  对应代码 Q = self.q_proj(x)
K = XWk  对应代码 K = self.k_proj(x)
V = XWv  对应代码 V = self.v_proj(x)
x是[N, D]  Wq,Wk,Wv都是[D, D] 对应得到的 Q,K,V都是[N, D]
Q就是Query——我现在想找什么？   (eg:“我现在需要找和 machine 相关的信息”)
K就是Key——我这里有什么信息，可以被匹配？    (eg:每个 token 提供一个“标签”)
V就是Value——如果匹配到了，我真正拿走什么信息？  (eg:真正传递的信息)
Q × K 负责计算“我应该关注谁”   Attention × V  负责把我关注的信息拿过来 
QKᵀ —— 我们希望每个 token和所有 token计算相似度  所以需要QKᵀ   Q[N,D] Kᵀ[D,N]  相乘得到[N,N] 也就是每个token和所有token的相似度 [一行就是一个Token和其他token的相似度] 或者称为每一个 token 对每一个 token 的注意力分数
scores = Q @ K.transpose(-2, -1)  交换张量 K 的最后两个维度 负数维度索引规则——-1 表示最后一维  -2 表示倒数第二维
score 的本质是相似度：scores[i,j] = "query i 对 key j 的相关程度"

3、为什么要除以 √d
得到 scores = Q @ K.transpose(-2, -1)   不能直接 Softmax
还需要 scores = scores / math.sqrt(d)  d就是Key的维度(特征维度) 为什么要除？——防止 QKᵀ 的数值随着维度增加而变得过大，从而让 Softmax 过于尖锐 相当于一个scale 然后再softmax再乘以V

Softmax 假设分数里面其中一行是[2.0, 1.0, 0.1, -1.0]  经过softmax [0.65, 0.24, 0.10, 0.01]  这些数字的意思就是当前 token 应该把多少注意力放到每个 token 上
attention = torch.softmax(scores, dim=-1)  dim=-1表示在最后一个维度上做 Softmax  也就是每一行进行softmax  最后一个维度表示每个 Query 对所有 Key 的分数
最后再跟V乘   [N,N]X[N,D]=[N,D]  out = attention @ V 


4、Multi-Head 的思想是 不要让一个 Attention 头负责学习所有关系，而是拆成多个 head，让不同 head 学习不同的关系
        Input
          │
 ┌────────┼────────┐
 ↓        ↓        ↓
Head 1   Head 2   Head 3
 ↓        ↓        ↓
关系A     关系B     关系C
 └────────┼────────┘
          ↓
       Concat
          ↓
      Linear
    
假设B = 2  N = 16  D = 64  我们设置 num_heads = 4 那么：head_dim = 64 / 4 = 16  
分头是在x分别计算出Q,K,V之后，对Q,K,V的最后一维D分，Q,K,V都要分头  分的是特征维度 D 
过程是首先x分别计算得到Q,K,V  然后对Q,K,V都分头，然后每个头Q1,K1,V2独立计算Attention 然后合并所有头输出线性层
也就是x [2,16,64]  计算Q,K,V 还是[2,16,64] 然后对Q,K,V 对最后一维64拆成[4,16] 也就是：
Q: [2, 16, 64]
→ reshape: [2, 16, 4, 16]
→ transpose: [2, 4, 16, 16]  [B, H, N, head_dim]   H就是头数  N是token数量
每个头
Q_h.shape = [B, N, d] = [2, 16, 16]
K_h.shape = [B, N, d] = [2, 16, 16]
V_h.shape = [B, N, d] = [2, 16, 16]
后面还是一样计算
最后每个头计算合并后先transpose把维度放回去然后reshap变回[2, 16, 64]  在经过linear输出
分头并不是把 16 个 token 分成 4 组，而是把每个 token 的 64 维特征拆成 4 组  

分头分的是每个 token 的特征维度（D），不是 token 数量（N）

"""
import math

import torch
import torch.nn as nn


class SelfAttention(nn.Module):

    def __init__(self, embed_dim):
        super().__init__()

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.embed_dim = embed_dim

    def forward(self, x):

        # x: [B, N, D]

        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        scores = Q @ K.transpose(-2,-1)

        scores = scores / math.sqrt(self.embed_dim)

        attention = torch.softmax(scores,dim=-1) 
        print(attention.sum(dim=-1))

        out = attention @ V

        print("Q:", Q.shape)
        print("K:", K.shape)
        print("V:", V.shape)
        print("scores:", scores.shape)
        print("attention:", attention.shape)
        print("out:", out.shape)

        return out


class MultiHeadAttention(nn.Module):

    def __init__(self, embed_dim, num_heads):
        super().__init__()

        assert embed_dim % num_heads == 0

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x,attention_mask=None):

        B, N, D = x.shape   # 这里 N 就是序列长度 L

        Q = self.q_proj(x)  # [B, N, D]
        K = self.k_proj(x)  # [B, N, D]
        V = self.v_proj(x)  # [B, N, D]

        # [B, N, D] ->[B, N, H, head_dim]->[B, H, N, head_dim]
        Q_h = Q.reshape(B,N,self.num_heads,self.head_dim).transpose(1,2)
        K_h = K.reshape(B,N,self.num_heads,self.head_dim).transpose(1,2)
        V_h = V.reshape(B,N,self.num_heads,self.head_dim).transpose(1,2)


        scores = Q_h @ K_h.transpose(-2,-1)
        """
        因为 batch 维 B 和 head 维 H 在矩阵乘法里是"批量"的，不参与矩阵乘的形状推导
        Q_h:                  [B, H, N, head_dim]
        K_h.transpose(-2,-1): [B, H, head_dim, N]
        按 matmul 规则，前两维 [B, H] 是批量维，逐对独立处理；后两维 [N, head_dim] @ [head_dim, N] = [N, N] 才是真正做矩阵乘的部分
        scores: [B, H, N, N]
                 └┬─┘ └──┬──┘
                 批量维 每个(b,h)的 [N,N] 分数矩阵
        B=2, H=4 时，批量维有 2 × 4 = 8 个组合：
        (b=0, h=0) → Q_h[0,0]: [N, head_dim]  @  K_h[0,0]ᵀ: [head_dim, N]  →  scores[0,0]: [N, N]
        (b=0, h=1) → Q_h[0,1]: [N, head_dim]  @  K_h[0,1]ᵀ: [head_dim, N]  →  scores[0,1]: [N, N]
        (b=0, h=2) → ...                                                    →  scores[0,2]: [N, N]
        (b=0, h=3) → ...                                                    →  scores[0,3]: [N, N]
        (b=1, h=0) → Q_h[1,0]: [N, head_dim]  @  K_h[1,0]ᵀ: [head_dim, N]  →  scores[1,0]: [N, N]
        (b=1, h=1) → ...                                                    →  scores[1,1]: [N, N]
        (b=1, h=2) → ...                                                    →  scores[1,2]: [N, N]
        (b=1, h=3) → ...                                                    →  scores[1,3]: [N, N]
        一共 8 次独立的 [N, head_dim] @ [head_dim, N] = [N, N] 矩阵乘，每次只用 Q_h[b,h] 和 K_h[b,h]，乘出来的 [N,N] 放进 scores[b,h]。
        [B, H, N,  N]
        │  │   │   └── Key 位置：这一行分数对应"每个 key 被打了多少分"
        │  │   └────── Query 位置：这一行是"哪个 query token 打出来的分"
        │  └────────── Head 编号
        └───────────── Batch 编号
        具体地，scores[b, h, i, j] = "第 b 条样本、第 h 个头里，query 位置 i 对 key 位置 j 的相似度分数"。
        """
        # 除以 sqrt(head_dim)，是为了抑制点积随维度增大而变大的趋势，避免softmax过于尖锐、梯度变差
        scores = scores / math.sqrt(self.head_dim)  # 应该是 self.head_dim 因为分头了 现在的特征维度是head_dim 

        """
        PyTorch 索引式增维（广播准备）。逐个拆：
        attention_mask          # [B, L]      dtype=bool
        attention_mask[:, None, None, :]  方括号里 4 个位置对应结果张量的 4 个维度。None 表示"在这里插入一个长度为 1 的新维度"，冒号 : 表示"这一维原样保留"
        为什么这么扩？为了和 scores: [B, H, L, L] 对齐。
        PyTorch 的 masked_fill 需要 mask 和 scores 形状一致或可广播。广播规则是：从最后一维往前对齐，每一维要么相等，要么其中一个是 1。

        """
        if attention_mask is not None:
            key_mask = attention_mask[:, None, None, :]   # [B,1,1,N]
            scores = scores.masked_fill(~key_mask, float("-inf"))
            """
            torch.Tensor.masked_fill(mask, value) 的语义：凡是 mask 为 True 的位置，把原值替换成 value；  False 的位置保持不变
            ~ 是逻辑取反（对 bool 张量）。key_mask 里 True 表示有效 token，False 表示 PAD 
            key_mask : [True, True, False]    # 前两个有效，第三个 PAD
            ~key_mask: [False, False, True]   # True 的位置 = PAD 位置
            凡是 PAD 位置（~key_mask 为 True），把分数填成 -inf；有效位置分数不变
            PyTorch 的做法是：先把 mask 按广播规则"复制展开"成 [B,H,N,N]，再逐元素比。
            scores  : [B,  H,  N,  N]
            key_mask: [B,  1,  1,  N]
                        │   │   │   │
            第0维: B vs B  → 相等 ✓
            第1维: H vs 1  → 有1，mask 的第1维复制 H 份 ✓
            第2维: N vs 1  → 有1，mask 的第2维复制 N 份 ✓
            第3维: N vs N  → 相等 ✓
            展开后 key_mask 逻辑上变成 [B, H, N, N]
            设 B=1, H=2, N=3，简化到最小。mask 原始：
                key_mask = torch.tensor([[[[True, True, False]]]])   # [1,1,1,3]
                把它写成"格子"：
                key_mask[0,0,0,:] = [T, T, F]     ← 只有这一行数据
                现在要展开成 [1,2,3,3]。展开规则：
                第 1 维 1→2：把这一份复制 2 份（对应 H=2 个 head）
                第 2 维 1→3：把这一份复制 3 份（对应 N=3 个 query 行）
                展开后，key_mask[0,h,i,:] 对所有 h=0,1 和 i=0,1,2 都等于 [T,T,F]：
                展开后的 key_mask[0] （形状 [H=2, N=3, N=3]）:
                h=0:
                query0: [T, T, F]
                query1: [T, T, F]
                query2: [T, T, F]
                h=1:
                query0: [T, T, F]
                query1: [T, T, F]
                query2: [T, T, F]
                6 行，内容完全一样，因为原始只有 1 行，广播把它复制成了 6 行。
            每个 token 作为 query，和所有 token（作为 key）算分数得到 [L,L]；attention_mask 按 token 顺序标记每个 key 位置是否 PAD，扩成 [B,1,1,L] 后把 PAD 对应的整列在所有 query 行上填 -inf；
            softmax 后 PAD 列权重为 0，有效 token 不再读到 PAD 信息。PAD 行本身保留（输出非零），留待 pooling 阶段再用 mask 排除。
            """
        attention = torch.softmax(scores,dim=-1)  # Attention的softmax沿Key/token维度，不是特征维度。

        out = attention @ V_h

        # [B,H,N,d] →[B,N,H,d]→[B,N,D]
        out = out.transpose(1,2).reshape(B,N,D)

        out = self.out_proj(out)  # 它不仅是为了恢复维度，更是让拼接后的不同 Head 的输出能够进一步进行可学习的特征混合

        return out


if __name__ == "__main__":
    x = torch.randn(2, 8, 64)

    single_head = SelfAttention(embed_dim=64)
    multi_head = MultiHeadAttention(embed_dim=64, num_heads=4)

    out_single = single_head(x)
    out_multi = multi_head(x)

    print("input:", x.shape)
    print("single-head output:", out_single.shape)
    print("multi-head output:", out_multi.shape)



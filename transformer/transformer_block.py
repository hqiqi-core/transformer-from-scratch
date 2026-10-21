"""
Transformer Block 的结构 
Input X-> Multi-Head Self-Attention ->Residual Add + LayerNorm ->Feed-Forward Network ->Residual Add + LayerNorm ->output
这是一种常见的 Post-LayerNorm 结构 
Post-LN：先执行子层，再残差相加，最后 LayerNorm。
Pre-LN：先对输入做 LayerNorm，再执行子层，并进行残差相加。

1、Residual Connection：为什么要加回输入？
Y=X+Attention(X)   Attention 负责学习 token 之间的信息交互，残差连接则让原始输入信息有一条直接传递的路径，也有助于梯度传播
前提是相加的两个 Tensor 形状兼容。你当前的 Attention 输入和输出都是 [B,N,D]，因此可以直接相加
2、LayerNorm：它归一化什么？
输入形状为x.shape == [2, 16, 64]  使用nn.LayerNorm(64) 意味着对每个样本、每个 token 的最后 64 维特征进行归一化，而不是把 16 个 token 混在一起计算均值和方差 LayerNorm 不改变 Tensor 形状
3、FFN：Attention 之后为什么还需要 MLP？
Attention 负责不同 token 之间的信息交互；FFN 则对每个 token 的特征独立进行变换
FFN(x)=W2​ReLU(W1​x+b1​)+b2​
注意，FFN 的 Linear 层作用于最后一维，所以 [B,N,64] 会变成 [B,N,256]，再回到 [B,N,64]。它不会把 token 数量 N 改掉


"""
import math

import torch
import torch.nn as nn

from attention import MultiHeadAttention
    
class FeedForward(nn.Module):
    def __init__(self, embed_dim, hidden_dim):
        super().__init__()
        self.embed_dim=embed_dim
        self.hidden_dim = hidden_dim
        self.linear1 = nn.Linear(self.embed_dim,self.hidden_dim)
        self.linear2 = nn.Linear(self.hidden_dim,self.embed_dim)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.linear2(self.relu(self.linear1(x)))
        return x

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden_dim):
        super().__init__()
        self.embed_dim=embed_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.MultiHeadAttention = MultiHeadAttention(self.embed_dim,self.num_heads)
        # TODO: 定义 MultiHeadAttention
        # TODO: 定义第一个 LayerNorm
        self.layerNorm1 = nn.LayerNorm(self.embed_dim)
        # TODO: 定义 FeedForward
        self.FeedForward = FeedForward(self.embed_dim,self.hidden_dim)
        # TODO: 定义第二个 LayerNorm
        self.layerNorm2 = nn.LayerNorm(self.embed_dim)

    def forward(self, x):
        # TODO 1: Attention
        attention = self.MultiHeadAttention(x)
        # TODO 2: 残差连接 + LayerNorm
        x = self.layerNorm1(x + attention)
        # TODO 3: FeedForward
        ffn = self.FeedForward(x)
        # TODO 4: 残差连接 + LayerNorm
        x = self.layerNorm2(x + ffn)
        return x

    
if __name__ == "__main__":
    x = torch.randn(2, 64, 64)

    model = TransformerBlock(
        embed_dim=64,
        num_heads=4,
        hidden_dim=256,
    )

    out = model(x)
    loss = out.mean()
    loss.backward()
    print(f"loss:{loss.item():.4f}")

    print("input:", x.shape)
    print("output:", out.shape)

    assert out.shape == x.shape
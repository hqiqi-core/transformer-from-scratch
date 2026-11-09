
import math

import torch
import torch.nn as nn
class CrossAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()

        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(
        self,
        query: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # 1. 分别取 query 和 context 的序列长度
        batch_size, query_length, _ = query.shape  # （Lq）：query 序列多长
        context_length = context.shape[1]  # （Lc）：context 序列多长

        # 2. Q 投影 query；K/V 投影 context  query 是"我想找什么"  context 是"我有什么信息"
        Q = self.q_proj(query)      # [B, Lq, D]  query → Q
        K = self.k_proj(context)    # [B, Lc, D]  context → K   Q 和 K 算匹配  权重作用在 V 上
        V = self.v_proj(context)    # [B, Lc, D]  context → V

        # 3. 分头：Q 用 query_length，K/V 用 context_length
        Q_h = Q.reshape(
            batch_size, query_length, self.num_heads, self.head_dim
        ).transpose(1, 2)           # [B, H, Lq, dh]

        K_h = K.reshape(
            batch_size, context_length, self.num_heads, self.head_dim
        ).transpose(1, 2)           # [B, H, Lc, dh]

        V_h = V.reshape(
            batch_size, context_length, self.num_heads, self.head_dim
        ).transpose(1, 2)           # [B, H, Lc, dh]

        # 4. 缩放点积与 softmax
        scores = Q_h @ K_h.transpose(-2, -1)   # [B,H,Lq,dh] @ [B,H,dh,Lc] = [B,H,Lq,Lc]
        """
              Lc 列（context 的 token）
              ┌─────────────┐
        Lq 行 │  scores     │   ← 每个 query token 对所有 context token 的分数
              └─────────────┘
        scores[b,h,i,j] = "第 i 个 query token 对第 j 个 context token 的相似度"
        """
        scores = scores / math.sqrt(self.head_dim)
        # 有 mask 时：检查 + 在 softmax 前屏蔽 padding 的 key 列
        if context_mask is not None:
            if context_mask.dtype != torch.bool:
                raise ValueError(
                    f"context_mask must be bool, got {context_mask.dtype}"
                )
            if context_mask.shape != (batch_size, context_length):
                raise ValueError(
                    f"context_mask shape {tuple(context_mask.shape)} "
                    f"!= (batch_size, context_length) "
                    f"= ({batch_size}, {context_length})"
                )
            if not context_mask.any(dim=-1).all():
                raise ValueError(
                    "each sample must have at least one valid context position"
                )
            valid_keys = context_mask[:, None, None, :]   # [B,1,1,Lc]
            scores = scores.masked_fill(~valid_keys, float("-inf"))

        attention_weights = torch.softmax(scores, dim=-1)   # [B, H, Lq, Lc]  softmax 沿最后一维（Lc）：每个 query token 在所有 context token 上分配权重。

        # 5. 加权读取 V
        out = attention_weights @ V_h           # [B,H,Lq,Lc] @ [B,H,Lc,dh] = [B,H,Lq,dh]  每个 query 位置得到一份从 context 读取的 dh 维信息

        # 6. 合并头并做输出投影
        out = out.transpose(1, 2).contiguous().view(   # 输出长度跟随 query，不是 context
            batch_size, query_length, self.embed_dim
        )                                       # [B, Lq, D]
        out = self.out_proj(out)

        return out, attention_weights

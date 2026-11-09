"""
query [B,Lq,D], context [B,Lc,D]
    ↓ cross_attention(query, context, context_mask)
hidden [B,Lq,D]              ← 每个 query token 读到的 context 信息
    ↓
两个 masked mean（分别对 query 和 hidden）
q_pool [B,D], h_pool [B,D]
    ↓
q_pool * h_pool              ← 逐元素乘，逐维交互 为什么用逐元素乘不用点积：点积压成一个标量，丢掉逐维信息。逐元素乘保留 [B,D] 的逐维交互，score_head 能学到"哪个维度重要"。
    ↓ score_head
scores [B]
为什么要两个 masked mean：
query 的 masked mean：只汇总有效的 query 位置（query_mask=True），padding 不参与。
hidden 的 masked mean：Cross-Attention 输出的 hidden 在 query 位置上是"读到的 context 信息"，但query 是 padding 的位置，它的 hidden 也是 padding 读出来的，不该参与汇总。

"""
import torch
import torch.nn as nn

from cross_attention import CrossAttention


class MiniPairScorer(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.cross_attention = CrossAttention(embed_dim, num_heads)
        self.score_head = nn.Linear(embed_dim, 1)

    def forward(
        self,
        query: torch.Tensor,
        context: torch.Tensor,
        query_mask: torch.Tensor,
        context_mask: torch.Tensor,
    ) -> torch.Tensor:
        # 1. Cross-Attention：每个 query token 读取 context  设计选择——Cross-Attention 内部不区分 query 的有效性，只区分 context 的有效性（哪些 key 能读）
        hidden, _ = self.cross_attention(query, context, context_mask)  # 返回 hidden: [B, Lq, D]，每个 query 位置读到的 context 信息
        # hidden: [B, Lq, D]

        # 2. 用 query_mask 做两次 masked mean
        mask = query_mask.unsqueeze(-1).to(query.dtype)   # [B, Lq, 1]
        q_pool = (query * mask).sum(dim=1) / mask.sum(dim=1)      # [B, D]
        h_pool = (hidden * mask).sum(dim=1) / mask.sum(dim=1)     # [B, D]  PAD 位置的 query token 也会从 context 读出一个 hidden，数值上非零，但语义上没意义（它是"零向量的 Q 在算"），不能参与汇总

        # 3. 逐元素乘 → 打分头 → squeeze
        scores = self.score_head(q_pool * h_pool).squeeze(-1)     # [B]
        return scores


if __name__ == "__main__":
    torch.manual_seed(0)
    model = MiniPairScorer(embed_dim=16, num_heads=2)
    model.eval()

    query = torch.randn(2, 2, 16)
    context = torch.randn(2, 4, 16)
    query_mask = torch.tensor([[True, True], [True, False]])
    context_mask = torch.tensor([
        [True, True, True, False],
        [True, True, False, False],
    ])

    with torch.no_grad():
        scores = model(query, context, query_mask, context_mask)
        assert scores.shape == (2,), scores.shape
        print("scores shape:", tuple(scores.shape))

        # 只改 query_mask=False 的位置，分数应不变
        changed_query = query.clone()
        changed_query[~query_mask] += 100.0
        changed_scores = model(changed_query, context, query_mask, context_mask)

        assert torch.allclose(scores, changed_scores, atol=1e-6), (
            scores, changed_scores
        )
        print("query padding invariance: OK")

    print("MiniPairScorer check passed.")
import torch
import torch.nn as nn
import torch.nn.functional as F

from patch_embedding import PatchEmbedding
from transformer_block import TransformerBlock


class MiniViT(nn.Module):
    def __init__(
        self,
        image_size: tuple[int, int],
        in_channels: int,
        patch_size: int,
        embed_dim: int,
        num_heads: int,
        ffn_dim: int,
        num_layers: int,
        num_classes: int,
    ):
        super().__init__()
        self.image_size = image_size
        self.patch_embedding = PatchEmbedding(in_channels, embed_dim, patch_size)

        Gh = image_size[0] // patch_size
        Gw = image_size[1] // patch_size
        N = Gh * Gw

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))  #shape 是 [1, 1, D]：[1, 1] 是"batch 维和序列维都是 1"，forward 里 .expand(B, -1, -1) 扩到 [B, 1, D]。整个 batch 共享同一个可学习参数
        self.pos_embed = nn.Parameter(  # 位置嵌入——"位置的表示"是一个可学习向量。pos_embed[i] 不是"第 i 个位置在哪"，而是"给第 i 个位置加一个什么标记，让模型能区分它"
            torch.randn(1, N + 1, embed_dim) * 0.02   # * 0.02 把标准差压到 0.02
        )  # pos_embed[1] 就是"位置 1（左上角 patch）的标记向量"。位置 1 还是左上角，但标记向量从 randn*0.02 慢慢变成"能帮助模型区分左上角"的值
        # 学习的目标：让位置向量"对任务有用"
        # 分类"猫在左上角"和"猫在右下角"可能是不同类别（或同一类但需要不同特征）。模型通过梯度调整 pos_embed，让"左上角"和"右下角"的向量在特征空间里处于合适的位置——不是随便不同，而是"不同到能帮助分类"。

        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ffn_dim)
            for _ in range(num_layers)
        ])

        self.head = nn.Linear(embed_dim, num_classes)

    def forward_features(self, images: torch.Tensor) -> torch.Tensor:
        B, C, H, W = images.shape
        if (H, W) != self.image_size:
            raise ValueError(
                f"expected image_size={self.image_size}, got ({H}, {W})"
            )

        x = self.patch_embedding(images)            # [B, N, D]
        cls = self.cls_token.expand(B, -1, -1)      # [B, 1, D]
        x = torch.cat([cls, x], dim=1)              # [B, N+1, D] 
        x = x + self.pos_embed                      # 广播  [B,N+1,D] + [1,N+1,D] → [B,N+1,D]   加位置嵌入

        for block in self.blocks:
            x = block(x)                            # [B, N+1, D]

        return x

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(images)           # [B, N+1, D]
        """
        x: [B, N+1, D] = [2, 7, 32]
        │   │    │
        │   │    └── 第 3 维：每个 token 的特征向量（32 维）
        │   └─────── 第 2 维：序列位置（7 个：0 是 CLS，1..6 是 patch）
        └─────────── 第 1 维：batch（2 张图）
        x[:, 0, :] 只要索引 0   "每张图的 CLS 表示"
        """
        return self.head(x[:, 0, :])                # [B, num_classes]


if __name__ == "__main__":
    torch.manual_seed(0)

    image_size = (8, 12)
    in_channels = 3
    patch_size = 4
    embed_dim = 32
    num_heads = 4
    ffn_dim = 64
    num_layers = 2
    num_classes = 4

    model = MiniViT(
        image_size, in_channels, patch_size,
        embed_dim, num_heads, ffn_dim, num_layers, num_classes,
    )

    images = torch.randn(2, 3, 8, 12)
    labels = torch.tensor([0, 3], dtype=torch.long)

    tokens = model.forward_features(images)
    assert tokens.shape == (2, 7, 32), tokens.shape
    print("forward_features shape:", tuple(tokens.shape))

    logits = model(images)
    assert logits.shape == (2, 4), logits.shape
    print("logits shape:", tuple(logits.shape))

    loss = F.cross_entropy(logits, labels)
    assert torch.isfinite(loss), loss
    print("loss:", loss.item())

    loss.backward()

    for name in ("cls_token", "pos_embed"):
        param = getattr(model, name)
        assert param.grad is not None, name
        assert torch.isfinite(param.grad).all(), name
    assert model.patch_embedding.proj.weight.grad is not None
    assert torch.isfinite(model.patch_embedding.proj.weight.grad).all()

    print("gradients reach cls_token, pos_embed, patch_embedding.proj.weight: OK")
    print("All MiniViT checks passed.")
import torch
import torch.nn as nn

from patchify import patchify


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels: int, embed_dim: int, patch_size: int):
        super().__init__()
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.patch_size = patch_size

        self.proj = nn.Linear(  # Linear 的输入维度是 C*P*P——一个 patch 展平后的长度。输出 embed_dim，即每个 patch 变成 D 维 token。
            in_channels * patch_size * patch_size,
            embed_dim,
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        patches = patchify(images, self.patch_size)   # [B, N, C*P*P]
        return self.proj(patches)                      # [B, N, D] nn.Linear 对 [B, N, C*P*P] 自动作用在最后一维：把每个 [C*P*P] 向量映射成 [D]，N 个 token 各自变换。Linear 不改变 N 和 B。


if __name__ == "__main__":
    torch.manual_seed(0)

    B, C, H, W, P, D = 2, 3, 8, 12, 4, 16

    pe = PatchEmbedding(in_channels=C, embed_dim=D, patch_size=P)

    # 写法 A：patchify + Linear
    images = torch.randn(B, C, H, W)
    out_a = pe(images)                                 # [B, N, D]

    N = (H // P) * (W // P)
    assert out_a.shape == (B, N, D), out_a.shape
    print("PatchEmbedding output shape:", tuple(out_a.shape))

    # 写法 B：Conv2d(kernel=P, stride=P)，塞入同一组权重
    conv = nn.Conv2d(C, D, kernel_size=P, stride=P)
    with torch.no_grad():
        conv.weight.copy_(pe.proj.weight.reshape(D, C, P, P))
        conv.bias.copy_(pe.proj.bias)

    out_b = conv(images)                               # [B, D, Gh, Gw]
    out_b = out_b.flatten(2).transpose(1, 2)           # [B, N, D]

    assert torch.allclose(out_a, out_b, atol=1e-6), (
        (out_a - out_b).abs().max().item()
    )
    print("Linear and Conv2d patch embedding are equivalent: OK")

    print("All patch embedding checks passed.")
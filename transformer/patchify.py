import torch


def patchify(images: torch.Tensor, patch_size: int) -> torch.Tensor:
    """[B,C,H,W] -> [B,N,C*P*P]，行优先（从左到右、从上到下）。"""
    B, C, H, W = images.shape
    P = patch_size

    if P <= 0 or H % P != 0 or W % P != 0:
        raise ValueError(
            f"patch_size={P} must be positive and divide H={H}, W={W}"
        )

    Gh = H // P
    Gw = W // P
    N = Gh * Gw

    x = images.reshape(B, C, Gh, P, Gw, P)      # 拆
    x = x.permute(0, 2, 4, 1, 3, 5)             # 归位
    x = x.reshape(B, N, C * P * P)              # 合并
    return x


def unpatchify(
    patches: torch.Tensor,
    patch_size: int,
    image_size: tuple[int, int],
) -> torch.Tensor:
    """[B,N,C*P*P] -> [B,C,H,W]。"""
    B, N, D = patches.shape
    P = patch_size
    H, W = image_size

    if H % P != 0 or W % P != 0:
        raise ValueError(
            f"image_size={image_size} not divisible by patch_size={P}"
        )

    Gh = H // P
    Gw = W // P

    if Gh * Gw != N:
        raise ValueError(
            f"N={N} != Gh*Gw={Gh*Gw} for image_size={image_size}, P={P}"
        )

    C = D // (P * P)
    if C * P * P != D:
        raise ValueError(f"D={D} not divisible by P*P={P*P}")

    x = patches.reshape(B, Gh, Gw, C, P, P)     # 拆
    x = x.permute(0, 3, 1, 4, 2, 5)             # 逆排列
    x = x.reshape(B, C, H, W)                   # 合并
    return x


if __name__ == "__main__":
    images = torch.arange(2 * 3 * 4 * 6).reshape(2, 3, 4, 6)
    patches = patchify(images, patch_size=2)

    assert patches.shape == (2, 6, 12), patches.shape
    print("patches shape:", tuple(patches.shape))

    # 第 3 块 = 第二行第一列 = images[0, :, 2:4, 0:2]
    assert torch.equal(patches[0, 3], images[0, :, 2:4, 0:2].reshape(-1))
    print("patch 3 maps to (row=1, col=0): OK")

    restored = unpatchify(patches, patch_size=2, image_size=(4, 6))
    assert torch.equal(restored, images)
    print("unpatchify round-trip: OK")

    print("All patchify checks passed.")
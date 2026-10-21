import torch
import torch.nn.functional as F

from tiny_transformer import TinyTransformerEncoder


def build_model():
    return TinyTransformerEncoder(
        vocab_size=32,
        max_length=16,
        d_model=32,
        num_heads=4,
        ffn_dim=64,
        num_layers=2,
    )


def test_forward_shapes_and_independent_layers():
    torch.manual_seed(0)
    model = build_model()
    model.eval()

    # 可变 B、可变 L（(4,8) 和 (2,5)）
    input_ids = torch.randint(1, 32, (4, 8))
    short_ids = torch.randint(1, 32, (2, 5))

    with torch.no_grad():
        out = model(input_ids)
        out_short = model(short_ids)

    assert out.shape == (4, 8, 32), out.shape
    assert out_short.shape == (2, 5, 32), out_short.shape

    assert len(model.blocks) == 2
    assert model.blocks[0] is not model.blocks[1]    # 两个 Block 不是同一对象（is not）
    assert (   # 两个 Block 的首个参数不是同一张量（is not）→ 参数独立
        next(model.blocks[0].parameters())
        is not next(model.blocks[1].parameters())
    )

    print("Output shape:", tuple(out.shape))
    print("Short output shape:", tuple(out_short.shape))
    print("Independent layers: OK")


def test_backward():
    torch.manual_seed(0)
    model = build_model()

    input_ids = torch.randint(1, 32, (4, 8))

    model.zero_grad(set_to_none=True)
    output = model(input_ids)
    target = torch.randn_like(output)
    loss = F.mse_loss(output, target)
    loss.backward()

    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name   # 所有 named_parameters() 都有 grad 且有限 
        assert torch.isfinite(parameter.grad).all(), name

    print("Loss:", loss.item())
    print("Backward check passed.")


if __name__ == "__main__":
    test_forward_shapes_and_independent_layers()
    test_backward()
    print("All Mini Transformer tests passed.")
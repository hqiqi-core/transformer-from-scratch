import torch

from cross_attention import CrossAttention


def test_forward_and_context_dependence():
    torch.manual_seed(42)
    model = CrossAttention(embed_dim=32, num_heads=4)
    model.eval()

    query = torch.randn(2, 6, 32)
    context = torch.randn(2, 6, 32)

    with torch.no_grad():
        output, weights = model(query, context)
        changed_output, _ = model(query, context + 0.5)

    assert output.shape == (2, 6, 32), output.shape
    assert weights.shape == (2, 4, 6, 6), weights.shape

    assert torch.allclose(
        weights.sum(dim=-1),
        torch.ones(2, 4, 6),
        atol=1e-6,
    )

    # 固定 query，改变 context，输出必须变化
    assert not torch.allclose(output, changed_output)

    print("output shape:", tuple(output.shape))
    print("weights shape:", tuple(weights.shape))
    print("weights sum to 1: OK")
    print("output depends on context: OK")


def test_backward():
    torch.manual_seed(42)
    model = CrossAttention(embed_dim=32, num_heads=4)

    query = torch.randn(2, 6, 32, requires_grad=True)
    context = torch.randn(2, 6, 32, requires_grad=True)

    output, _ = model(query, context)
    target = torch.randn_like(output)
    loss = (output - target).square().mean()
    loss.backward()

    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name

    for x in (query, context):
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()
        assert x.grad.abs().sum() > 0

    print("loss:", loss.item())
    print("all parameters have finite grads: OK")
    print("both inputs have finite non-zero grads: OK")


if __name__ == "__main__":
    test_forward_and_context_dependence()
    test_backward()
    print("All cross-attention tests passed.")

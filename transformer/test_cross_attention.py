import torch

from cross_attention import CrossAttention

import math


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

def test_unequal_lengths_and_padding_invariance():
    torch.manual_seed(42)
    model = CrossAttention(embed_dim=32, num_heads=4)
    model.eval()

    query = torch.randn(2, 4, 32)
    context = torch.randn(2, 7, 32)

    context_mask = torch.tensor([
        [True, True, True, False, False, False, False],
        [True, True, True, True, True, False, False],
    ])

    # 只改 padding 位置的值
    changed_context = context.clone()
    changed_context[~context_mask] += 10.0

    with torch.no_grad():
        output, weights = model(query, context, context_mask)
        changed_output, _ = model(query, changed_context, context_mask)

    assert output.shape == (2, 4, 32), output.shape
    assert weights.shape == (2, 4, 4, 7), weights.shape

    # padding 位置被屏蔽，改它们不影响输出
    torch.testing.assert_close(output, changed_output)

    print("output shape:", tuple(output.shape))
    print("weights shape:", tuple(weights.shape))
    print("padding invariance: OK")


def test_reject_all_invalid_sample():
    torch.manual_seed(42)
    model = CrossAttention(embed_dim=32, num_heads=4)
    model.eval()

    query = torch.randn(2, 4, 32)
    context = torch.randn(2, 7, 32)

    # 样本 0 全 False，样本 1 有有效位置
    bad_mask = torch.tensor([
        [False, False, False, False, False, False, False],
        [True, True, True, True, True, False, False],
    ])

    with torch.no_grad():
        try:
            model(query, context, bad_mask)
        except ValueError:
            pass
        else:
            raise AssertionError("all-False sample should raise ValueError")

    print("all-invalid sample rejected: OK")

def test_masked_cross_attention_hand_calculation():
    model = CrossAttention(embed_dim=2, num_heads=1)

    query = torch.tensor([[[0.0, 0.0], [math.sqrt(2) * math.log(2), 0.0]]])
    context = torch.tensor([[[0.0, 2.0], [8.0, 100.0], [1.0, 5.0]]])
    context_mask = torch.tensor([[True, False, True]])

    with torch.no_grad():
        for layer in (model.q_proj, model.k_proj, model.v_proj, model.out_proj):
            layer.weight.copy_(torch.eye(2))
            layer.bias.zero_()

        output, weights = model(query, context, context_mask)

    assert output.shape == (1, 2, 2), output.shape
    assert weights.shape == (1, 1, 2, 3), weights.shape

    # 1. padding 位置的权重全为 0
    assert torch.all(weights[..., 1] == 0), weights[..., 1]

    # 2. 完整权重对照手算
    expected_weights = torch.tensor([[
        [
            [0.5, 0.0, 0.5],
            [1.0 / 3.0, 0.0, 2.0 / 3.0],
        ]
    ]])
    torch.testing.assert_close(weights, expected_weights)

    # 3. 输出对照手算
    expected_output = torch.tensor([[
        [0.5, 3.5],
        [2.0 / 3.0, 4.0],
    ]])
    torch.testing.assert_close(output, expected_output)

    print("weights[..., 1] all zero: OK")
    print("weights match hand calculation: OK")
    print("output matches hand calculation: OK")


if __name__ == "__main__":
    test_forward_and_context_dependence()
    test_backward()
    test_unequal_lengths_and_padding_invariance()
    test_reject_all_invalid_sample()
    test_masked_cross_attention_hand_calculation()
    print("All cross-attention tests passed.")

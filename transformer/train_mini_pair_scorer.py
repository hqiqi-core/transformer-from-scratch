import torch
import torch.nn.functional as F

from mini_pair_scorer import MiniPairScorer

EMBED_DIM = 16
NUM_HEADS = 2
QUERY_MASK = torch.tensor([[True, False]])
CONTEXT_MASK = torch.tensor([[True, True, False, False]])

"""
样本	query	正例	负例
样本 0	[a, 0]	A [a, d, 0, 0]	B [b, d, 0, 0]
样本 1	[b, 0]	B [b, d, 0, 0]	A [a, d, 0, 0]

"""
def make_data(num_groups: int, seed: int):
    g = torch.Generator().manual_seed(seed)
    queries, pos_contexts, neg_contexts = [], [], []

    for _ in range(num_groups):
        a = torch.randn(EMBED_DIM, generator=g)
        a = a / a.norm()
        b = torch.randn(EMBED_DIM, generator=g)
        b = b / b.norm()
        d = torch.randn(EMBED_DIM, generator=g)
        d = d / d.norm()

        zero = torch.zeros(EMBED_DIM)

        ctx_A = torch.stack([a, d, zero, zero])   # [4, D]
        ctx_B = torch.stack([b, d, zero, zero])   # [4, D]

        q_a = torch.stack([a, zero])               # [2, D]
        q_b = torch.stack([b, zero])               # [2, D]

        # 样本 1：query=[a,0]，正例 A，负例 B
        queries.append(q_a)
        pos_contexts.append(ctx_A)
        neg_contexts.append(ctx_B)

        # 样本 2：query=[b,0]，正例 B，负例 A
        queries.append(q_b)
        pos_contexts.append(ctx_B)
        neg_contexts.append(ctx_A)

    return (
        torch.stack(queries),
        torch.stack(pos_contexts),
        torch.stack(neg_contexts),
    )


def pair_accuracy(model, query, pos_ctx, neg_ctx):
    model.eval()
    b = query.shape[0]
    q_mask = QUERY_MASK.expand(b, -1)
    c_mask = CONTEXT_MASK.expand(b, -1)
    with torch.no_grad():
        s_pos = model(query, pos_ctx, q_mask, c_mask)
        s_neg = model(query, neg_ctx, q_mask, c_mask)
    return (s_pos > s_neg).float().mean().item()


def main():
    torch.manual_seed(0)

    train_q, train_pos, train_neg = make_data(512, seed=42)
    val_q, val_pos, val_neg = make_data(128, seed=43)

    print("train samples:", len(train_q))
    print("val samples:", len(val_q))

    model = MiniPairScorer(EMBED_DIM, NUM_HEADS)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)

    before_acc = pair_accuracy(model, val_q, val_pos, val_neg)
    print(f"\nbefore training: val pair accuracy = {before_acc:.4f}")

    batch_size = 64
    num_train = len(train_q)

    for epoch in range(10):
        model.train()
        perm = torch.randperm(num_train)
        total_loss = 0.0

        for start in range(0, num_train, batch_size):
            idx = perm[start:start + batch_size]
            q = train_q[idx]
            pos = train_pos[idx]
            neg = train_neg[idx]
            b = q.shape[0]
            q_mask = QUERY_MASK.expand(b, -1)
            c_mask = CONTEXT_MASK.expand(b, -1)

            s_pos = model(q, pos, q_mask, c_mask)# 对正例和负例各跑一次
            s_neg = model(q, neg, q_mask, c_mask)
            loss = F.softplus(-(s_pos - s_neg)).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(idx)

        avg_loss = total_loss / num_train
        acc = pair_accuracy(model, val_q, val_pos, val_neg)
        print(f"epoch {epoch+1:02d}: train_loss={avg_loss:.4f}  val_acc={acc:.4f}")

    after_acc = pair_accuracy(model, val_q, val_pos, val_neg)
    print(f"\nafter training: val pair accuracy = {after_acc:.4f}")


if __name__ == "__main__":
    main()
import argparse
import random

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from tiny_transformer import TinyTransformerEncoder

def make_train_sequences(num_samples, seed):
    """固定 seed 生成一次。label 各半（i%2），label0 标记在头，label1 标记在尾。 要自己生成"标记在头/尾"的序列"""
    rng = random.Random(seed)  # random.Random(seed) 创建一个独立的随机数发生器，固定 seed 后每次调用生成完全相同的序列，可复现。而且不污染全局 random 状态（别的代码用全局 random 不受影响）
    sequences, labels = [], []
    for i in range(num_samples):
        length = rng.randint(4, 12)  # 长度 4–12 随机
        label = i % 2   # make_train_sequences 里 label = i % 2：i 是循环变量，每条样本都递增，严格交替 0/1，各半
        # 为什么 length-1：整条序列长度是 length，其中 1 个位置留给标记 token 1，剩下 length-1 个是普通 token。为什么 2–31：0 是 PAD、1 是标记，普通 token 避开这两个，用 2–31 共 30 个值。
        # 普通 token 从 [2,31] 随机，长度 length-1
        ordinary = [rng.randint(2, 31) for _ in range(length - 1)]  # 列表 
        # label 0 → 标记在开头；label 1 → 标记在结尾。
        # 注意运算符优先级：[1] + ordinary if label == 0 else ordinary + [1] 等价于 ([1] + ordinary) if (label == 0) else (ordinary + [1])，条件表达式优先级低于 +，所以两种写法都对，但加括号更清楚。
        tokens = [1] + ordinary if label == 0 else ordinary + [1]
        sequences.append(tokens)
        labels.append(label)
    return sequences, labels  # 元组


def make_validation_sequences(num_samples, seed, train_set):  # validation 和 train 不能有完全重复的样本
    """独立 seed；与 train 完全重复的样本跳过重新生成；label 用 len%2 保证各半。"""
    rng = random.Random(seed)
    sequences, labels = [], []
    while len(sequences) < num_samples:  # validation 遇到重复要 continue 丢掉，不能保证循环 256 次就得到 256 条有效的（因为有些被丢了）。所以要用 while len(sequences) < 256：一直生成，直到凑够 256 条不重复的
        length = rng.randint(4, 12)
        label = len(sequences) % 2
        ordinary = [rng.randint(2, 31) for _ in range(length - 1)]
        tokens = [1] + ordinary if label == 0 else ordinary + [1]
        # tokens 是新生成的 list，比如 [1,7,9,4]。train_set 里存的是 tuple (1,7,9,4)。要判断"这条新序列在不在 train 里"，必须用同一种类型比较——把 tokens 也转成 tuple，才能和 set 里的 tuple 对上
        if tuple(tokens) in train_set:  # 重复，丢掉，继续 while  查重用 tuple 不是 list
            continue   
        sequences.append(tokens)
        labels.append(label)
    return sequences, labels

"""
DataLoader 一次给一个 batch，但每条序列长度不同——补 PAD
batch = [
    ([1, 7, 9, 4], 0),                      # 长度4
    ([8, 6, 2, 5, 9, 1], 1),                # 长度6
    ([1, 11, 6, 26, 5, 21], 0),             # 长度6
    ...
]
collate 要产出三样东西:
input_ids:      [B, Lmax]  long   —— 补好PAD的id
attention_mask: [B, Lmax]  bool   —— ne(0)，True=有效
labels:         [B]        long   —— 每条一个label


"""
def collate_sequences(batch):
    # pad_sequence 要求每条先转成 torch.long 张量，不能直接传 list。所以 collate 里先 torch.tensor(seq, dtype=torch.long)
    sequences = [torch.tensor(seq, dtype=torch.long) for seq, _ in batch] 
    labels = torch.tensor([label for _, label in batch], dtype=torch.long)
    """
    sequences：一个 list，每个元素是 1D torch.long 张量（一条序列）
    batch_first=True：输出形状 [B, Lmax]；不加是 [Lmax, B]
    padding_value=0：补 0，和 ne(0) 的约定一致 默认右侧补（短的在右边补到最长）
    """
    input_ids = pad_sequence(sequences, batch_first=True, padding_value=0)  # [B, Lmax]
    attention_mask = input_ids.ne(0)                                        # [B, Lmax] bool
    return input_ids, attention_mask, labels


class SequenceDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


class SequenceClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = TinyTransformerEncoder(
            vocab_size=32,
            max_length=12,
            d_model=32,
            num_heads=4,
            ffn_dim=64,
            num_layers=2,
        )
        self.classifier = nn.Linear(32, 2)

    def forward(self, input_ids, attention_mask):
        hidden = self.encoder(input_ids, attention_mask=attention_mask)  # [B,L,32]  每个 token 一个 32 维向量 

        # 分类要 [B,2]，一条序列一个结果——masked mean pooling  只对有效 token 求平均，PAD 位置被排除。既用全部真实位置的信息，又不受 PAD 污染
        """
        attention_mask 是 [B,L] bool。hidden 是 [B,L,32] float
        unsqueeze(-1)：在最后插一维 → [B,L,1]。为什么是 -1（最后）：要和 hidden 的最后一维（32）广播，mask 的 [B,L,1] 广播到 [B,L,32]，每个 token 的 32 维特征一起乘同一个 0/1。
        .to(hidden.dtype)：bool 不能和 float 直接乘（PyTorch 会报错或隐式转换有坑），显式转 float，True→1.0，False→0.0
        """
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)             # [B,L,1]  
        """
        hidden * mask：[B,L,32] * [B,L,1] 广播 → [B,L,32]。PAD 位置的 32 维全被乘 0，有效位置不变。
        .sum(dim=1)：沿 L 维求和 → [B,32] 得到"有效位置向量之和"    mask.sum(dim=1)：[B,L,1] 沿 L 求和 → [B,1]。数的是有效 token 数
        相除：[B,32] / [B,1] 广播 → [B,32]。每个样本除以自己的有效 token 数
        """
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1)            # [B,32]

        logits = self.classifier(pooled)                                 # [B,2]  
        return logits


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    total = 0
    for input_ids, attention_mask, labels in loader:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)  # 最后一个 batch 可能不满  平均loss × 本批样本数 = 本批总loss
        total += labels.size(0)  # 累加样本数
    return total_loss / total  # 总loss / 总样本数 = 真正的平均loss


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    for input_ids, attention_mask, labels in loader:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        logits = model(input_ids, attention_mask)
        preds = logits.argmax(dim=1)
        # 如果 batch 大小不同，小 batch 的 accuracy 被过度加权。而且平均 accuracy 和真实准确率不是一回事 所以不能逐batch计算acc再取平均
        correct += (preds == labels).sum().item()  # 累加正确个数
        total += labels.size(0)
    return correct / total   # 总正确数 / 总样本数



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    train_seqs, train_labels = make_train_sequences(1024, seed=args.seed)
    train_set = set(tuple(s) for s in train_seqs)  # 把 train 序列转成 set，供 validation 查重
    val_seqs, val_labels = make_validation_sequences(256, seed=args.seed + 1000, train_set=train_set)

    train_ds = SequenceDataset(train_seqs, train_labels)
    val_ds = SequenceDataset(val_seqs, val_labels)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_sequences)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate_sequences)

    model = SequenceClassifier().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_acc = evaluate(model, val_loader, device)
        print(f"Epoch {epoch:02d} | train_loss={train_loss:.4f} | val_accuracy={val_acc:.4f}")


if __name__ == "__main__":
    main()
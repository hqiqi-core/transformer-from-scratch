import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from mini_vit import MiniViT
from patchify import patchify

IMAGE_SIZE = (8, 12)
IN_CHANNELS = 3
PATCH_SIZE = 4
EMBED_DIM = 32
NUM_HEADS = 4
FFN_DIM = 64
NUM_LAYERS = 2
NUM_CLASSES = 4
GW = IMAGE_SIZE[1] // PATCH_SIZE   # 3

CLASS_TO_PATCH = [0, 2, 3, 5]


def make_split(num_samples: int, seed: int):
    g = torch.Generator().manual_seed(seed)
    samples_per_class = num_samples // NUM_CLASSES
    labels = torch.arange(NUM_CLASSES).repeat_interleave(samples_per_class)

    images = 0.1 * torch.rand(    # *IMAGE_SIZE 解包成 8, 12。所以形状是 [N,3,8,12]。0.1 * rand：背景值在 [0, 0.1)，暗但不全黑，有噪声
        num_samples, IN_CHANNELS, *IMAGE_SIZE, generator=g
    )

    for i, label in enumerate(labels.tolist()):
        patch_idx = CLASS_TO_PATCH[label]
        row, col = divmod(patch_idx, GW)
        r0, c0 = row * PATCH_SIZE, col * PATCH_SIZE
        images[i, :, r0:r0 + PATCH_SIZE, c0:c0 + PATCH_SIZE] = 1.0

    perm = torch.randperm(num_samples, generator=g)
    return images[perm], labels[perm]


def check_labels(images, labels):
    brightest = patchify(images, PATCH_SIZE).mean(dim=-1).argmax(dim=1)
    expected = torch.tensor(CLASS_TO_PATCH)[labels]
    assert torch.equal(brightest, expected), (
        f"label-pixel mismatch: got {brightest[:5]}, expected {expected[:5]}"
    )


def train_one_epoch(model, loader, optimizer):
    model.train()
    total_loss, total = 0.0, 0
    for images, labels in loader:
        optimizer.zero_grad()
        logits = model(images)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(labels)
        total += len(labels)
    return total_loss / total


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0
    for images, labels in loader:
        logits = model(images)
        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += len(labels)
    return correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1)
    args = parser.parse_args()

    torch.manual_seed(0)

    train_images, train_labels = make_split(512, seed=42)
    val_images, val_labels = make_split(128, seed=43)

    check_labels(train_images, train_labels)
    check_labels(val_images, val_labels)
    print("label-pixel check: OK")

    print("train samples:", len(train_images))
    print("val samples:", len(val_images))
    for c in range(NUM_CLASSES):
        print(f"  class {c}: train={(train_labels == c).sum().item()}  val={(val_labels == c).sum().item()}")

    train_ds = TensorDataset(train_images, train_labels)
    val_ds = TensorDataset(val_images, val_labels)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

    model = MiniViT(
        IMAGE_SIZE, IN_CHANNELS, PATCH_SIZE,
        EMBED_DIM, NUM_HEADS, FFN_DIM, NUM_LAYERS, NUM_CLASSES,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)

    initial_val_acc = evaluate(model, val_loader)
    print(f"epoch 00: val_acc={initial_val_acc:.4f}")

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer)
        val_acc = evaluate(model, val_loader)
        print(f"epoch {epoch:02d}: train_loss={train_loss:.4f}  val_acc={val_acc:.4f}")


if __name__ == "__main__":
    main()

# Transformer from Scratch

A from-scratch implementation of the core components of the Transformer architecture.

The goal of this repository is to understand Transformer models from mathematical formulation to actual PyTorch implementation.

## 🎯 Goals

This repository focuses on implementing and understanding the fundamental components behind modern Transformer-based models.

The main topics include:

- Self-Attention
- Multi-Head Attention
- Positional Encoding
- Feed-Forward Networks
- Residual Connections
- Layer Normalization
- Transformer Encoder
- Transformer Decoder
- Computational complexity

## 🧠 Core Architecture

The implementation follows the basic Transformer structure:

```text
Input Tokens
     │
     ▼
Embedding
     │
     ▼
Positional Encoding
     │
     ▼
Multi-Head Self-Attention
     │
     ▼
Residual + LayerNorm
     │
     ▼
Feed-Forward Network
     │
     ▼
Residual + LayerNorm
     │
     ▼
Transformer Block
```

## 📚 Contents

```text
transformer-from-scratch/
│
├── attention/
│   ├── scaled_dot_product.py
│   └── multi_head_attention.py
│
├── transformer/
│   ├── positional_encoding.py
│   ├── feed_forward.py
│   ├── transformer_block.py
│   └── transformer.py
│
├── experiments/
│   └── ...
│
├── README.md
└── requirements.txt
```

The repository will evolve as more Transformer components are implemented.

## 🔬 Implementation Roadmap

### Phase 1 — Scaled Dot-Product Attention

-  Query / Key / Value
-  Attention score
-  Scaling by √d
-  Softmax
-  Attention output
-  Attention mask

### Phase 2 — Multi-Head Attention

-  Head splitting
-  Parallel attention heads
-  Concatenation
-  Output projection

### Phase 3 — Transformer Block

-  Feed-forward network
-  Residual connection
-  Layer normalization
-  Dropout

### Phase 4 — Complete Transformer

-  Encoder
-  Decoder
-  Positional encoding
-  End-to-end training
-  Small-scale experiments

## 📐 Mathematical Reference

The core attention operation is:

```text
Attention(Q, K, V)
    = softmax(QKᵀ / √dₖ)V
```

Multi-head attention applies this operation across multiple representation subspaces.

The implementation will connect each mathematical operation to its corresponding PyTorch code.

## 🔬 Experiments

Experiments will investigate:

- Attention behavior
- Sequence length
- Number of attention heads
- Hidden dimension
- Training stability
- Computational complexity

## 🚀 Philosophy

The goal is not simply to reproduce an existing Transformer implementation.

Each core component is implemented and tested independently to build an understanding of how Transformer architectures work internally.

## 📌 Status

🚧 Under active development

This repository serves as the theoretical and implementation foundation for later multimodal and VLM projects.

## Implementation provenance

The initial Attention and Transformer Block implementations were
migrated from `hqiqi-core/pytorch-from-scratch` at commit `25bcedb`.

Their maintained versions now live in this repository.
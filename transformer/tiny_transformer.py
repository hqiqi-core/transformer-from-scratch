import torch
import torch.nn as nn

from transformer_block import TransformerBlock


class TinyTransformerEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,   # 词表容量 也就是数据库的caption进行处理以后得到的 每个token有自己的编号 还有占位0和位置1
        max_length: int = 16,
        d_model: int = 32,  # dimension of the model
        num_heads: int = 4, 
        ffn_dim: int = 64, 
        num_layers: int = 2, 
    ):
        super().__init__()
        assert num_layers >= 1 

        self.max_length = max_length 
        """
        模型内部只认 ID，不认原始字符串
        Token Embedding —— 把编号变成向量 input_ids 是整数，比如 [[3,7,2,9],[5,1,8,4]]。这些整数是词表编号，不是数值特征
        input_ids [B,L] int64  →  token_vectors [B,L,D] float32 同一 token ID 在不同位置先查出相同向量——位置信息还没进来，下一步补
        每个 ID 变成一行 32 维向量
        序列 A: [3, 7, 2, 9]  → token 7 的向量是 W[7]
        序列 B: [9, 2, 7, 3]  → token 7 的向量还是 W[7]
        经过学习，W 这张 [vocab_size, 32] 的矩阵被更新；每个 ID 对应的 32 维向量就是 W 的第 id 行
        """
        self.token_embedding = nn.Embedding(vocab_size, d_model)  # 内部只有一个权重矩阵 W: [vocab_size, d_model] = [32, 32] 
        """
        Position Embedding —— 告诉模型顺序   Self-Attention 做的是加权求和，没有天然的"第几个位置"概念
        只做 token 查表时，[3,7,2,9] 和 [9,2,7,3] 里相同 token 拿到相同向量，Attention 看到的是同一个多重集，无法区分顺序。所以必须显式注入位置。
        可学习位置 Embedding  
        为什么 max_length 而不是 L：Embedding 表要覆盖未来可能的最长序列。max_length 是表容量上限，不是当前输入必须达到的长度。当前 L 可以是 1..max_length 任意值，arange(L) 只取前 L 行
        位置 Embedding 是训练参数，训练中会学到"每个位置的偏移量"，加到 token 向量上就完成了"给同一 token 在不同位置一个不同表示"。

        """
        self.position_embedding = nn.Embedding(max_length, d_model)   
        # nn.ModuleList 是 nn.Module 的子类，会注册每个子模块，参数正常进入 model.parameters() / state_dict()，.to(device) 也递归作用
        # 为什么两层不是"重复执行同一个变换"：每层是独立实例、独立参数，Block1 和 Block2 权重不同。
        # 输入 [B,L,D] 逐层经过，shape 不变，但内容逐层变化——每个位置的表示逐层聚合其他位置的信息。shape 相同不代表变换相同
        self.blocks = nn.ModuleList([   # 普通 Python list：list 不是 nn.Module，不会注册子模块。model.parameters()、state_dict()、.to(device) 都会漏掉这些 Block 的参数。模型看起来有层，实际优化器和设备迁移都管不到
            TransformerBlock(    #  每次重新创建  因为如果用[block] * num_layers：这是同一个对象引用 n 次。所有"层"共享一套参数，反向时梯度累加到同一组参数上，等价于只有一层在重复算
                embed_dim=d_model,
                num_heads=num_heads,
                hidden_dim=ffn_dim,
            )
            for _ in range(num_layers)  
        ])
        

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        assert input_ids.dim() == 2, "input_ids must be [B, L]"    
        assert input_ids.dtype == torch.long, "input_ids must be int64"   

        B, L = input_ids.shape  
        assert 1 <= L <= self.max_length, (
            f"sequence length {L} out of range [1, {self.max_length}]"
        )

        # 为什么用 input_ids.device 建 positions：模型在 cuda、输入在 cuda，如果 arange 默认在 cpu，position_embedding(positions) 会 device mismatch。位置索引必须和输入同 device
        positions = torch.arange(L, device=input_ids.device)  

        token_vectors = self.token_embedding(input_ids)          # [B,L,D]
        position_vectors = self.position_embedding(positions)    # [L,D]  
        # [B,L,D] + [1,L,D] → [B,L,D]。位置向量在 batch 内共享——同一位置对所有样本用同一个位置向量，这是合理的（位置语义与样本无关）
        x = token_vectors + position_vectors.unsqueeze(0)        # [B,L,D]   position_vectors.unsqueeze(0)：[L,D] → [1,L,D]，再与 [B,L,D] 广播相加

        for block in self.blocks:
            x = block(x)

        return x
import argparse

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from transformers import CLIPImageProcessor, CLIPModel

MODEL_NAME = "openai/clip-vit-base-patch32"

"""
所有图片 → 512 维向量 → 入 FAISS
query → 512 维向量 → FAISS 查 → Top-10 候选
    ↓
对 Top-10 每个候选，取它的 [49,768] patch tokens
    ↓
Cross-Attention(query, patch_tokens) → 重新打分 → 重排


张量	                shape	           存在哪	                    用途
retrieval_features	  [N, 512]	        FAISS 索引	              召回，余弦相似度
patch_tokens	    [N, 49, 768]	内存/磁盘，不进 FAISS	精排的 Cross-Attention context
cls_tokens	          [N, 768]	        一般不用	               末层 CLS，pooled 的输入
pooler_output	       [N, 768]	        一般不用	                投影的输入，中间量
为什么 patch_tokens 不进 FAISS：FAISS 索引是"一个向量 → 一个位置"，49 个 token 没法直接塞。而且召回只要全局相似度，49 个 token 是精排的事。FAISS 只存 512 维向量，patch_tokens 另存。
"""
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, nargs="+", required=True) # 可以传入多张图
    parser.add_argument("--patch-index", type=int, default=0)
    parser.add_argument("--output", type=str, default="/tmp/clip_patch_preview.png")
    args = parser.parse_args()

    images = [Image.open(p).convert("RGB") for p in args.image]
    for i, im in enumerate(images):
        print(f"image[{i}] original size:", im.size)

    processor = CLIPImageProcessor.from_pretrained(MODEL_NAME)
    model = CLIPModel.from_pretrained(MODEL_NAME).eval()  # CLIPVisionModel 没有 visual_projection，拿不到 512 维召回向量。CLIPModel 有视觉塔 + 投影

    pixel_values = processor(images=images, return_tensors="pt")["pixel_values"]
    print("pixel_values shape:", tuple(pixel_values.shape))

    with torch.inference_mode():
        vision_out = model.vision_model(pixel_values=pixel_values)  # CLIPModel.forward 会同时跑文本塔，浪费。只调 vision_model 取视觉输出
        tokens = vision_out.last_hidden_state
        cls_tokens = tokens[:, 0, :]
        patch_tokens = tokens[:, 1:, :]
        pooled = vision_out.pooler_output
        projected = model.visual_projection(pooled)
        retrieval_features = F.normalize(projected, dim=-1)  # 归一化

    print("last_hidden_state shape:", tuple(tokens.shape))
    print("cls_tokens shape:", tuple(cls_tokens.shape))
    print("patch_tokens shape:", tuple(patch_tokens.shape))
    print("pooler_output shape:", tuple(pooled.shape))
    print("retrieval_features shape:", tuple(retrieval_features.shape))
    print("retrieval_features norms:", retrieval_features.norm(dim=-1).tolist())

    H, W = pixel_values.shape[-2:]
    P = model.config.vision_config.patch_size  # CLIPModel 顶层 config 有视觉和文本两套，patch_size 在 vision_config 下
    grid_h = H // P
    grid_w = W // P
    num_patches = grid_h * grid_w

    print("patch size:", P)
    print("patch grid:", f"{grid_h} x {grid_w}")

    assert tokens.shape[1] == num_patches + 1, (
        f"token count {tokens.shape[1]} != num_patches+1 {num_patches + 1}"
    )

    if not (0 <= args.patch_index < num_patches):
        raise ValueError(
            f"patch_index {args.patch_index} out of range [0, {num_patches})"
        )

    row, col = divmod(args.patch_index, grid_w)
    seq_idx = args.patch_index + 1
    y0, y1 = row * P, (row + 1) * P
    x0, x1 = col * P, (col + 1) * P

    print("patch index:", args.patch_index)
    print("sequence index:", seq_idx)
    print("row, col:", (row, col))
    print("box xyxy:", (x0, y0, x1, y1))
    print("selected token shape:", tuple(tokens[0, seq_idx].shape))

    # 批次翻转检查
    with torch.inference_mode():
        flipped_out = model.vision_model(pixel_values=pixel_values.flip(0))

    torch.testing.assert_close(
        flipped_out.last_hidden_state[:, 1:, :],
        patch_tokens.flip(0),
    )
    print("batch flip check: OK")

    # 预览第 0 张
    mean = torch.tensor(processor.image_mean).view(1, 3, 1, 1)
    std = torch.tensor(processor.image_std).view(1, 3, 1, 1)
    pixels = pixel_values[0] * std[0] + mean[0]
    pixels = pixels.clamp(0, 1)
    pixels = (pixels.permute(1, 2, 0) * 255).byte()
    display = Image.fromarray(pixels.numpy())

    draw = ImageDraw.Draw(display)
    draw.rectangle([x0, y0, x1 - 1, y1 - 1], outline="red", width=3)

    display.save(args.output)
    print("saved preview to:", args.output)


if __name__ == "__main__":
    main()
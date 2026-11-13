import argparse

import torch
from PIL import Image, ImageDraw
from transformers import CLIPImageProcessor, CLIPVisionModel

MODEL_NAME = "openai/clip-vit-base-patch32"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--patch-index", type=int, default=0)
    parser.add_argument("--output", type=str, default="/tmp/clip_patch_preview.png")
    args = parser.parse_args()

    image = Image.open(args.image).convert("RGB")
    print("original image size:", image.size)   # (W, H)

    processor = CLIPImageProcessor.from_pretrained(MODEL_NAME)  # CLIPImageProcessor：只做图像预处理（resize、裁剪、归一化），不加载文本塔
    model = CLIPVisionModel.from_pretrained(MODEL_NAME).eval()  # 只加载视觉编码器，不要 CLIPModel（那是完整的图文模型，有文本塔）。今天只关心视觉 token 

    pixel_values = processor(images=image, return_tensors="pt")["pixel_values"]  
    print("pixel_values shape:", tuple(pixel_values.shape))

    with torch.inference_mode():
        tokens = model(pixel_values=pixel_values).last_hidden_state  # last_hidden_state：末层输出，[1,50,768]。1 CLS + 49 patch
    print("last_hidden_state shape:", tuple(tokens.shape))

    # 从配置读尺寸，不硬编码 
    H, W = pixel_values.shape[-2:]    # 224, 224
    P = model.config.patch_size       # 32
    grid_h = H // P         # 7 
    grid_w = W // P        # 7
    num_patches = grid_h * grid_w   # 49

    print("patch size:", P)
    print("patch grid:", f"{grid_h} x {grid_w}")

    assert tokens.shape[1] == num_patches + 1, (
        f"token count {tokens.shape[1]} != num_patches+1 {num_patches + 1}"
    )

    if not (0 <= args.patch_index < num_patches):
        raise ValueError(
            f"patch_index {args.patch_index} out of range [0, {num_patches})"
        )

    # 索引转换
    row, col = divmod(args.patch_index, grid_w)
    seq_idx = args.patch_index + 1   # CLS 占 0
    y0, y1 = row * P, (row + 1) * P
    x0, x1 = col * P, (col + 1) * P

    print("patch index:", args.patch_index)
    print("sequence index:", seq_idx)
    print("row, col:", (row, col))
    print("box xyxy:", (x0, y0, x1, y1))
    print("selected token shape:", tuple(tokens[0, seq_idx].shape))

    # 反归一化，恢复可显示像素
    mean = torch.tensor(processor.image_mean).view(1, 3, 1, 1)
    std = torch.tensor(processor.image_std).view(1, 3, 1, 1)
    pixels = pixel_values * std + mean
    pixels = pixels.clamp(0, 1)[0]                    # [3,H,W]
    pixels = (pixels.permute(1, 2, 0) * 255).byte()   # [H,W,3] uint8
    display = Image.fromarray(pixels.numpy())

    draw = ImageDraw.Draw(display)
    draw.rectangle([x0, y0, x1 - 1, y1 - 1], outline="red", width=3)

    display.save(args.output)
    print("saved preview to:", args.output)


if __name__ == "__main__":
    main()
#!/usr/bin/env python3

import torch
from diffusers import BitsAndBytesConfig, SD3Transformer2DModel, StableDiffusion3Pipeline
from PIL import Image
import os
import glob

# 1. 讀取 prompt
prompt_path = "/home/pmd/Desktop/Alex/prompts/SD3_prompt.txt"
if not os.path.exists(prompt_path):
    raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
with open(prompt_path, "r", encoding="utf-8") as f:
    user_prompt = f.read().strip()

# 在用戶原始 prompt 前加上指定前綴
prefix = ""
prompt = f"{prefix} {user_prompt} " 
print(f"Using prompt: {prompt}")

model_id = "stabilityai/stable-diffusion-3.5-large"
# # 2. 設定 4-bit NF4 量化
# nf4_config = BitsAndBytesConfig(
#     load_in_8bit=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype=torch.float16
# )
# # 載入 transformer model
# model_nf4 = SD3Transformer2DModel.from_pretrained(
#     model_id,
#     subfolder="transformer",
#     quantization_config=nf4_config,
#     torch_dtype=torch.float16,
#     cache_dir="/home/pmd/Desktop/Alex/Tools/SD3"
# )

# 3. 建立並載入 Pipeline，並啟用 CPU Offload
pipeline = StableDiffusion3Pipeline.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    cache_dir="/home/pmd/Desktop/Alex/.cache/sd3"
)
pipeline.enable_model_cpu_offload()

# 4. 推論圖片
result = pipeline(
    prompt=prompt,
    num_inference_steps=50,
    guidance_scale=7.5,
)
orig_image = result.images[0]

# 5. 建立並檢查輸出目錄
output_dir = "/home/pmd/Desktop/Alex/images/SD3"
os.makedirs(output_dir, exist_ok=True)
# 找到現有檔案 SD3_*.png 和 SD3_*_*.png
pattern = os.path.join(output_dir, "SD3_*.png")
existing_files = glob.glob(pattern)
# 提取索引
indices = []
for path in existing_files:
    name = os.path.splitext(os.path.basename(path))[0]  # e.g. "SD3_1_1"
    parts = name.split("_")
    if len(parts) >= 2 and parts[1].isdigit():
        indices.append(int(parts[1]))
next_n = max(indices) + 1 if indices else 1

# 6. 儲存圖片至 SD3 目錄
final_basename = f"SD3_{next_n}_1"
final_filename = final_basename + ".png"
output_path = os.path.join(output_dir, final_filename)
orig_image.save(output_path)
print(f"Image saved to {output_path}")

# 7. 同步存到 Final_Output
final_base = "/home/pmd/Desktop/Alex/images/Final_Output"
final_dir = os.path.join(final_base, final_basename, "SD")
os.makedirs(final_dir, exist_ok=True)
final_output_path = os.path.join(final_dir, final_filename)
orig_image.save(final_output_path)
print(f"Also saved to {final_output_path}")

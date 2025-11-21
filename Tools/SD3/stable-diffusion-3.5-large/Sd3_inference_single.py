#!/home/pmd/anaconda3/envs/sd3/bin/python

import os
import torch
from diffusers import StableDiffusion3Pipeline

def main():
    # 1. 載入模型
    pipe = StableDiffusion3Pipeline.from_pretrained(
        "stabilityai/stable-diffusion-3.5-large",
        torch_dtype=torch.bfloat16,
        cache_dir="/home/pmd/Desktop/Alex/.cache/sd3"
    )
    pipe = pipe.to("cuda")
    
    # 2. 先找出已存在的最大資料夾編號
    base_out = "/home/pmd/Desktop/Alex/mywork_sd3"
    os.makedirs(base_out, exist_ok=True)
    existing = [
        d for d in os.listdir(base_out)
        if os.path.isdir(os.path.join(base_out, d)) and d.isdigit()
    ]
    max_idx = max(map(int, existing)) if existing else 0
    print(f"[INFO] 已有資料夾最大編號：{max_idx}")

    # 3. 打開 prompt 檔案，並從第一行開始
    prompts_path = "/home/pmd/Desktop/Alex/Tools/SD3/stable-diffusion-3.5-large/prompt.txt"
    with open(prompts_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            prompt = line.strip()
            if not prompt:
                continue  # 空行就跳過
            
            # 4. 計算新資料夾編號，從 max_idx+1 開始
            folder_idx = max_idx + idx
            img_folder = str(folder_idx)
            output_dir = os.path.join(base_out, img_folder)
            os.makedirs(output_dir, exist_ok=True)
            
            # 5. 生成影像
            image = pipe(
                prompt,
                num_inference_steps=50,
                guidance_scale=6.5,
            ).images[0]
            
            # 6. 存檔
            output_path = os.path.join(output_dir, "SD3_1_1.png")
            image.save(output_path)
            print(f"[{folder_idx}] prompt: {prompt!r} → saved {output_path}")

if __name__ == "__main__":
    main()


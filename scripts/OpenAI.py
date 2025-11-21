#!/usr/bin/env python3

import os
import base64
from mimetypes import guess_type
import argparse
from openai import AzureOpenAI
from azure.core.credentials import AzureKeyCredential

# 模型配置
model_name = "gpt-4o"
deployment = "gpt-4o"

def local_image_to_data_url(path):
    mime, _ = guess_type(path)
    mime = mime or "application/octet-stream"
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode()
    return f"data:{mime};base64,{data}"


def main():
    parser = argparse.ArgumentParser(
        description="将本地 txt 与本地图片一起发给 GPT-4o Vision，并把结果存成文件。"
    )
    parser.add_argument(
        "--input_txt_path", "-i",
        required=True,
        help="要发送给模型的 txt 文件路径，比如 /path/to/MLLM_input_prompt.txt"
    )
    parser.add_argument(
        "--img_path", "-m",
        # required=True,
        help="要发送给模型的图片文件路径，比如 /path/to/SD3_1.png"
    )
    parser.add_argument(
        "--output_txt_path", "-o",
        required=True,
        help="模型输出保存的文件路径，比如 /path/to/MLLM_output_instruction.txt"
    )
    args = parser.parse_args()

    # 1. 读取 txt 文件
    with open(args.input_txt_path, "r", encoding="utf-8") as f:
        user_text = f.read()

    # 2. 本地图片转 data-URL
    # img_data_url = local_image_to_data_url(args.img_path)

    # 3. 调用模型
    client = AzureOpenAI(
        api_version="2024-12-01-preview",
        azure_endpoint="Your API endpoint",
        api_key="Your API KEY"
    )

    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": 
             "You are an AI assistant specialized in verifying and correcting the text generated in Stable Diffusion-generated images."},
            {"role": "user", "content": [
                {"type": "text",      "text": user_text},
                # {"type": "image_url", "image_url": {"url": img_data_url}}
            ]}
        ],
        max_tokens=4096,
        temperature=1.0,
        top_p=1.0,
        model=deployment,
    )

    # 4. 将结果写到用户指定的输出文件
    with open(args.output_txt_path, "w", encoding="utf-8") as out_f:
        out_f.write(response.choices[0].message.content)

    print(f"已将模型输出保存到：{args.output_txt_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import sys
import json
from pathlib import Path
import argparse

# -------------------------------------------------------
# 接收參數
# -------------------------------------------------------
parser = argparse.ArgumentParser(description="Generate MLLM input prompt for a given image folder")
parser.add_argument(
    "--image_name", "-n",
    required=True,
    help="Final_Output 下的子資料夾名稱，例如 SD3_1_1"
)
args = parser.parse_args()
image_base_dir = Path("/home/pmd/Desktop/Alex/images/Final_Output")
image_name = args.image_name

# 定義路徑
sd_dir = image_base_dir / image_name / "SD"
prompt_dir = Path(f"/home/pmd/Desktop/Alex/prompts/{image_name}")
prompt_dir.mkdir(parents=True, exist_ok=True)
output_system = prompt_dir / "MLLM_input_prompt.txt"

# 全域設定
prompt_path = Path("/home/pmd/Desktop/Alex/prompts/SD3_prompt.txt")
system_prompt_path = Path("/home/pmd/Desktop/Alex/prompts/MLLM_system_prompt.txt")

# 構建 orig_prompt
sd3_text = prompt_path.read_text(encoding='utf-8')
segments = re.findall(r"[‘'“\"”](.+?)[’'”\"\"]", sd3_text)
if segments:
    orig_prompt = " ".join(seg.strip() for seg in segments)
else:
    # ⚠️ 沒有偵測到引號包裹的文字，就填空
    print("⚠️ 在 SD3_prompt.txt 找不到引號包裹文字，orig_prompt 將為空", file=sys.stderr)
    orig_prompt = ""


# 檢查 SD 資料夾
if not sd_dir.exists():
    print(f"Error: 找不到 SD 資料夾: {sd_dir}", file=sys.stderr)
    sys.exit(1)

# 收集檢測文字
from pathlib import Path
ocr_base = Path("/home/pmd/Desktop/Alex/images/OCR")

detected_texts = []
for img_path in sd_dir.glob("*.png"):
    stem = img_path.stem
    ann_file = ocr_base / stem / "Units" / "box" / "annotations.json"
    if not ann_file.exists():
        print(f"Warning: 找不到 Units/annotations.json: {ann_file}", file=sys.stderr)
        continue
    raw_units = json.loads(ann_file.read_text(encoding="utf-8"))
    for filename, info in raw_units.items():
        base_name = Path(filename).stem
        text = info.get("text", "").replace("[unk]", "").strip()
        if text:
            detected_texts.append(f"{base_name}: {text}")

# 提取純文字，轉成 quoted phrases
phrases = []
for line in detected_texts:
    txt = line.split(":", 1)[1].strip()
    txt = txt.strip("'\"")
    phrases.append(txt)
# 組成: "shared", "A", "grief's"
detected_texts_str = ", ".join(f'"{p}"' for p in phrases)

# 填充 system prompt 模板
sys_tpl = system_prompt_path.read_text(encoding='utf-8')

# 更新 Ground truth
filled_sys = re.sub(
    r'^(Ground truth phrases:).*$',
    lambda m: f"{m.group(1)} \"{orig_prompt}\"",
    sys_tpl,
    flags=re.MULTILINE
)
# 更新 OCR detected phrases
filled_sys = re.sub(
    r'^(OCR detected phrases:).*$',
    lambda m: f"OCR detected phrases: {detected_texts_str}",
    filled_sys,
    flags=re.MULTILINE
)

# 輸出
output_system.write_text(filled_sys, encoding='utf-8')
print(f"✅ 已寫入填充後的 system prompt：{output_system} (for {image_name})")

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import json
import re
import argparse
from pathlib import Path

# -----------------------------
# 路徑設定
# -----------------------------
RAW_IMG_DIR   = "/home/pmd/Desktop/Alex/images/SD3"  # 原圖所在
OCR_BASE_DIR  = "/home/pmd/Desktop/Alex/images/OCR"  # OCR 資料根目錄
PADDLE_SUBDIR = "Paddle"
BOX_SUB       = os.path.join("Units", "box")

OUTPUT_BASE    = "/home/pmd/Desktop/Alex/prompts"
# JSON 指令檔路徑根目錄
INSTR_ROOT     = OUTPUT_BASE
INSTR_FILENAME = "MLLM_output_instruction.txt"

# Layout 提示模板路徑
LAYOUT_TEMPLATE = os.path.join(OUTPUT_BASE, "Layout_system_prompt.txt")

# -----------------------------
# JSON 抽取輔助函式
# -----------------------------
def extract_json_text(raw: str) -> str:
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, flags=re.DOTALL)
    if m:
        return m.group(1)
    start = raw.find('{')
    end   = raw.rfind('}')
    if start != -1 and end != -1 and end > start:
        return raw[start:end+1]
    return raw.strip()

# -----------------------------
# Main
# -----------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='處理指定 image_name 的 OCR 與 MLLM 指令生成流程'
    )
    parser.add_argument(
        '--image_name', '-n',
        required=True,
        help='要處理的影像名稱（對應於 SD3_x 或 SD3_x_x）'
    )
    args = parser.parse_args()
    image_name = args.image_name

    # 建立各資料夾路徑
    ocr_dir    = os.path.join(OCR_BASE_DIR, image_name)
    paddle_dir = os.path.join(ocr_dir, PADDLE_SUBDIR)
    box_dir    = os.path.join(ocr_dir, BOX_SUB)

    # 讀原始 annotations
    box_json = os.path.join(box_dir, 'annotations.json')
    if not os.path.isfile(box_json):
        print(f"[SKIP] {image_name} 缺少 annotations.json，跳過")
        exit(1)
    with open(box_json, 'r', encoding='utf-8') as f:
        annotations_box = json.load(f)

    # 計算 box 大小
    for info in annotations_box.values():
        (x1, y1), (x2, y2) = info.get('coord', [(0,0),(0,0)])
        info['width']  = abs(x2 - x1)
        info['height'] = abs(y2 - y1)

    # 讀取並更新 MLLM 校正計劃（僅 JSON）
    instr_path = os.path.join(INSTR_ROOT, image_name, INSTR_FILENAME)
    plans = []
    if os.path.isfile(instr_path):
        try:
            raw = open(instr_path, 'r', encoding='utf-8').read()
            instr = json.loads(extract_json_text(raw))
            plans = instr.get('Correction_Plan', [])
        except Exception as e:
            print(f"[WARNING] 解析 {instr_path} 失敗: {e}")

    # 處理校正計劃並標記 annotations
    for plan in plans:
        tool = plan.get('tool')
        action_str = plan.get('action', '')

        # Scene Text Edit
        if tool == 'Scene Text Edit' and action_str.lower() != 'no-op':
            for orig, target in re.findall(r"'([^']+)' to '([^']+)'", action_str):
                for info in annotations_box.values():
                    if info.get('text') == orig:
                        info['tool'] = tool
                        info['action'] = target
                        break

        # Scene Text Removal (改用整字比對，不分大小寫，包含單字母)
        elif tool == 'Scene Text Removal' and action_str.lower() != 'no-op':
            # 1️⃣ 拆出所有 removal phrases
            items_str = action_str.replace("Remove unwanted text", "").strip()
            phrases = [
                p.strip().strip("\"'")
                for p in items_str.split(',')
                if p.strip()
            ]
            # 2️⃣ 去重
            phrases = list(dict.fromkeys(phrases))

            # 3️⃣ 針對每個 phrase，用 regex \b...\b 整字比對
            for ph in phrases:
                pattern = rf"\b{re.escape(ph)}\b"
                for info in annotations_box.values():
                    txt = info.get('text', '').strip()
                    if re.search(pattern, txt, flags=re.IGNORECASE):
                        info['tool']   = tool
                        info['action'] = action_str
                        break

        # Scene Text Generate
        elif tool == 'Scene Text Generate':
            missing_texts = re.findall(r"'([^']+)'", action_str)
            if missing_texts and os.path.isfile(LAYOUT_TEMPLATE):
                tpl = open(LAYOUT_TEMPLATE, 'r', encoding='utf-8').read()
                combined = ' '.join(missing_texts)
                layout_content = tpl.replace(
                    '("Text")',
                    f"('{combined}')"
                )
                layout_out_dir = os.path.join(OUTPUT_BASE, image_name)
                os.makedirs(layout_out_dir, exist_ok=True)
                with open(os.path.join(layout_out_dir, 'Layout_input_prompt.txt'),
                          'w', encoding='utf-8') as lf:
                    lf.write(layout_content)
                print(f"  → Layout_input_prompt.txt 已更新為 The text you need generate:('{combined}')")

    # 輸出 annotations
    out_dir = os.path.join(OUTPUT_BASE, image_name)
    os.makedirs(out_dir, exist_ok=True)
    out_box = os.path.join(out_dir, 'annotations_box.json')
    with open(out_box, 'w', encoding='utf-8') as f:
        json.dump(annotations_box, f, ensure_ascii=False, indent=2)

    print(f"✅ {image_name} 處理完成，輸出到：{out_dir}")

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from pathlib import Path
from typing import Optional
import shutil

from lava_engine import LlavaEngine  # ← 直接用上面的引擎

USE_SYMLINK = True

TEMPLATE_FILE = Path("/home/pmd/Desktop/Alex/prompts/llava/MLLM_eval_prompt.txt")
FINAL_PROMPT_FILE = Path("/home/pmd/Desktop/Alex/prompts/llava/MLLM_eval_input.txt")
PROMPT_LIST_FILE = Path("/home/pmd/Desktop/Alex/Datasets/MARIOEval/MARIOEval/LAIONEval4000/LAIONEval4000_5414.txt")
IMAGE_DIR = Path("/home/pmd/Desktop/Alex/Datasets/Generate/Image1_Edit")
LAVA_FIXED_IMAGE = Path("/home/pmd/Desktop/Alex/Datasets/Generate/SD3_FID/sd3_LAIONEval4000_0_1.png")
OUTPUT_DIR = Path("/home/pmd/Desktop/Alex/Datasets/Generate/LLava_Image1/1")

def safe_mkdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def load_lines(p: Path):
    return [line.strip() for line in p.read_text(encoding="utf-8").splitlines()]

def read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8")

def write_text(p: Path, text: str):
    p.write_text(text, encoding="utf-8")

def replace_prompt_in_template(template_str: str, real_prompt: str) -> str:
    return template_str.replace("<prompt>", real_prompt)

def ensure_fixed_image_points_to(src_image: Path):
    safe_mkdir(LAVA_FIXED_IMAGE.parent)
    if LAVA_FIXED_IMAGE.exists() or LAVA_FIXED_IMAGE.is_symlink():
        LAVA_FIXED_IMAGE.unlink()
    if USE_SYMLINK:
        LAVA_FIXED_IMAGE.symlink_to(src_image)
    else:
        shutil.copy2(src_image, LAVA_FIXED_IMAGE)

def find_image_for_index(idx: int) -> Optional[Path]:
    patterns = [
        f"mineXgpt_LAIONEval4000_{idx}_1.jpg",
        f"mineXgpt_LAIONEval4000_{idx}_1.jpeg",
        f"mineXgpt_LAIONEval4000_{idx}_1.png",
        f"*_{idx}_*.jpg",
        f"*_{idx}_*.jpeg",
        f"*_{idx}_*.png",
    ]
    for pat in patterns:
        candidates = sorted(IMAGE_DIR.glob(pat))
        if candidates:
            return candidates[0]
    return None

def main():
    for path in [TEMPLATE_FILE, PROMPT_LIST_FILE, IMAGE_DIR]:
        if not path.exists():
            print(f"[ERROR] 找不到：{path}")
            return

    safe_mkdir(OUTPUT_DIR)
    log_file = OUTPUT_DIR / "run_log.txt"

    prompts = load_lines(PROMPT_LIST_FILE)
    template = read_text(TEMPLATE_FILE)
    total = len(prompts)
    print(f"[INFO] 共讀取 {total} 筆 prompt。開始執行。")

    engine = LlavaEngine()  # ← 模型只載入一次

    ok_count = 0
    skip_count = 0
    fail_count = 0

    with log_file.open("w", encoding="utf-8") as log:
        for i, real_prompt in enumerate(prompts):
            out_path = OUTPUT_DIR / f"{i}.txt"

            if out_path.exists():
                skip_count += 1
                continue

            filled_prompt = replace_prompt_in_template(template, real_prompt)
            write_text(FINAL_PROMPT_FILE, filled_prompt)  # 仍然保留可視化檢查

            img = find_image_for_index(i)
            if img is None:
                skip_count += 1
                log.write(f"[WARN] {i}: 找不到圖片\n")
                continue

            try:
                ensure_fixed_image_points_to(img)  # 可選：仍保留 symlink 給你外部工具檢查
                # 直接由引擎執行推論（不經 subprocess）
                result = engine.run(filled_prompt, img, max_new_tokens=200, do_sample=False)
                write_text(out_path, result)
                ok_count += 1

                percent = (i + 1) / total * 100
                print(f"[進度] {i+1}/{total} ({percent:.2f}%) → {out_path.name}")

            except Exception as e:
                fail_count += 1
                log.write(f"[FAIL] {i}: {e}\n")

    print(f"\n[SUMMARY] OK={ok_count}, SKIP={skip_count}, FAIL={fail_count}")
    print(f"[SUMMARY] 詳細記錄：{log_file}")

if __name__ == "__main__":
    main()

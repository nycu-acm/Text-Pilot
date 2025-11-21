#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from pathlib import Path
from typing import Optional
import shutil

from lava_engine import LlavaEngine  # ← 直接用上面的引擎

USE_SYMLINK = True

# 固定不變的來源
PROMPT_LIST_FILE = Path("/home/pmd/Desktop/Alex/Datasets/MARIOEval/MARIOEval/LAIONEval4000/LAIONEval4000_5414.txt")
IMAGE_DIR = Path("/home/pmd/Desktop/Alex/Datasets/Generate/Textdiffuser2_Edit")
LAVA_FIXED_IMAGE = Path("/home/pmd/Desktop/Alex/Datasets/Generate/SD3_FID/sd3_LAIONEval4000_0_1.png")

# 依 phase 切換的路徑規則
def phase_paths(phase: int):
    tmpl = Path(f"/home/pmd/Desktop/Alex/prompts/llava/{phase}/MLLM_eval_prompt.txt")
    final_in = Path(f"/home/pmd/Desktop/Alex/prompts/llava/{phase}/MLLM_eval_input.txt")
    out_dir = Path(f"/home/pmd/Desktop/Alex/Datasets/Generate/LLava_TD2/{phase}")
    return tmpl, final_in, out_dir

def safe_mkdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def load_lines(p: Path):
    return [line.strip() for line in p.read_text(encoding="utf-8").splitlines()]

def read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8")

def write_text(p: Path, text: str):
    safe_mkdir(p.parent)
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

def find_first_missing_index(out_dir: Path, total: int) -> int:
    """回傳 out_dir 中第一個缺少的 i.txt 索引（從 0 起算，最多到 total-1）。"""
    i = 0
    while i < total and (out_dir / f"{i}.txt").exists():
        i += 1
    return i

def run_phase(phase: int, engine: LlavaEngine, prompts, template: str):
    TEMPLATE_FILE, FINAL_PROMPT_FILE, OUTPUT_DIR = phase_paths(phase)

    # 檢查該 phase 的模板是否存在
    if not TEMPLATE_FILE.exists():
        print(f"[WARN][phase={phase}] 找不到模板：{TEMPLATE_FILE}，略過此階段。")
        return {"ok": 0, "skip": 0, "fail": 0, "phase": phase, "skipped": True}

    # 以該 phase 的實際模板內容覆蓋「template」變數
    try:
        template = read_text(TEMPLATE_FILE)
    except Exception as e:
        print(f"[WARN][phase={phase}] 讀取模板失敗：{TEMPLATE_FILE} → {e}，略過此階段。")
        return {"ok": 0, "skip": 0, "fail": 0, "phase": phase, "skipped": True}

    safe_mkdir(OUTPUT_DIR)
    log_file = OUTPUT_DIR / "run_log.txt"

    total = len(prompts)
    start_idx = find_first_missing_index(OUTPUT_DIR, total)

    if start_idx >= total:
        print(f"[INFO][phase={phase}] 輸出已齊全（0..{total-1}.txt 皆存在），略過此階段。")
        return {"ok": 0, "skip": total, "fail": 0, "phase": phase, "skipped": True}

    print(f"[INFO][phase={phase}] 從第 {start_idx} 筆開始（因為 0..{start_idx-1}.txt 已存在）。輸出到：{OUTPUT_DIR}")

    ok_count = 0
    skip_count = 0
    fail_count = 0

    with log_file.open("a", encoding="utf-8") as log:
        for i in range(start_idx, total):
            out_path = OUTPUT_DIR / f"{i}.txt"

            if out_path.exists():
                skip_count += 1
                print(f"[SKIP][phase={phase}] {out_path.name} 已存在，略過。")
                continue

            # 準備本筆資料
            real_prompt = prompts[i]
            filled_prompt = replace_prompt_in_template(template, real_prompt)
            write_text(FINAL_PROMPT_FILE, filled_prompt)  # 保留可視化檢查

            img = find_image_for_index(i)
            if img is None:
                skip_count += 1
                log.write(f"[WARN] {i}: 找不到圖片\n")
                print(f"[WARN][phase={phase}] {i}: 找不到圖片，略過。")
                continue

            try:
                ensure_fixed_image_points_to(img)
                result = engine.run(filled_prompt, img, max_new_tokens=200, do_sample=False)
                write_text(out_path, result)
                ok_count += 1

                percent = (i + 1) / total * 100
                print(f"[進度][phase={phase}] {i+1}/{total} ({percent:.2f}%) → {out_path.name}")

            except Exception as e:
                fail_count += 1
                log.write(f"[FAIL] {i}: {e}\n")
                print(f"[FAIL][phase={phase}] {i}: {e}")

    print(f"[SUMMARY][phase={phase}] OK={ok_count}, SKIP={skip_count}, FAIL={fail_count}")
    return {"ok": ok_count, "skip": skip_count, "fail": fail_count, "phase": phase, "skipped": False}

def main():
    # 共同必需資源檢查
    for path in [PROMPT_LIST_FILE, IMAGE_DIR]:
        if not path.exists():
            print(f"[ERROR] 找不到：{path}")
            return

    prompts = load_lines(PROMPT_LIST_FILE)

    # 先隨便讀一份模板以確保格式（實際每個 phase 會覆蓋）
    # 若 phase 1 的模板不存在也沒關係，run_phase 會各自檢查
    template_default = ""
    p1_template = Path("/home/pmd/Desktop/Alex/prompts/llava/1/MLLM_eval_prompt.txt")
    if p1_template.exists():
        template_default = read_text(p1_template)

    engine = LlavaEngine()  # 模型只載入一次

    grand = {"ok": 0, "skip": 0, "fail": 0}
    for phase in range(1, 6):
        stats = run_phase(phase, engine, prompts, template_default)
        grand["ok"] += stats["ok"]
        grand["skip"] += stats["skip"]
        grand["fail"] += stats["fail"]

    print("\n========== ALL PHASES SUMMARY ==========")
    print(f"TOTAL OK={grand['ok']}, SKIP={grand['skip']}, FAIL={grand['fail']}")
    print("========================================")

if __name__ == "__main__":
    main()


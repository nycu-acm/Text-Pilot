#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import shutil
import re
import time
import argparse
from pathlib import Path
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes, ComputerVisionOcrErrorException
from msrest.authentication import CognitiveServicesCredentials

# ------------------------------------------
# Configuration
# ------------------------------------------
subscription_key  = "Your API Key"
endpoint          = "Your End Point"

# Source / target directories
MYWORK_BASE    = Path("/home/pmd/Desktop/Alex/Datasets/MARIOEval/MARIOEval/LAIONEval4000/mywork_image1")
sd3_out        = Path("/home/pmd/Desktop/Alex/Datasets/Generate/Image1")
edit_out       = Path("/home/pmd/Desktop/Alex/Datasets/Generate/Image1_Edit")
ocr_output_dir = Path("/home/pmd/Desktop/Alex/Datasets/Eval/Image1")

# Flags: set to True to enable each step True False
RUN_STEP1 = True   # Flatten and copy images
RUN_STEP2 = True   # Perform OCR

# Ensure output directories exist
sd3_out.mkdir(parents=True, exist_ok=True)
edit_out.mkdir(parents=True, exist_ok=True)
ocr_output_dir.mkdir(parents=True, exist_ok=True)

# Characters to strip from OCR output
CHARS_TO_REMOVE = set(["'", '"', "/", "#", "\\", "%", "&", "@", "®", ","])

# Initialize Computer Vision client
gvision_client = ComputerVisionClient(endpoint, CognitiveServicesCredentials(subscription_key))

# Patterns for selecting the second image
SECOND_PATTERNS = [
    "SD3_1_4.png","SD3_1_3.png","SD3_1_2-gen*.png","SD3_1_2_AE-flux*.png","SD3_1_2_AE-textctrl*.png",
    "SD3_1_2_AE.png","SD3_1_2.png","SD3_1_1-gen*.png","SD3_1_1_AE-flux*.png",
    "SD3_1_1_AE-textctrl*.png","SD3_1_1_AE.png","output_flux.png","SD3_1_1.png",
]

# ------------------------------------------
# Step 1: Flatten and copy images
# ------------------------------------------
def flatten_and_copy(base_dir: Path):
    for folder in base_dir.iterdir():
        if not folder.is_dir() or not folder.name.isdigit():
            continue
        print(f"\nProcessing folder: {folder.name}")

        # Flatten nested subfolders
        for root, dirs, files in os.walk(folder):
            root_path = Path(root)
            if root_path == folder:
                continue
            for fname in files:
                if fname.lower().endswith('.png'):
                    src = root_path / fname
                    dst = folder / fname
                    if dst.exists():
                        stem, suf = dst.stem, dst.suffix
                        i = 1
                        while True:
                            candidate = folder / f"{stem}_{i}{suf}"
                            if not candidate.exists():
                                dst = candidate
                                break
                            i += 1
                    print(f"  Move: {src.name} -> {dst.name}")
                    shutil.move(str(src), str(dst))
        # Remove empty dirs
        for root, dirs, files in os.walk(folder, topdown=False):
            for d in dirs:
                try:
                    Path(root, d).rmdir()
                except OSError:
                    pass

        idx = int(folder.name) - 1
        # Copy first image
        src1 = folder / "SD3_1_1.png"
        if src1.exists():
            dst1 = sd3_out / f"gpt_LAIONEval4000_{idx}_1{src1.suffix}"
            print(f"  Copy first: {src1.name} -> {dst1.name}")
            shutil.copy(src1, dst1)
        else:
            print(f"  Missing: {src1}")

        # Copy second image
        second_src = None
        for pat in SECOND_PATTERNS:
            matches = sorted(folder.glob(pat))
            if matches:
                second_src = matches[0]
                break
        if second_src:
            dst2 = edit_out / f"mineXgpt_LAIONEval4000_{idx}_1{second_src.suffix}"
            print(f"  Copy second: {second_src.name} -> {dst2.name}")
            shutil.copy(second_src, dst2)
        else:
            print("  No second image found")

# ------------------------------------------
# Step 2: OCR on images with range support
# ------------------------------------------
def perform_ocr(input_dir: Path, output_dir: Path, start_idx: int, end_idx: int):
    for idx in range(start_idx, end_idx + 1):
        # 先把这一批图片都收集出来
        pattern = f"*_{idx}_*.png"
        img_paths = sorted(input_dir.glob(pattern))
        if not img_paths:
            continue  # 这一批根本没图，直接下一个 idx

        # 如果这一批中任意一张的 OCR 文件已经存在，就整个 idx 跳过
        already_done = any((output_dir / img.stem).exists() for img in img_paths)
        if already_done:
            print(f"索引 {idx} 的 OCR 結果已存在，整批跳過。")
            continue

        # 否則才对这一批图一张张做 OCR
        for img_path in img_paths:
            print(f"OCR reading {img_path.name}...")
            try:
                with open(img_path, 'rb') as stream:
                    read_resp = gvision_client.read_in_stream(stream, raw=True)
            except ComputerVisionOcrErrorException as e:
                print(f"Error on {img_path.name}: {e}")
                break

            op_loc = read_resp.headers.get("Operation-Location", "")
            if not op_loc:
                print(f"No operation location for {img_path.name}")
                break
            op_id = op_loc.split('/')[-1]

            # Poll until done
            while True:
                result = gvision_client.get_read_result(op_id)
                if result.status not in ('notStarted','running'):
                    break
                time.sleep(1)

            if result.status == OperationStatusCodes.succeeded:
                tokens = []
                for page in result.analyze_result.read_results:
                    for line in page.lines:
                        for word in line.text.split():
                            for part in re.split(r'[_\-]', word):
                                cleaned = ''.join(ch for ch in part if ch not in CHARS_TO_REMOVE)
                                if cleaned:
                                    tokens.append(cleaned)

                out_file = output_dir / img_path.stem
                with open(out_file, 'w', encoding='utf-8') as f:
                    f.write("\n".join(tokens))
                print(f"Saved OCR to {out_file.name}")
            else:
                print(f"OCR failed for {img_path.name}, status {result.status}")

# ------------------------------------------
# Main: parse --range and run accordingly
# ------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Flatten/Copy + OCR for idx range (START-END)")
    parser.add_argument('--range', type=str, required=True,
                        help="處理 OCR 的索引範圍，例如: 500-2500 或 1-1000")
    args = parser.parse_args()

    try:
        start_str, end_str = args.range.split('-')
        start_idx, end_idx = int(start_str), int(end_str)
    except Exception as e:
        parser.error("範圍格式錯誤，請使用 START-END，例如 1-1000")

    if start_idx < 0 or end_idx < start_idx:
        parser.error("請確保 START >= 1 且 END >= START")

    # Conditionally run steps
    if RUN_STEP1:
        flatten_and_copy(MYWORK_BASE)
    if RUN_STEP2:
        perform_ocr(sd3_out, ocr_output_dir, start_idx, end_idx)
        perform_ocr(edit_out, ocr_output_dir, start_idx, end_idx)

if __name__ == '__main__':
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse
from PIL import Image, ImageDraw

# -------------------------------------------------------
# 透過命令列輸入四個路徑參數，並提供預設值
# -------------------------------------------------------

def infer_mode(coord):
    n = len(coord)
    if n == 1:
        return "single"
    if n == 2:
        return "box"
    if n == 4:
        return "quad"
    return "polygon"


def paste_back(img, edited, coord, mode):
    """
    支援 single / box / quad / polygon，
    quad 和 polygon 都用多邊形遮罩方式貼回。
    """
    if mode == "box":
        (x1, y1), (x2, y2) = coord
        patch = edited.resize((x2 - x1, y2 - y1))
        img.paste(patch, (x1, y1))

    elif mode in ("quad", "polygon"):
        xs = [p[0] for p in coord]
        ys = [p[1] for p in coord]
        x1, y1 = min(xs), min(ys)
        x2, y2 = max(xs), max(ys)
        patch = edited.resize((x2 - x1, y2 - y1))

        pts = [tuple(p) for p in coord]
        mask = Image.new("L", img.size, 0)
        ImageDraw.Draw(mask).polygon(pts, fill=255)

        temp = img.copy()
        temp.paste(patch, (x1, y1))
        img = Image.composite(temp, img, mask)

    else:  # single
        x, y = coord[0]
        r = 4
        patch = edited.resize((2 * r, 2 * r))
        img.paste(patch, (x - r, y - r))

    return img


def main():
    parser = argparse.ArgumentParser(description="貼回小塊編輯結果至原始大圖")
    parser.add_argument("--original_img_path", type=str,
                        default="/home/pmd/Desktop/Alex/images/SD3/SD3_3.png",
                        help="原始大圖路徑 (default: %(default)s)")
    parser.add_argument("--coords_path", type=str,
                        default="/home/pmd/Desktop/Alex/images/OCR/SD3_3/Units/annotations.json",
                        help="annotations.json 路徑 (default: %(default)s)")
    parser.add_argument("--edited_dir", type=str,
                        default="/home/pmd/Desktop/Alex/Tools/STE/TextCtrl/example_result",
                        help="編輯後的小圖資料夾 (default: %(default)s)")
    parser.add_argument("--output_img_path", type=str,
                        default="/home/pmd/Desktop/Alex/images/Textctrl/edited.png",
                        help="輸出結果圖檔路徑 (default: %(default)s)")

    args = parser.parse_args()

    original_img_path = args.original_img_path
    coords_path       = args.coords_path
    edited_dir        = args.edited_dir
    output_img_path   = args.output_img_path

    # 1. 讀取原始大圖
    img = Image.open(original_img_path).convert("RGB")

    # 2. 讀取 annotations.json
    with open(coords_path, "r", encoding="utf-8") as f:
        ann = json.load(f)

    # 3. 列出編輯後的小圖檔名
    print("Edited files:", os.listdir(edited_dir))

    # 4. 逐一貼回
    for fname, entry in ann.items():
        coord = entry.get("coord", [])
        mode  = infer_mode(coord)
        edited_path = os.path.join(edited_dir, fname)
        if not os.path.exists(edited_path):
            print(f"[跳過] 找不到小圖：{edited_path}")
            continue
        edited = Image.open(edited_path).convert("RGB")
        img = paste_back(img, edited, coord, mode)

    # 5. 儲存結果
    os.makedirs(os.path.dirname(output_img_path), exist_ok=True)
    img.save(output_img_path)
    print(f"✅ 已完成貼回，輸出到：{output_img_path}")


if __name__ == "__main__":
    main()

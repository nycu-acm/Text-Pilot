#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import json
import re
import numpy as np
import torch
import cv2
from PIL import Image, ImageDraw, ImageFont, ImageOps
from tensorfn import load_arg_config
from tensorfn.config import instantiate
from units.config import E2EConfig
from units.dataset import MultitaskCollator
from units.structures import Sample
from units.transform import Compose

# -------------------------------------------------------
# Configuration
# -------------------------------------------------------
FONT_PATH   = "TC/Georgia.ttf"
FONT_SIZE   = 24
IMG_DIR     = "/home/pmd/Desktop/Alex/images/SD3"      # 原始圖片資料夾
OUTPUT_DIR  = "/home/pmd/Desktop/Alex/images/OCR"     # OCR 輸出根資料夾
BASE_IN_DIR = "/home/pmd/Desktop/Alex/images/Final_Output"

# -------------------------------------------------------
# Helper functions
# -------------------------------------------------------

def resize_instance(polygon, image_size, orig_size):
    """
    Undo the resizing + padding applied during preprocessing.
    """
    ratio = max(orig_size) / max(image_size)
    return polygon * ratio

def natural_key(name):
    """
    把 'SD3_1_12' 拆成 ['SD3', 1, 12] 這種格式，才能正確做數字比較。
    """
    parts = name.split('_')
    key = []
    for part in parts:
        if part.isdigit():
            key.append(int(part))
        else:
            m = re.match(r'^(\D+?)(\d+)$', part)
            if m:
                alpha, num = m.groups()
                key.append(alpha)
                key.append(int(num))
            else:
                key.append(part)
    return key


def draw_ocr(img, coords, texts, detect_type="quad", draw_width=5):
    font = ImageFont.truetype(FONT_PATH, size=FONT_SIZE)
    ocr_img = img.copy()
    draw = ImageDraw.Draw(ocr_img)

    for coord in coords:
        if detect_type in ["quad", "polygon"]:
            pts = [tuple(p) for p in coord]
            draw.polygon(pts, outline="red", width=draw_width)
        elif detect_type == "single":
            x, y = coord[0]
            r = draw_width
            draw.ellipse((x - r, y - r, x + r, y + r), outline="red", width=draw_width)
        else:  # 'box'
            (x1, y1), (x2, y2) = coord
            draw.rectangle([x1, y1, x2, y2], outline="red", width=draw_width)

    for coord, text in zip(coords, texts):
        size = font.getsize(text)
        px, py = coord[0]
        label_box = [px - 1, py - size[1] - 1, px + size[0] + 1, py + 1]
        draw.rectangle(label_box, fill=(0, 0, 0))
        draw.text((px, py - size[1]), text, font=font, fill=(255, 255, 255))

    return ocr_img


def run_ocr(img, model, mapper, transform, collator, device, detect_type):
    img = img.convert("RGB")
    o_w, o_h = img.size

    img_tensor, sample = transform(img, Sample(image_size=(o_h, o_w)))
    _, n_h, n_w = img_tensor.shape
    sample.image_size = (n_h, n_w)
    batch = collator([(img_tensor, sample)]).to(device)

    with torch.inference_mode():
        out_raw = model(batch, detect_type)
    out = mapper.postprocess(batch, out_raw)[0]
    if len(out.coords) == 0:
        return [], []
    coords = torch.stack(out.coords, 0).numpy()
    coords = resize_instance(coords, (n_h, n_w), (o_w, o_h))
    coords = coords.round().astype(int).tolist()

    return coords, out.texts


def centroid(coord):
    xs = [p[0] for p in coord]
    ys = [p[1] for p in coord]
    return (sum(xs) / len(xs), sum(ys) / len(ys))


def match_by_centroid(coords_ref, texts_ref, coords_target, texts_target):
    """
    按质心把 coords_target/texts_target 按 coords_ref 顺序重排
    """
    ref_c = [centroid(c) for c in coords_ref]
    tgt_c = [centroid(c) for c in coords_target]
    used = set()
    new_coords, new_texts = [], []
    for rc in ref_c:
        dists = [
            np.hypot(rc[0] - tc[0], rc[1] - tc[1]) if j not in used else np.inf
            for j, tc in enumerate(tgt_c)
        ]
        j = int(np.argmin(dists))
        used.add(j)
        new_coords.append(coords_target[j])
        new_texts.append(texts_target[j])
    return new_coords, new_texts

# -------------------------------------------------------
# Main
# -------------------------------------------------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    conf = load_arg_config(E2EConfig, elastic=False, show=True)
    model = instantiate(conf.model).to(device)

    ckpt = torch.load(conf.ckpt, map_location="cpu")
    state_dict = {}
    for k, v in ckpt["model"].items():
        name = k if conf.n_gpu > 1 else k.replace("module.", "", 1)
        state_dict[name] = v
    model.load_state_dict(state_dict)
    model.eval()

    mapper    = instantiate(conf.training.mappers)[0]
    transform = Compose(instantiate(conf.evaluate.transform))
    collator  = MultitaskCollator([], evaluate=True)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # list only the immediate subdirectories
    all_dirs = [
        d for d in os.listdir(BASE_IN_DIR)
        if os.path.isdir(os.path.join(BASE_IN_DIR, d))
    ]
    # sort by natural_key and take the last (max) name
    all_dirs.sort(key=natural_key)
    image_names = [all_dirs[-1]] if all_dirs else []
    # 2. 針對每個 image_name，到 Final_Output/{image_name}/SD 底下抓圖

    for image_name in image_names:
        input_dir = os.path.join(BASE_IN_DIR, image_name, "SD")
        for img_path in glob.glob(os.path.join(input_dir, "*.png")):
            img_name = image_name
            img = Image.open(img_path)
            img = ImageOps.exif_transpose(img)

            # —— 先跑 polygon —— #
            coords_poly, texts_poly = run_ocr(
                img, model, mapper, transform, collator, device, "polygon"
            )
            print(f"[DEBUG] {img_name} (polygon) got {len(coords_poly)} regions.")

            # —— 再跑 box —— #
            coords_box, texts_box = run_ocr(
                img, model, mapper, transform, collator, device, "box"
            )
            print(f"[DEBUG] {img_name} (box) got {len(coords_box)} regions.")

            # —— 用 polygon 的顺序去匹配 box —— #
            coords_box, texts_box = match_by_centroid(coords_poly, texts_poly, coords_box, texts_box)

            # —— 按相同索引依次处理两种类型 —— #
            for detect_type, (coords, texts) in {
                "polygon": (coords_poly, texts_poly),
                "box":     (coords_box, texts_box),
            }.items():
                print(f"Processing {img_name} [{detect_type}]...")

                # overlay
                overlay = draw_ocr(img, coords, texts, detect_type=detect_type)
                base_out = os.path.join(OUTPUT_DIR, img_name, "Units", detect_type)
                os.makedirs(base_out, exist_ok=True)
                overlay_path = os.path.join(base_out, f"{img_name}_ocr_overlay.png")
                overlay.save(overlay_path)
                print(f"[完成] {img_name} ({detect_type}) overlay -> {overlay_path}")

                # 剪裁與 prompts
                cropped_dir = os.path.join(base_out, "cropped")
                os.makedirs(cropped_dir, exist_ok=True)
                annotations = {}

                for i, (coord, text) in enumerate(zip(coords, texts)):
                    fn = f"box_{i:02d}.png"
                    try:
                        if detect_type == "polygon":
                            # 1) 計算外接矩形
                            xs = [p[0] for p in coord]
                            ys = [p[1] for p in coord]
                            x1, x2 = min(xs), max(xs)
                            y1, y2 = min(ys), max(ys)
                            w, h = x2 - x1, y2 - y1

                            # 2) 裁取矩形區域並轉為 RGBA
                            region = img.crop((x1, y1, x2, y2)).convert("RGBA")

                            # 3) 建立多邊形 mask 並填充
                            mask = Image.new("L", (w, h), 0)
                            draw_m = ImageDraw.Draw(mask)
                            shifted = [(px - x1, py - y1) for px, py in coord]
                            draw_m.polygon(shifted, fill=255)

                            # 4) 以 mask 作為 alpha 通道
                            region.putalpha(mask)

                            # 5) 保存帶透明背景的 PNG
                            region.save(os.path.join(cropped_dir, fn), format="PNG")
                        else:  # box
                            (x1, y1), (x2, y2) = coord
                            margin = 1
                            x1, y1 = max(x1+margin, 0), max(y1+margin, 0)
                            x2, y2 = min(x2-margin, img.width), min(y2-margin, img.height)
                            crop_img = img.crop((x1, y1, x2, y2)).convert("RGB")
                            crop_img.save(os.path.join(cropped_dir, fn), format="PNG", compress_level=0)

                        annotations[fn] = {"coord": coord, "text": text}
                    except Exception as e:
                        print(f"[WARN] crop failed for region {i}: {e}")

                # masks
                for i, coord in enumerate(coords):
                    mask_img = Image.new("L", img.size, 0)
                    draw_m = ImageDraw.Draw(mask_img)
                    if detect_type == "polygon":
                        draw_m.polygon([tuple(p) for p in coord], fill=255)
                    else:
                        (x1, y1), (x2, y2) = coord
                        draw_m.rectangle([x1, y1, x2, y2], fill=255)
                    mask_img.save(os.path.join(base_out, f"white_{i:02d}.png"))

                # write annotations
                with open(os.path.join(base_out, "annotations.json"), "w", encoding="utf-8") as f:
                    json.dump(annotations, f, indent=2, ensure_ascii=False)

                print(f"[完成] {img_name} ({detect_type}) -> {base_out}")

    print("所有圖片處理完畢。")


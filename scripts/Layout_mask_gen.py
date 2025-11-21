#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import json
from json import JSONDecoder, JSONDecodeError
import re
import sys
from pathlib import Path
from PIL import Image, ImageDraw

CANVAS_SIZE = (1024, 1024)  # 固定输出尺寸


def safe_filename(name: str) -> str:
    """
    将任意字符串转换为文件友好的名称：
    把所有不是英数、下划线、破折号或点号的字符替换为下划线
    """
    return re.sub(r'[^A-Za-z0-9._-]', '_', name)


def load_json(json_path: Path):
    # 1) 读取文件，去 BOM，并保留所有内容
    raw = json_path.read_text(encoding='utf-8-sig')
    # 2) 移除可能的代码块标记
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.DOTALL).strip()
    if not raw:
        print(f"[Warning] JSON is empty: {json_path.name}")
        return []

    # 3) 尝试一次性解析整个 JSON（支持 object 或 array）
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            return [data]
        if isinstance(data, list):
            return data
    except JSONDecodeError:
        pass

    # 4) 后备：连续解析多个 JSON object
    entries = []
    decoder = JSONDecoder()
    idx = 0
    length = len(raw)
    while idx < length:
        fragment = raw[idx:].lstrip()
        if not fragment:
            break
        try:
            obj, offset = decoder.raw_decode(fragment)
            entries.append(obj)
            consumed = len(raw[idx:]) - len(fragment) + offset
            idx += consumed
        except JSONDecodeError:
            break

    if not entries:
        print(f"[Error] 无法解析任何 JSON 对象: {json_path.name}")
    return entries


def main():
    parser = argparse.ArgumentParser(
        description="Generate 1024×1024 mask images from JSON coordinates."
    )
    parser.add_argument(
        '--json_path',
        default="/home/pmd/Desktop/Alex/images/SD3/output.txt",
        help='Path to JSON file with coordinates (object, list, or multiple objects).'
    )
    parser.add_argument(
        '--output_base',
        default='/home/pmd/Desktop/Alex/images/SD3',
        help='Base directory for output masks.'
    )
    args = parser.parse_args()

    json_path = Path(args.json_path)
    if not json_path.exists():
        print(f"[Error] JSON file not found: {json_path}")
        sys.exit(1)

    mask_dir = Path(args.output_base)
    mask_dir.mkdir(parents=True, exist_ok=True)

    entries = load_json(json_path)
    if not entries:
        print("No valid entries, exiting Generate Mask.")
        return

    width, height = CANVAS_SIZE
    for item in entries:
        raw_text = str(item.get('text', 'mask')).strip()
        # 去除尖括号
        text = raw_text.strip('<>')
        if text.lower() in ('no-op', 'noop'):
            print(f"  ⚠️ 跳過無效鍵: '{text}'")
            continue

        pos = item.get('position', {})
        try:
            x1, y1 = pos['top_left']
            x2, y2 = pos['bottom_right']
        except Exception:
            print(f"  [Warning] 坐标格式错误，跳过: {item}")
            continue

        # 裁剪到画布范围内
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width-1, x2), min(height-1, y2)

        mask = Image.new('L', CANVAS_SIZE, 0)
        ImageDraw.Draw(mask).rectangle([x1, y1, x2, y2], fill=255)

        fname = safe_filename(text)
        out_path = mask_dir / f"{fname}.png"
        mask.save(str(out_path))
        print(f"Saved {width}×{height} mask for '{text}' at: {out_path}")


if __name__ == '__main__':
    main()

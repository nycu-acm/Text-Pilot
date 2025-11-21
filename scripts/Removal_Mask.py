#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import argparse
from PIL import Image, ImageDraw


def main():
    parser = argparse.ArgumentParser(
        description="生成合并所有指定多边形区域的总遮罩图。"
    )
    parser.add_argument(
        "--annotations",
        required=True,
        help="原始 polygon annotations.json 路径"
    )
    parser.add_argument(
        "--keys",
        required=True,
        help="要合并的 box_key，以空格分隔"
    )
    parser.add_argument(
        "--output",
        required=True,
        help="输出 total_mask.png 路径"
    )
    args = parser.parse_args()

    # 读取 annotations 文件
    with open(args.annotations, encoding="utf-8") as f:
        data = json.load(f)

    keys = args.keys.split()

    # 如需根据原图动态获取尺寸，可在此改为打开图片，获取 img.size
    # 示例固定画布大小为 1024×1024，可根据实际调整
    canvas_width, canvas_height = 1024, 1024
    mask = Image.new("L", (canvas_width, canvas_height), 0)
    draw = ImageDraw.Draw(mask)

    for key in keys:
        ann = data.get(key)
        if not ann:
            print(f"⚠️ 找不到 key：{key}，已跳过。")
            continue

        # 取多边形坐标字段 'coord'
        poly = ann.get("coord")
        if not isinstance(poly, list) or len(poly) < 2:
            print(f"⚠️ key {key} 的 coord 数据不足: {poly}，已跳过。")
            continue

        # 转换为整数坐标
        coords = []
        for pt in poly:
            try:
                x, y = int(pt[0]), int(pt[1])
                coords.append((x, y))
            except Exception as e:
                print(f"⚠️ 转换 {key} 坐标 {pt} 失败: {e}")
        if len(coords) < 2:
            print(f"⚠️ key {key} 转换后点数不足，已跳过。")
            continue

        # 绘制填充多边形
        draw.polygon(coords, fill=255)
        print(f"✔ 已添加多边形: {key}, 共 {len(coords)} 点")

    # 保存遮罩图
    mask.save(args.output)
    print(f"✅ 总遮罩已保存到: {args.output}")


if __name__ == "__main__":
    main()

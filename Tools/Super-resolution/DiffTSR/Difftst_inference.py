#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# —— Monkey-patch distutils.version.LooseVersion ——
import types, distutils
from packaging.version import Version as LooseVersion
distutils.version = types.SimpleNamespace(LooseVersion=LooseVersion)
# ————————————————————————————————
from pathlib import Path
import os
import torch
from PIL import Image
from omegaconf import OmegaConf
from model.IDM.utils.util import instantiate_from_config
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Batch super-resolution with DiffTSR')
    parser.add_argument(
        '--config', type=str,
        default='./model/DiffTSR_config.yaml',
        help='Path to DiffTSR config YAML'
    )
    parser.add_argument(
        '--ckpt', type=str,
        default='./ckpt/DiffTSR.ckpt',
        help='Path to DiffTSR checkpoint'
    )
    parser.add_argument(
        '--input_dir', type=str,
        default='./testset/0_lr_synth/',
        help='Directory containing low-resolution input images'
    )
    parser.add_argument(
        '--output_dir', type=str,
        default='./testset/0_sr_synth/',
        help='Directory to save super-resolved images'
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Load config and model
    config = OmegaConf.load(args.config)
    model = instantiate_from_config(config.model)
    model.load_model(args.ckpt)

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process each image
    for fname in sorted(os.listdir(input_dir)):
        in_path = input_dir / fname
        out_path = output_dir / fname

        # Open and record original size
        with Image.open(in_path) as img:
            img_rgb = img.convert('RGB')
            orig_size = img_rgb.size  # (width, height)

        # Resize to model input size
        lq = img_rgb.resize((512, 128), Image.LANCZOS)

        # Inference super-resolution
        with torch.no_grad():
            sr_array = model.DiffTSR_sample(lq)
        sr = Image.fromarray(sr_array, 'RGB')

        # Resize back to original size
        sr_resized = sr.resize(orig_size, Image.LANCZOS)

        # Save result
        sr_resized.save(out_path)
        print(f"{fname} → super-resolved then resized to {orig_size}, saved to {out_path}")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import logging
import os
import sys
import traceback

import cv2
import numpy as np
import torch
import tqdm
import yaml
from omegaconf import OmegaConf
from torch.utils.data._utils.collate import default_collate

from saicinpainting.evaluation.utils import move_to_device
from saicinpainting.evaluation.refinement import refine_predict
from saicinpainting.training.trainers import load_checkpoint
from saicinpainting.utils import register_debug_signal_handlers

LOGGER = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(
        description="LAMA inpainting inference: specify model, input, mask, output folders and output key"
    )
    p.add_argument(
        "--model_path", "-m", required=True,
        help="Path to the checkpoint folder (must contain config.yaml and models/<ckpt>)"
    )
    p.add_argument(
        "--checkpoint", "-c", required=True,
        help="Name of the checkpoint file under models/, e.g. ckpt.pth"
    )
    p.add_argument(
        "--input_dir", "-i", required=True,
        help="Folder containing input images (e.g. 1.png, 2.png, …)"
    )
    p.add_argument(
        "--mask_dir", "-k", required=True,
        help="Folder containing mask images (e.g. 1_mask.png, 2_mask.png, …), can be separate from input_dir"
    )
    p.add_argument(
        "--output_dir", "-o", default="./output",
        help="Where to write the predicted inpainted images"
    )
    p.add_argument(
        "--out_ext", default="_AE.png",
        help="Extension for output files (default: .png)"
    )
    p.add_argument(
        "--out_key", default="inpainted",
        help="Key in model output dict to use for the inpainted tensor (default: inpainted)"
    )
    p.add_argument(
        "--refine", action="store_true",
        help="Whether to run the refinement stage instead of direct predict"
    )
    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    try:
        # single-thread for numpy/OpenBLAS
        os.environ['OMP_NUM_THREADS'] = '1'
        os.environ['OPENBLAS_NUM_THREADS'] = '1'
        os.environ['MKL_NUM_THREADS'] = '1'
        os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
        os.environ['NUMEXPR_NUM_THREADS'] = '1'

        if sys.platform != 'win32':
            register_debug_signal_handlers()

        # load train config
        cfg_file = os.path.join(args.model_path, 'config.yaml')
        with open(cfg_file, 'r') as f:
            train_cfg = OmegaConf.create(yaml.safe_load(f))
        train_cfg.training_model.predict_only = True
        train_cfg.visualizer.kind = 'noop'

        # load model
        ckpt_path = os.path.join(args.model_path, 'models', args.checkpoint)
        model = load_checkpoint(train_cfg, ckpt_path, strict=False, map_location='cpu')
        model.freeze()
        if not args.refine:
            model.to(torch.device("cpu"))

        # gather input / mask file lists
        inp_files = sorted([
            fn for fn in os.listdir(args.input_dir)
            if fn.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])
        if not inp_files:
            LOGGER.critical(f"No input images found in {args.input_dir}")
            sys.exit(1)

        mask_files = [
            fn.replace(os.path.splitext(fn)[1], "_mask" + os.path.splitext(fn)[1])
            for fn in inp_files
        ]
        # check existence
        for mf in mask_files:
            if not os.path.exists(os.path.join(args.mask_dir, mf)):
                LOGGER.critical(f"Mask file missing: {os.path.join(args.mask_dir, mf)}")
                sys.exit(1)

        os.makedirs(args.output_dir, exist_ok=True)

        # inference loop
        for img_name, mask_name in tqdm.tqdm(zip(inp_files, mask_files), total=len(inp_files)):
            inp_path = os.path.join(args.input_dir, img_name)
            msk_path = os.path.join(args.mask_dir, mask_name)

            # read image & mask
            img = cv2.imread(inp_path, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
            msk = cv2.imread(msk_path, cv2.IMREAD_GRAYSCALE)
            msk = (msk > 0).astype(np.uint8)

            # build batch
            batch = {
                'image': torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float(),
                'mask':  torch.from_numpy(msk).unsqueeze(0).unsqueeze(0).float()
            }
            if args.refine:
                # refinement stage
                res = refine_predict(batch, model)[0]
            else:
                with torch.no_grad():
                    batch = move_to_device(batch, torch.device("cpu"))
                    res_batch = model(batch)
                    if args.out_key not in res_batch:
                        LOGGER.critical(
                            f"Output key '{args.out_key}' not found. Available keys: {list(res_batch.keys())}"
                        )
                        sys.exit(1)
                    res = res_batch[args.out_key][0]

            # formatting
            res_np = res.permute(1, 2, 0).cpu().numpy()
            res_np = np.clip(res_np * 255, 0, 255).astype(np.uint8)
            res_bgr = cv2.cvtColor(res_np, cv2.COLOR_RGB2BGR)

            base = os.path.splitext(img_name)[0]
            out_path = os.path.join(args.output_dir, base + args.out_ext)
            cv2.imwrite(out_path, res_bgr)

        LOGGER.info("All done.")

    except KeyboardInterrupt:
        LOGGER.warning('Interrupted by user')
    except Exception as ex:
        LOGGER.critical(f'Prediction failed due to {ex}:\n{traceback.format_exc()}')
        sys.exit(1)


if __name__ == '__main__':
    main()

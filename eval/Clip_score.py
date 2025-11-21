#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import warnings
from pathlib import Path

import clip
import torch
import tqdm
import numpy as np
import sklearn.preprocessing
from packaging import version
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

# -- JSON generation -------------------------------------------------------

def generate_image2prompt(image_dir: Path, prompt_txt: Path, output_json: Path):
    """
    Scan image_dir for *.png files named method_dataset_pidx_iidx,
    read prompt_txt lines, map pidx to prompt, and save to output_json.
    """
    prompt_lines = [line.rstrip("\n") for line in prompt_txt.open("r", encoding="utf-8")]
    mapping = {}
    for img_path in image_dir.glob("*.png"):
        stem = img_path.stem
        parts = stem.split("_")
        if len(parts) != 4:
            continue
        _, dataset, pidx, _ = parts
        try:
            idx = int(pidx)
        except ValueError:
            continue
        if 0 <= idx < len(prompt_lines):
            mapping[stem] = prompt_lines[idx]
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)
    print(f"✅ Wrote {len(mapping)} entries to {output_json}")

# -- CLIPScore helpers -----------------------------------------------------

class CLIPCapDataset(torch.utils.data.Dataset):
    def __init__(self, caps, prefix="A photo depicts "):
        self.data = caps
        self.prefix = prefix if prefix.endswith(" ") else prefix + " "
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        token = clip.tokenize(self.prefix + self.data[idx], truncate=True).squeeze()
        return {"caption": token}

class CLIPImageDataset(torch.utils.data.Dataset):
    def __init__(self, paths):
        self.paths = paths
        self.prep = Compose([
            Resize(224, interpolation=Image.BICUBIC),
            CenterCrop(224),
            lambda img: img.convert("RGB"),
            ToTensor(),
            Normalize((0.48145466,0.4578275,0.40821073),
                      (0.26862954,0.26130258,0.27577711)),
        ])
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, idx):
        img = Image.open(self.paths[idx])
        return {"image": self.prep(img)}

def extract_all_images(paths, model, device, batch_size=64, num_workers=8):
    ds = CLIPImageDataset(paths)
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size,
                                         shuffle=False, num_workers=num_workers)
    feats = []
    with torch.no_grad():
        for batch in tqdm.tqdm(loader):
            imgs = batch["image"].to(device)
            if device.startswith("cuda"):
                imgs = imgs.to(torch.float16)
            feats.append(model.encode_image(imgs).cpu().numpy())
    return np.vstack(feats)

def extract_all_captions(caps, model, device, batch_size=256, num_workers=8):
    ds = CLIPCapDataset(caps)
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size,
                                         shuffle=False, num_workers=num_workers)
    feats = []
    with torch.no_grad():
        for batch in tqdm.tqdm(loader):
            txt = batch["caption"].to(device)
            feats.append(model.encode_text(txt).cpu().numpy())
    return np.vstack(feats)

def get_clip_score(image_feats, text_feats, w=2.5):
    if version.parse(np.__version__) < version.parse("1.21"):
        image_feats = sklearn.preprocessing.normalize(image_feats, axis=1)
        text_feats  = sklearn.preprocessing.normalize(text_feats,  axis=1)
    else:
        image_feats = image_feats / np.linalg.norm(image_feats, axis=1, keepdims=True)
        text_feats  = text_feats  / np.linalg.norm(text_feats,  axis=1, keepdims=True)
    per = w * np.clip(np.sum(image_feats * text_feats, axis=1), 0, None)
    return np.mean(per), per

# -- Main workflow ---------------------------------------------------------

def main():
    prompt_txt = Path(
    "/home/pmd/Desktop/Alex/Datasets/MARIOEval/MARIOEval"
    "/LAIONEval4000/LAIONEval4000_Clip.txt"
    )
    # 两个目录及其 JSON 输出
    dirs = {
        "edit": Path("/home/pmd/Desktop/Alex/Datasets/Generate/Edit_FID"),
        "sd3":  Path("/home/pmd/Desktop/Alex/Datasets/Generate/SD3_FID"),
    }
    json_base = Path("//home/pmd/Desktop/Alex/Datasets/Generate")

    # 生成 JSON
    for name, dpath in dirs.items():
        out_json = json_base / f"image2prompt_{name}.json"
        generate_image2prompt(dpath, prompt_txt, out_json)

    # 加载 CLIP
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = clip.load("ViT-B/32", device=device, jit=False)
    model.eval()

    # 计算并打印 CLIPScore
    for name, dpath in dirs.items():
        json_path = json_base / f"image2prompt_{name}.json"
        with json_path.open("r", encoding="utf-8") as f:
            mapping = json.load(f)
        image_ids   = list(mapping.keys())
        captions    = list(mapping.values())
        image_paths = [str(dpath / f"{iid}.png") for iid in image_ids]

        print(f"\n▶️  Computing CLIPScore for '{name}'")
        img_feats = extract_all_images(image_paths, model, device)
        txt_feats = extract_all_captions(captions,   model, device)
        mean_score, _ = get_clip_score(img_feats, txt_feats)

        print(f"  Mean CLIPScore (w=2.5): {mean_score:.4f}")
        print(f"  Approx cosine*100: {mean_score/2.5*100:.2f}")

if __name__ == "__main__":
    main()

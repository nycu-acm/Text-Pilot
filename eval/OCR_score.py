#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import copy

# ground-truth collections
gts = {
    'LAIONEval4000': [],
}

# metrics containers (if you still want to compute them)
results = {
    'gpt': {'cnt': 0, 'p': 0, 'r': 0, 'f': 0, 'acc': 0},
    'mineXgpt':      {'cnt': 0, 'p': 0, 'r': 0, 'f': 0, 'acc': 0},
}

# collect all OCR tokens not in ground-truth
typos = []

def get_key_words(text: str):
    """
    Extract words enclosed in single or double quotes from the line
    """
    words = []
    # --- Change: Now matches single or double quotes ---
    matches = re.findall(r"['\"](.*?)['\"]", text)
    for match in matches:
        words.extend(match.split())
    return words

def get_p_r_acc(pred, gt):
    """
    Compute precision, recall, and exact-match accuracy
    Also return the set of differing tokens
    """
    pred_norm = [p.strip().lower() for p in pred]
    gt_norm   = [g.strip().lower() for g in gt]

    # compute metrics
    pred_orig = copy.deepcopy(pred_norm)
    gt_orig   = copy.deepcopy(gt_norm)
    for p in pred_norm:
        if p in gt_orig:
            pred_orig.remove(p)
            gt_orig.remove(p)

    precision = (len(pred_norm) - len(pred_orig)) / (len(pred_norm) + 1e-8)
    recall    = (len(gt_norm)   - len(gt_orig))   / (len(gt_norm)   + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)
    acc       = int(sorted(pred_norm) == sorted(gt_norm))

    return precision, recall, f1, acc, set(pred_norm).symmetric_difference(set(gt_norm))

# Load ground-truth data
# --- IMPORTANT: Make sure this path is correct ---
gt_root = '/home/pmd/Desktop/Alex/Datasets/MARIOEval/MARIOEval'
for ds in gts:
    txt_file = os.path.join(gt_root, ds, f'{ds}.txt')
    if not os.path.exists(txt_file):
        print(f"Warning: Ground-truth file not found at {txt_file}")
        continue
    with open(txt_file, 'r', encoding='utf-8') as fp:
        for line in fp:
            gts[ds].append(get_key_words(line.strip().lower()))

# Process OCR results and compare
# --- IMPORTANT: Make sure this path is correct ---
ocr_root = '/home/pmd/Desktop/Alex/Datasets/Eval/Image1'
for fname in os.listdir(ocr_root):
    # skip the typo.txt we will write later
    if fname == 'typo.txt':
        continue

    # Use os.path.splitext to handle filenames correctly
    name, _ = os.path.splitext(fname)
    parts = name.split('_')
    if len(parts) != 4:
        # not in expected format, skip
        print(f"Skipping file with unexpected format: {fname}")
        continue
    method, dataset, pidx_str, iidx = parts

    # Ensure pidx is a valid integer
    try:
        pidx = int(pidx_str)
    except ValueError:
        print(f"Skipping file with non-integer pidx: {fname}")
        continue

    # Read OCR tokens (one per line)
    path = os.path.join(ocr_root, fname)
    with open(path, 'r', encoding='utf-8') as f:
        ocr_tokens = [l.strip() for l in f if l.strip()]

    # Flatten tokens for typo checking
    print_tokens = []
    for tok in ocr_tokens:
        print_tokens.extend(tok.split())

    # --- ERROR FIX STARTS HERE ---
    # Retrieve GT tokens (all lowercase)
    gt_list = gts.get(dataset)
    if not gt_list or pidx >= len(gt_list):
        print(f"Warning: Index {pidx} is out of range for dataset '{dataset}'. File: {fname}. Skipping.")
        continue
    gt_tokens = gt_list[pidx]
    # --- ERROR FIX ENDS HERE ---

    # Optional: compute metrics
    p, r, f1, acc, _ = get_p_r_acc(ocr_tokens, gt_tokens)
    if method in results:
        stats = results[method]
        stats['cnt'] += 1
        stats['p']   += p
        stats['r']   += r
        stats['acc'] += acc

    # Collect tokens not in ground-truth
    gt_set = set(gt_tokens)
    photo_name = f"{method}_{dataset}_{pidx}_{iidx}"
    for token in print_tokens:
        if token.lower() not in gt_set:
            typos.append(f"{photo_name}: {token}")

# Write all mismatched tokens to typo.txt
typo_file = os.path.join(ocr_root, 'typo.txt')
with open(typo_file, 'w', encoding='utf-8') as f:
    for line in typos:
        f.write(line + '\n')

print(f"✅ Saved all mismatched tokens to {typo_file}\n")

# Optional: print aggregate metrics
print("Aggregate Results:")
for method, stats in results.items():
    if stats['cnt'] > 0:
        stats['p'] /= stats['cnt']
        stats['r'] /= stats['cnt']
        stats['f'] = 2 * stats['p'] * stats['r'] / (stats['p'] + stats['r'] + 1e-8)
        stats['acc'] /= stats['cnt']
    print(f"Method: {method}")
    print(f"  Precision: {stats['p']:.4f}")
    print(f"  Recall:    {stats['r']:.4f}")
    print(f"  F1:        {stats['f']:.4f}")
    print(f"  Acc:       {stats['acc']:.4f}\n")


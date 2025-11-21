#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import copy
from collections import defaultdict

# --- Configuration ---
# Define the order of methods to check. This is crucial for the cascade logic.
METHOD_ORDER = [f"sd3X{i}" for i in range(1, 7)] 

# Paths (ensure these are correct)
GT_ROOT = '/home/pmd/Desktop/Alex/Datasets/MARIOEval/MARIOEval'
OCR_ROOT = '/home/pmd/Desktop/Alex/Datasets/Eval/OCR_results'
GT_DATASET = 'LAIONEval4000'

# --- Helper Functions (from previous script) ---
def get_key_words(text: str):
    words = []
    matches = re.findall(r"['\"](.*?)['\"]", text)
    for match in matches:
        words.extend(match.split())
    return words

def is_correct(pred, gt):
    """Checks for exact match accuracy."""
    pred_norm = sorted([p.strip().lower() for p in pred])
    gt_norm   = sorted([g.strip().lower() for g in gt])
    return pred_norm == gt_norm

# --- Main Logic ---

# 1. Load Ground Truth
gt_file = os.path.join(GT_ROOT, GT_DATASET, f'{GT_DATASET}.txt')
ground_truths = []
with open(gt_file, 'r', encoding='utf-8') as fp:
    for line in fp:
        ground_truths.append(get_key_words(line.strip().lower()))
total_prompts = len(ground_truths)
print(f"Loaded {total_prompts} ground truth prompts.\n")


# 2. Group OCR result files by prompt index (pidx)
grouped_files = defaultdict(list)
print("Scanning OCR result files...")
for fname in os.listdir(OCR_ROOT):
    name, _ = os.path.splitext(fname)
    parts = name.split('_')
    if len(parts) != 4:
        continue
    method, dataset, pidx_str, iidx = parts
    try:
        pidx = int(pidx_str)
        if dataset == GT_DATASET:
             grouped_files[pidx].append({'method': method, 'path': os.path.join(OCR_ROOT, fname)})
    except ValueError:
        continue

# 3. Simulate the cascade process and gather stats
solved_by = defaultdict(int)
unsolved_count = 0

for pidx in range(total_prompts):
    gt_tokens = ground_truths[pidx]
    is_prompt_solved = False
    
    if pidx not in grouped_files:
        unsolved_count += 1
        continue
        
    # Check methods in the specified order
    for method_name in METHOD_ORDER:
        # Find the file for the current method in the group
        method_file = next((f for f in grouped_files[pidx] if f['method'] == method_name), None)
        
        if method_file:
            with open(method_file['path'], 'r', encoding='utf-8') as f:
                ocr_tokens = [l.strip() for l in f if l.strip()]
            
            if is_correct(ocr_tokens, gt_tokens):
                solved_by[method_name] += 1
                is_prompt_solved = True
                break # <-- This is the "stop when correct" logic
    
    if not is_prompt_solved:
        unsolved_count += 1

# 4. Print the final analysis report
print("\n--- Cumulative Contribution Analysis Report ---")
print(f"Based on a total of {total_prompts} prompts.\n")

cumulative_solved = 0
print(f"{'Method':<10} | {'Newly Solved':<15} | {'Cumulative Solved':<20} | {'Cumulative Accuracy':<20}")
print("-" * 75)

for method_name in METHOD_ORDER:
    newly_solved = solved_by.get(method_name, 0)
    cumulative_solved += newly_solved
    cumulative_acc = (cumulative_solved / total_prompts) * 1000
    
    print(f"{method_name:<10} | {newly_solved:<15} | {cumulative_solved:<20} | {cumulative_acc:.2f}%")

print("-" * 75)
print(f"{'Unsolved':<10} | {unsolved_count:<15} |")
print("\nReport Explanation:")
print("- 'Newly Solved': The number of prompts solved for the FIRST time by this method.")
print("- 'Cumulative Solved': The total number of unique prompts solved up to this point.")
print("- 'Cumulative Accuracy': The overall success rate of the system after this step.\n")

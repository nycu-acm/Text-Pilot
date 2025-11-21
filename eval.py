#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Eval Script: iterate through a list of prompts (default from LAIONEval4000.txt), write each to SD3_prompt.txt,
then execute Agent.py for each prompt.
"""
import subprocess
import shlex
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Agent.py on a list of SD3 prompts"
    )
    parser.add_argument(
        "--prompts_file", "-p",
        default="/home/pmd/Desktop/Alex/Datasets/MARIOEval/MARIOEval/LAIONEval4000/LAIONEval4000_sd3.txt",
        help="Path to the txt file containing one prompt per line (default LAIONEval4000.txt)"
    )
    parser.add_argument(
        "--sd3_prompt", "-s",
        default="/home/pmd/Desktop/Alex/prompts/SD3_prompt.txt",
        help="Path to SD3_prompt.txt to overwrite for each run"
    )
    parser.add_argument(
        "--agent_script", "-a",
        default="Agent.py",
        help="Agent script to execute after writing prompt"
    )
    args = parser.parse_args()

    prompts_path = Path(args.prompts_file)
    sd3_path = Path(args.sd3_prompt)
    agent_path = Path(args.agent_script)

    if not prompts_path.exists():
        print(f"Error: prompts file not found: {prompts_path}")
        return
    if not agent_path.exists():
        print(f"Error: Agent script not found: {agent_path}")
        return

    # Read all prompts
    with prompts_path.open('r', encoding='utf-8') as f:
        prompts = [line.strip() for line in f if line.strip()]

    total = len(prompts)
    print(f"Starting evaluation: {total} prompts found.")

    for idx, prompt in enumerate(prompts, start=1):
        print(f"\n[{idx}/{total}] Writing prompt and running Agent.py...")
        # Write prompt to SD3_prompt.txt
        sd3_path.write_text(prompt, encoding='utf-8')
        print(f"  -> Wrote prompt to {sd3_path}")

        # Execute Agent.py
        cmd = f"python {shlex.quote(str(agent_path))}"
        try:
            subprocess.run(cmd, shell=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error: Agent.py failed on prompt #{idx} with exit code {e.returncode}")

    print("\n✅ Evaluation complete.")

if __name__ == "__main__":
    main()
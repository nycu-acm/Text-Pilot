#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import shutil
from pathlib import Path

# 要清空子資料夾的根目錄清單（只刪子資料夾，不碰檔案）
ROOT_DIRS = [
    "/home/pmd/Desktop/Alex/images/STR",
    "/home/pmd/Desktop/Alex/images/STG",
    "/home/pmd/Desktop/Alex/images/STE",
    "/home/pmd/Desktop/Alex/images/OCR",
    "/home/pmd/Desktop/Alex/images/Final_Output",
    "/home/pmd/Desktop/Alex/prompts",
]

# 要刪除所有檔案的資料夾
SD3_DIR = "/home/pmd/Desktop/Alex/images/SD3"

def delete_subdirectories(root_path: Path):
    """
    刪除 root_path 底下所有的子資料夾（並遞迴刪除其內容）
    不會刪除 root_path 本身，也不會碰檔案。
    """
    for child in root_path.iterdir():
        if child.is_dir():
            try:
                shutil.rmtree(child)
                print(f"刪除資料夾: {child}")
            except Exception as e:
                print(f"無法刪除資料夾 {child}：{e}")

def delete_files_only(folder_path: Path):
    """
    刪除 folder_path 底下所有的檔案，不刪子資料夾
    """
    for child in folder_path.iterdir():
        if child.is_file():
            try:
                child.unlink()
                print(f"刪除檔案: {child}")
            except Exception as e:
                print(f"無法刪除檔案 {child}：{e}")
        else:
            print(f"保留資料夾: {child}")

def main():
    # 1. 清空指定目錄底下的所有子資料夾
    for dir_path in ROOT_DIRS:
        p = Path(dir_path)
        if p.exists() and p.is_dir():
            delete_subdirectories(p)
        else:
            print(f"路徑不存在或不是資料夾: {p}")

    # 2. 清空 SD3_DIR 底下的所有檔案
    sd3 = Path(SD3_DIR)
    if sd3.exists() and sd3.is_dir():
        delete_files_only(sd3)
    else:
        print(f"路徑不存在或不是資料夾: {sd3}")

if __name__ == "__main__":
    main()

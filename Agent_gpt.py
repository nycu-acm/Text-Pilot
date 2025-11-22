#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import subprocess
import shlex
import json
import shutil
import re
from pathlib import Path
import subprocess

# ========= 1. 在此處設定開關，用 True ／ False 控制是否執行對應任務 =========
RUN_SD3            = False
RUN_TEXT_DETECT    = True
RUN_MLLM           = True
RUN_REMOVAL        = True  
RUN_EDIT           = True
RUN_GENERATE       = True 
# =============================================================================
RUN_TEXT_RECOGNIZE = False

class EnvTask:
    def __init__(self, env_name: str, commands: list[str]):
        self.env_name = env_name
        self.commands = commands

    def run(self):
        lines = [
            "source ~/anaconda3/etc/profile.d/conda.sh",
            f"conda activate {self.env_name}",
        ]
        lines.extend(self.commands)
        lines.append(
            "python - <<'EOF'\n"
            "try:\n"
            "    import torch\n"
            "    torch.cuda.empty_cache()\n"
            "except ImportError:\n"
            "    pass\n"
            "EOF"
        )
        bash_script = "\n".join(lines)
        subprocess.run(['bash', '-lc', bash_script], check=True)

def extract_json_text(raw: str) -> str:
    # 如果整個 JSON 被 ``` 或 ```json 包住，直接取中間的 {...}
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, flags=re.DOTALL)
    if m:
        return m.group(1)
    # 否則找第一個 { 到最後一個 } 之間的內容
    start = raw.find('{')
    end   = raw.rfind('}')
    if start != -1 and end != -1 and end > start:
        return raw[start:end+1]
    # 回退：去除前後空白
    return raw.strip()

def main():
    try:
        print("===== 任務啟動狀態 =====")
        print(f"{'啟動' if RUN_SD3 else '跳過'} SD3 生成步驟。")
        print(f"{'啟動' if RUN_TEXT_DETECT else '跳過'} Units（OCR 模型）步驟。")
        print(f"{'啟動' if RUN_TEXT_RECOGNIZE else '跳過'} PaddleOCR 文字辨識步驟。")
        print(f"{'啟動' if RUN_MLLM else '跳過'} MLLM 推理步驟。")
        print(f"{'啟動' if RUN_REMOVAL else '跳過'} Scene Text Removal 步驟。")
        print(f"{'啟動' if RUN_EDIT else '跳過'} JSON/TextCtrl/Flux 步驟。")
        print(f"{'啟動' if RUN_GENERATE else '跳過'} Scene Text Generate 步驟。")
        print("=========================")

        # 路徑設定
        sd3_script     = Path("/home/pmd/Desktop/Alex/Tools/SD3/stable-diffusion-3.5-large/Sd3_inference.py")
        ocr_img_dir    = Path("/home/pmd/Desktop/Alex/images/OCR") 
        units_dir      = "/home/pmd/Desktop/Alex/Tools/OCR/Units_Detector/units"
        paddleocr_dir  = "/home/pmd/Desktop/Alex/Tools/OCR/PaddleOCR"
        textctrl_dir   = "/home/pmd/Desktop/Alex/Tools/STE/TextCtrl"
        edit_base_dir  = Path("/home/pmd/Desktop/Alex/prompts")
        final_base     = Path("/home/pmd/Desktop/Alex/images/Final_Output")
        flux_dir       = "/home/pmd/Desktop/Alex/Tools/STE/FLUX"
        ae_dir         = Path("/home/pmd/Desktop/Alex/Tools/STR/Lama")
        scripts_dir    = Path("/home/pmd/Desktop/Alex/scripts")
        ste_base       = Path("/home/pmd/Desktop/Alex/images/STE")

        # 1. SD3 生成（僅初次執行一次原始圖像生成）
        if RUN_SD3:
            print("執行 SD3 生成步驟…")
            subprocess.run([
                'bash', '-lc',
                (
                    "source ~/anaconda3/etc/profile.d/conda.sh && "
                    "conda activate sd3 && "
                    f"python {shlex.quote(str(sd3_script))} "
                    "--prompt $(< /home/pmd/Desktop/Alex/prompts/SD3_prompt.txt)"
                )
            ], check=True)

        # 收集生成的圖像資料夾名稱
        image_names = [p.name for p in final_base.iterdir() if p.is_dir()]

        # 逐一處理每個圖像，評估並進行必要的多輪修正
        for image_name in image_names:
            base_name = image_name
            print(f"開始處理圖像資料夾: {base_name}")
            iteration = 1
            total_score = 0

            # 持續迭代，直到文字偵測評分達到 4 為止
            while True:
                if iteration > 2:
                    print(f"⚠️ 迭代次數已達 {iteration-1} 次，超過上限 3 次，結束修正迴圈。")
                    break
                if iteration > 1:
                    print(f"🔄 第 {iteration} 輪修正後重新檢測與評估（圖像: {image_name}）...")
                else:
                    print(f"🔍 初次 OCR 檢測與評估（圖像: {image_name}）...")

                # 2. Units OCR 檢測
                if RUN_TEXT_DETECT:
                    print("啟動 Units 環境並執行 OCR 檢測…")
                    # 組出本輪要處理的 SD 資料夾路徑
                    img_dir_units = final_base / image_name / "SD"
                    EnvTask(
                        env_name="units",
                        commands=[
                            # 加上 --img_dir 參數
                            f"cd {shlex.quote(units_dir)} && PYTHONPATH=$PWD python script/Units_inference.py "
                            f"--conf configs/finetune.py --ckpt weights/shared.pt "
                        ]
                    ).run()

                # 3. PaddleOCR 文字辨識
                if RUN_TEXT_RECOGNIZE:
                    print("啟動 PaddleOCR 環境並執行文字辨識…")
                    EnvTask(
                        env_name="Paddleocr",
                        commands=[
                            "cd {} && python Paddle_Inference.py "
                            "--main_image_dir {} "
                            "--tools_ocr_root {} "
                            "--output_root {}".format(
                                shlex.quote(str(paddleocr_dir)),
                                shlex.quote(f"/home/pmd/Desktop/Alex/images/Final_Output/{image_name}/SD"),
                                shlex.quote(str(ocr_img_dir)),
                                shlex.quote(str(ocr_img_dir)),
                            )
                        ]
                    ).run()

                # 4. MLLM 評估（OpenAI 推理）
                if RUN_MLLM:
                    print(f"啟動 OPENAI 環境，對圖像 {image_name} 執行 MLLM 推理評估…")
                    # 根據最新 OCR 結果產生 MLLM 輸入 prompt
                    subprocess.run([
                        'bash', '-lc',
                        # 直接把当前的 image_name 传进去
                        f"cd scripts && python LLM_fill_prompt.py --image_name {image_name}"
                    ], check=True)
                    # 執行 OpenAI 推理
                    subprocess.run([
                        'bash', '-lc',
                        f"python /home/pmd/Desktop/Alex/scripts/OpenAI.py "
                        f"--input_txt_path /home/pmd/Desktop/Alex/prompts/{image_name}/MLLM_input_prompt.txt "
                        f"--img_path /home/pmd/Desktop/Alex/images/Final_Output/{image_name}/SD/*.png "
                        f"--output_txt_path /home/pmd/Desktop/Alex/prompts/{image_name}/MLLM_output_instruction.txt"
                    ], check=True)
            
                # 更新 JSON（將 MLLM 輸出寫入標註檔）
                EnvTask(
                    env_name="base311",
                    commands=[f"cd scripts && python JSON_Update.py --image_name {image_name}"]
                ).run()
                # 讀取 MLLM 輸出，解析 Total_Score
                instr_path = edit_base_dir / image_name / "MLLM_output_instruction.txt"
                if instr_path.exists():
                    raw_output = instr_path.read_text(encoding="utf-8-sig")
                    clean_json = extract_json_text(raw_output)
                    try:
                        result_data = json.loads(clean_json)
                    except json.JSONDecodeError:
                        result_data = {}
                    total_score = result_data.get("Total_Score", 0)
                    print(f"圖像 {image_name} 評估 Total_Score = {total_score}")
                else:
                    print(f"⚠️ 無法找到 {instr_path}，預設 Total_Score = 0")
                    total_score = 0
                # 如果得分滿分 (4)，退出迴圈（圖像文字表現已很好）
                if total_score >= 4:
                    print(f"✅ 圖像 {image_name} 的文字品質得分 {total_score}，已達理想狀態，結束修正迴圈。")
                    break

                # 若得分不足 4，執行 Scene Text Manipulation 修正流程
                print(f"🔧 圖像 {image_name} 評分 {total_score}，開始進行文字修正...")

                # 5. Scene Text Removal（依照 Correction_Plan 將不需要的文字去除）
                if RUN_REMOVAL:
                    coords_box_path = edit_base_dir / image_name / "annotations_box.json"
                    box_data = {}
                    if coords_box_path.exists():
                        box_data = json.loads(coords_box_path.read_text(encoding="utf-8"))
                    removal_items = [(k, info) for k, info in box_data.items() if info.get("tool") == "Scene Text Removal"]
                    if removal_items:
                        sd_dir_for_image = final_base / image_name / "SD"
                        sd_images = sorted(sd_dir_for_image.glob("*.png"))
                        if sd_images:
                            source_image = sd_images[0]  # 使用當前圖像作為移除文字輸入
                            # 讀取 OCR 偵測出的多邊形標註
                            poly_ann_path = ocr_img_dir / image_name / "Units" / "polygon" / "annotations.json"
                            if poly_ann_path.exists():
                                # 準備遮罩生成
                                mask_keys = [k for k, _ in removal_items]
                                mask_keys_str = " ".join(mask_keys)
                                mask_output_dir = Path("/home/pmd/Desktop/Alex/images/STR") / image_name
                                mask_output_dir.mkdir(parents=True, exist_ok=True)
                                total_mask_path = mask_output_dir / "total_mask.png"
                                # 產生總遮罩（融合所有需移除區域）
                                subprocess.run([
                                    'bash', '-lc',
                                    f"cd {shlex.quote(str(scripts_dir))} && python Removal_Mask.py "
                                    f"--annotations {shlex.quote(str(poly_ann_path))} "
                                    f"--keys {shlex.quote(mask_keys_str)} "
                                    f"--output {shlex.quote(str(total_mask_path))}"
                                ], check=True)
                                print(f"   ✅ 已生成移除區域總遮罩：{total_mask_path.name}")
                                # 在呼叫 EnvTask(...) 之前，先確保 STR 輸出資料夾已經存在
                                final_str_dir = final_base / image_name / "STR"
                                final_str_dir.mkdir(parents=True, exist_ok=True)

                                # 原來的 AE 推理命令，確保 output_path 指向剛剛建立的目錄
                                output_path = final_str_dir / f"{image_name}_AE.png"

                                EnvTask(
                                    env_name="lama",
                                    commands=[
                                        # 一行搞定：先切到 Lama 根目录，设 PYTHONPATH，
                                        # 然后把 total_mask.png 复制（或链接）为 1_mask.png 等脚本需要的名字，
                                        # 最后再跑 predict.py
                                        (
                                            f"cd {ae_dir} && "
                                            f"export PYTHONPATH={ae_dir}/big-lama/src:$PYTHONPATH && "
                                            # 把 total_mask.png 复制成脚本要找的 <stem>_mask<ext>
                                            f"cp {total_mask_path} {total_mask_path.parent}/{source_image.stem}_mask{source_image.suffix} && "
                                            # 真正调用 predict.py
                                            f"python bin/predict.py "
                                            f"--model_path {ae_dir}/big-lama "
                                            f"--checkpoint best.ckpt "
                                            f"--input_dir {source_image.parent} "
                                            f"--mask_dir {total_mask_path.parent} "
                                            f"--output_dir {final_str_dir}"
                                        )
                                    ]
                                ).run()

                            else:
                                print(f"⚠️ 無法找到 OCR 標註檔 {poly_ann_path}，跳過 Removal 步驟。")
                        else:
                            print("⚠️ 找不到待處理的 SD 圖像，跳過文字移除步驟。")
                    else:
                        print("無需移除任何文字。")

                # 6. Scene Text Edit（修正錯誤文本）
                if RUN_EDIT:
                    # 6a. 使用 TextCtrl 修正小文字框並超解析
                    coords_box_path = edit_base_dir / image_name / "annotations_box.json"
                    box_data = {}
                    if coords_box_path.exists():
                        box_data = json.loads(coords_box_path.read_text(encoding="utf-8"))
                    edit_items = [(k, info) for k, info in box_data.items() if info.get("tool") == "Scene Text Edit"]
                    textctrl_base = ste_base / image_name / "Textctrl"
                    output_dir = textctrl_base / "output"
                    super_output_dir = textctrl_base / "super_output"
                    i_s_dir = textctrl_base / "i_s"
                    textctrl_base.mkdir(parents=True, exist_ok=True)
                    output_dir.mkdir(parents=True, exist_ok=True)
                    super_output_dir.mkdir(parents=True, exist_ok=True)
                    i_s_dir.mkdir(parents=True, exist_ok=True)
                    # 準備 TextCtrl 模型的輸入檔（小圖及對應文本）
                    open(textctrl_base / "i_s.txt", "w").close()
                    open(textctrl_base / "i_t.txt", "w").close()
                    small_edit_performed = False
                    for box_key, info in edit_items:
                        w, h = info.get("width", 0), info.get("height", 0)
                        if w <= 75 or h <= 50:  # 只處理較小的文字框
                            small_edit_performed = True
                            # 複製該框的小圖作為輸入
                            src_crop = ocr_img_dir / image_name / "Units" / "box" / "cropped" / box_key
                            if src_crop.exists():
                                shutil.copy(src_crop, i_s_dir / src_crop.name)
                            # 寫入原文與修改後文字
                            with open(textctrl_base / "i_s.txt", "a", encoding="utf-8") as fs:
                                fs.write(f"{box_key} {info.get('text', '')}\n")
                            with open(textctrl_base / "i_t.txt", "a", encoding="utf-8") as ft:
                                ft.write(f"{box_key} {info.get('action', '')}\n")
                    if small_edit_performed:
                        # 執行 TextCtrl 模型推理修正小框文字
                        print("執行 TextCtrl 模型修正小文字框…")
                        EnvTask(
                            env_name="textctrl",
                            commands=[
                                f"cd {shlex.quote(str(textctrl_dir))} && python Textctrl_inference.py "
                                f"--dataset_dir={shlex.quote(str(textctrl_base))} --output_dir={shlex.quote(str(output_dir))}"
                            ]
                        ).run()
                        # 執行 DiffTSR 超解析提升小框文字品質
                        print("執行 DiffTSR 超解析提升文字清晰度…")
                        EnvTask(
                            env_name="DiffTSR",
                            commands=[
                                "cd /home/pmd/Desktop/Alex/Tools/Super-resolution/DiffTSR && conda activate DiffTSR && "
                                f"python Difftst_inference.py --input_dir {shlex.quote(str(output_dir))} --output_dir {shlex.quote(str(super_output_dir))}"
                            ]
                        ).run()
                    else:
                        print("無需進行小文字框 TextCtrl 修正。")

                    # 6b. 將修正後的小框影像貼回原圖
                    base_image_path = None
                    # 確定貼回所用的原圖（優先使用最新移除了文字的圖）
                    str_dir = final_base / image_name / "STR"
                    str_imgs = sorted(str_dir.glob("*.png")) if str_dir.exists() else []
                    if str_imgs:
                        base_image_path = str_imgs[-1]  # 使用移除文字後的圖像作為貼圖基底
                    else:
                        sd_dir = final_base / image_name / "SD"
                        sd_imgs = sorted(sd_dir.glob("*.png"))
                        if sd_imgs:
                            base_image_path = sd_imgs[0]
                    # 準備包含小框修正項目的標註 JSON（僅保留小框）
                    filtered_path = textctrl_base / "filtered_annotations.json"
                    filtered_data = {k: v for k, v in box_data.items()
                                    if v.get("tool") == "Scene Text Edit" and (v.get("width", 0) <= 75 or v.get("height", 0) <= 50)}
                    filtered_path.write_text(json.dumps(filtered_data, ensure_ascii=False, indent=2), encoding="utf-8")
                    # 如果有進行小框修正，執行貼回操作
                    if base_image_path and small_edit_performed:
                        ste_output_dir = final_base / image_name / "STE"
                        ste_output_dir.mkdir(parents=True, exist_ok=True)
                        existing_files = list(ste_output_dir.glob(f"{base_image_path.stem}-textctrl*{base_image_path.suffix}"))
                        output_index = len(existing_files) + 1
                        pasted_image_path = ste_output_dir / f"{base_image_path.stem}-textctrl{output_index}{base_image_path.suffix}"
                        EnvTask(
                            env_name="base311",
                            commands=[
                                f"cd {shlex.quote(str(scripts_dir))} && python Textctrl_paste_flow.py "
                                f"--original_img_path {shlex.quote(str(base_image_path))} "
                                f"--coords_path {shlex.quote(str(filtered_path))} "
                                f"--edited_dir {shlex.quote(str(super_output_dir))} "
                                f"--output_img_path {shlex.quote(str(pasted_image_path))}"
                            ]
                        ).run()
                        print(f"   ✅ 小框文字已貼回至圖像：{pasted_image_path.name}")
                        base_image_path = pasted_image_path

                    # 6c. 使用 Flux 修正較大範圍的文字錯誤
                    # 決定 Flux 模型的初始輸入圖（經過以上步驟修正後的最新圖像）
                    flux_input = None
                    ste_dir_full = final_base / image_name / "STE"
                    ste_imgs = sorted(ste_dir_full.glob("*textctrl*.png")) if ste_dir_full.exists() else []
                    if base_image_path:
                        flux_input = base_image_path
                    elif ste_imgs:
                        flux_input = ste_imgs[-1]
                    elif str_imgs:
                        flux_input = str_imgs[-1]
                    else:
                        sd_dir = final_base / image_name / "SD"
                        sd_imgs = sorted(sd_dir.glob("*.png"))
                        if sd_imgs:
                            flux_input = sd_imgs[0]
                    flux_folder = ste_base / image_name / "Flux"
                    flux_folder.mkdir(parents=True, exist_ok=True)
                    flux_base = flux_input.stem.split("-textctrl")[0] if flux_input else "image"
                    flux_count = 0
                    for box_key, info in edit_items:
                        w, h = info.get("width", 0), info.get("height", 0)
                        if not (w < 75 or h < 50):  # 僅處理較大文字區域
                            mask_idx = Path(box_key).stem.split("_")[-1]
                            mask_file = ocr_img_dir / image_name / "Units" / "box" / f"white_{mask_idx}.png"
                            if flux_input and mask_file.exists():
                                flux_count += 1
                                flux_output_path = flux_folder / f"{flux_base}-flux{flux_count}{flux_input.suffix}"
                                EnvTask(
                                    env_name="Flux",
                                    commands=[
                                        f"cd {shlex.quote(flux_dir)} && python EditFlux_Inference.py "
                                        f"--image_path {shlex.quote(str(flux_input))} "
                                        f"--mask_path {shlex.quote(str(mask_file))} "
                                        f"--prompt \"Generating text '{info.get('action','')}'.\" "
                                        # f"--prompt \"Edit the text '{info.get('action','')}' in this mask.\" "
                                        f"--output_path {shlex.quote(str(flux_output_path))}"
                                    ]
                                ).run()
                                print(f"   ✅ Flux 修正完成：框 {box_key} -> {info.get('action','')}")
                                # 更新下一輪 Flux 輸入圖像
                                flux_input = flux_output_path
                    # 如果進行了 Flux 修正，將最終結果存到 Final_Output/STE
                    if flux_count > 0 and flux_input:
                        final_flux_img = flux_input
                        final_ste_dir = final_base / image_name / "STE"
                        final_ste_dir.mkdir(parents=True, exist_ok=True)
                        shutil.copy(final_flux_img, final_ste_dir / final_flux_img.name)
                        print(f"   ✅ 大範圍文字已修正，輸出圖像：{final_flux_img.name}")
                    else:
                        print("無需進行大範圍文字 Flux 修正或未產生新的修正輸出。")

                # 7. Scene Text Generate（產生缺失文字）
                if RUN_GENERATE:
                    gen_action = None
                    instr_file = edit_base_dir / image_name / "MLLM_output_instruction.txt"
                    if instr_file.exists():
                        raw_instr = instr_file.read_text(encoding="utf-8-sig")
                        clean_instr = extract_json_text(raw_instr)
                        try:
                            plan_data = json.loads(clean_instr)
                        except json.JSONDecodeError:
                            plan_data = {}
                        plan_list = plan_data.get("Correction_Plan", [])
                        gen_item = next((p for p in plan_list if p.get("tool") == "Scene Text Generate" and p.get("action")), None)
                        if gen_item:
                            gen_action = gen_item.get("action", "").strip()
                    # 如果沒有需要生成的文字，跳過該步驟
                    if not gen_action:
                        print("無需額外文字生成。")
                    else:
                        # 檢查是否為 no-op
                        action_lower = gen_action.strip("\"'").lower()
                        if "no-op" in action_lower or "<no-op>" in action_lower:
                            print("Scene Text Generate 指令為 no-op，跳過文字生成。")
                        else:
                            print("✨ 執行 Scene Text Generate 以產生缺失文字...")
                            layout_prompt = edit_base_dir / image_name / "Layout_input_prompt.txt"
                            output_path = edit_base_dir / image_name / "Layout_output_instruction.txt"
                            # 7a. 調用 OpenAI 生成文字佈局
                            if layout_prompt.exists():
                                # 選擇最新修正圖像作為生成參考
                                flux_candidates = sorted((final_base / image_name / "STE").glob("*flux*.png"))
                                textctrl_candidates = sorted((final_base / image_name / "STE").glob("*textctrl*.png"))
                                str_candidates = sorted((final_base / image_name / "STR").glob("*.png")) if (final_base / image_name / "STR").exists() else []
                                if flux_candidates:
                                    layout_input_image = flux_candidates[-1]
                                elif textctrl_candidates:
                                    layout_input_image = textctrl_candidates[-1]
                                elif str_candidates:
                                    layout_input_image = str_candidates[-1]
                                else:
                                    sd_candidates = sorted((final_base / image_name / "SD").glob("*.png"))
                                    layout_input_image = sd_candidates[0] if sd_candidates else None
                                if layout_input_image:
                                    subprocess.run([
                                        'bash', '-lc',
                                        f"python /home/pmd/Desktop/Alex/scripts/OpenAI_layout.py "
                                        f"--input_txt {shlex.quote(str(layout_prompt))} "
                                        f"--img_path {shlex.quote(str(layout_input_image))} "
                                        f"--output_txt_path {shlex.quote(str(output_path))}"
                                    ], check=True)
                            # 7b. 根據 OpenAI 版面配置輸出，產生遮罩圖
                            layout_output = edit_base_dir / image_name / "Layout_output_instruction.txt"
                            if layout_output.exists():
                                mask_dir = Path("/home/pmd/Desktop/Alex/images/STG") / image_name / "mask"
                                mask_dir.mkdir(parents=True, exist_ok=True)
                                subprocess.run([
                                    'bash', '-lc',
                                    f"python /home/pmd/Desktop/Alex/scripts/Layout_mask_gen.py "
                                    f"--json_path {shlex.quote(str(layout_output))} "
                                    f"--output_base {shlex.quote(str(mask_dir))}"
                                ], check=True)
                            # 7c. 使用 Flux 在圖像上生成新文字
                            mask_files = sorted((Path("/home/pmd/Desktop/Alex/images/STG") / image_name / "mask").glob("*.png"))
                            if mask_files:
                                # 先防呆：檔案存在且非空才讀
                                layout_output = edit_base_dir / image_name / "Layout_output_instruction.txt"
                                instructions = []
                                if layout_output.exists() and layout_output.stat().st_size > 0:
                                    raw = layout_output.read_text(encoding="utf-8-sig")
                                    clean = extract_json_text(raw)
                                    try:
                                        instructions = json.loads(clean)
                                        if isinstance(instructions, dict):
                                            instructions = [instructions]
                                    except json.JSONDecodeError:
                                        print(f"⚠️ 無法解析 JSON ({layout_output.name})，前 200 字：\n{raw[:200]}…，跳過文字生成。")
                                else:
                                    print(f"⚠️ 找不到或檔案為空：{layout_output.name}，跳過文字生成。")

                                # 選擇最新修正圖像作為生成基底
                                flux_inputs = sorted((final_base / image_name / "STE").glob("*flux*.png"))
                                textctrl_inputs = sorted((final_base / image_name / "STE").glob("*textctrl*.png"))
                                str_inputs = sorted((final_base / image_name / "STR").glob("*.png")) if (final_base / image_name / "STR").exists() else []
                                if flux_inputs:
                                    gen_flux_input = flux_inputs[-1]
                                elif textctrl_inputs:
                                    gen_flux_input = textctrl_inputs[-1]
                                elif str_inputs:
                                    gen_flux_input = str_inputs[-1]
                                else:
                                    sd_inputs = sorted((final_base / image_name / "SD").glob("*.png"))
                                    gen_flux_input = sd_inputs[0] if sd_inputs else None

                                if not gen_flux_input or not instructions:
                                    print("⚠️ 缺少基底圖或指令，跳過新文字生成。")
                                else:
                                    gen_output_dir = Path("/home/pmd/Desktop/Alex/images/STG") / image_name / "output"
                                    gen_output_dir.mkdir(parents=True, exist_ok=True)
                                    final_stg_dir = final_base / image_name / "STG"
                                    final_stg_dir.mkdir(parents=True, exist_ok=True)

                                    for idx, mask_path in enumerate(mask_files, start=1):
                                        if idx-1 < len(instructions):
                                            text_to_generate = instructions[idx-1].get("text", "").strip()
                                        else:
                                            print(f"⚠️ Layout 指令裡沒有第 {idx} 項，跳過。")
                                            continue

                                        gen_output_image = gen_output_dir / f"{image_name}-gen{idx}{gen_flux_input.suffix}"
                                        print(f"   ➕ 產生新文字區域 #{idx}: '{text_to_generate}'")
                                        EnvTask(
                                            env_name="Flux",
                                            commands=[
                                            f"cd {shlex.quote(flux_dir)} && python EditFlux_Inference.py "
                                            f"--image_path {shlex.quote(str(gen_flux_input))} "
                                            f"--mask_path {shlex.quote(str(mask_path))} "
                                            f"--prompt \"Generating text '{text_to_generate}'\" "
                                            f"--output_path {shlex.quote(str(gen_output_image))}"
                                        ]).run()
                                        # 累積效果：下一輪輸入為這輪輸出
                                        gen_flux_input = gen_output_image

                                    # 最後把生成結果複製到最終輸出
                                    final_generated_img = gen_flux_input
                                    shutil.copy(final_generated_img, final_stg_dir / final_generated_img.name)
                                    print(f"   ✅ 缺失文字已生成，最終圖像輸出：{final_generated_img.name}")
                            else:
                                print("沒有生成任何遮罩，跳過新文字生成步驟。")

                # 將本輪修正完的圖像作為下一輪輸入（覆蓋 SD 資料夾中的圖像）
                # 確定最新的最終圖像（優先順序：STG -> STE -> STR）
                updated_image_path = None
                stg_pngs = sorted((final_base / image_name / "STG").glob("*.png"))
                if stg_pngs:
                    updated_image_path = stg_pngs[-1]
                else:
                    # 2. STE
                    ste_dir = final_base / image_name / "STE"
                    if ste_dir.exists():
                        # 2.1 flux*.png
                        flux_pngs = sorted(ste_dir.glob(f"{image_name}-flux*.png"))
                        if flux_pngs:
                            updated_image_path = flux_pngs[-1]
                        else:
                            # 2.2 textctrl*.png
                            text_pngs = sorted(ste_dir.glob(f"{image_name}-textctrl*.png"))
                            if text_pngs:
                                updated_image_path = text_pngs[-1]
                            else:
                                # 2.3 其它 png
                                other_pngs = sorted(ste_dir.glob("*.png"))
                                if other_pngs:
                                    updated_image_path = other_pngs[-1]

                    # 3. STR（只有在 STE 也没找到时才检查）
                    if updated_image_path is None:
                        str_pngs = sorted((final_base / image_name / "STR").glob("*.png"))
                        if str_pngs:
                            updated_image_path = str_pngs[-1]

                if not updated_image_path:
                    print("⚠️ 无法找到修正后的图像，无法进行下一轮迭代。")
                    break
                # ========== manipulation 完成後，放在這裡 ==========
                # 確保 original_base、iteration 已經在迴圈外設定好
                # 迭代結束後
                root_name = "_".join(image_name.split("_")[:2])
                iteration += 1
                new_name = f"{root_name}_{iteration}"
                dst_dir = final_base / new_name            # 新資料夾
                sd_dir  = dst_dir / "SD"                   # 新的 SD 子資料夾
                # 1. 建立新資料夾及它的 SD 子資料夾
                sd_dir.mkdir(parents=True, exist_ok=True)
                # 2. 複製 updated_image_path 到 dst_dir/SD
                new_filename = f"{new_name}{updated_image_path.suffix}"               # e.g. "SD3_1_2.png"
                dst_img_path  = sd_dir / new_filename
                shutil.copy2(updated_image_path, dst_img_path)
                print(f"   ✅ 已將 {updated_image_path.name} 重新命名為 {new_filename} 並複製至 {sd_dir}")
                # 3. 更新 image_name 指向新的資料夾
                image_name = new_name
                print(f"🔄 複製完成：{updated_image_path.name} → {dst_img_path}，下一輪針對 {sd_dir} 繼續跑。")
            # ===============================================
    finally:
        print("✅ 全部階段任務執行完畢。")
        target_root = Path("/home/pmd/Desktop/Alex/Datasets/MARIOEval/MARIOEval/LAIONEval4000/mywork_image_regen")
        final_base = Path("/home/pmd/Desktop/Alex/images/Final_Output")
        prompt_file = Path("/home/pmd/Desktop/Alex/prompts/SD3_prompt.txt")
        prompts_folder = Path("/home/pmd/Desktop/Alex/prompts")

        target_root.mkdir(parents=True, exist_ok=True)

        # 找第一個不存在的編號
        idx = 1
        while True:
            dst = target_root / str(idx)
            if not dst.exists():
                break
            idx += 1

        # 複製 Final_Output 整個資料夾
        shutil.copytree(final_base, dst)
        # 複製 SD3_prompt.txt
        shutil.copy(prompt_file, dst / prompt_file.name)
        # 複製整個 prompts 資料夾
        shutil.copytree(prompts_folder, dst / prompts_folder.name)

        print(f"✅ 已將 Final_Output、SD3_prompt.txt 與 prompts 資料夾 複製到 {dst}")

        # 最後執行 cleanup.py
        subprocess.run(['python', 'cleanup.py'], check=True)
    

if __name__ == "__main__":
    main()

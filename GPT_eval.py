# ...前略...
import os
import time
import base64
import re
from mimetypes import guess_type
from typing import List
from openai import AzureOpenAI

# 進度列印頻率（每幾筆印一次），設 1 代表每筆都印
PROGRESS_EVERY = 1

# ======= 模型與 API 參數 =======
DEPLOYMENT_NAME = "gpt-4o"
API_VERSION     = "2024-12-01-preview"

AZURE_ENDPOINT  = "Your end point"
AZURE_API_KEY   = "Your API Key"

# ======= 檔案路徑 =======
PROMPT_TEMPLATE_PATH = "/media/alexkuo3090/KC3000/MYWORK/prompts/MLLM_eval_prompt.txt"
PROMPTS_LIST_PATH    = "/media/alexkuo3090/KC3000/MYWORK/Datasets/MARIOEval/MARIOEval/LAIONEval4000/LAIONEval4000.txt"

# IMAGES_DIR  = "/media/alexkuo3090/KC3000/MYWORK/Datasets/Generate/Textdiffuser"
IMAGES_DIR  = "/media/alexkuo3090/KC3000/MYWORK/Datasets/Generate/Textdiffuser2_Edit"
OUTPUT_DIR  = "/media/alexkuo3090/KC3000/MYWORK/Datasets/Generate/Textdiffuser2_eval"

SYSTEM_PROMPT_PATH = os.path.join(os.path.dirname(PROMPT_TEMPLATE_PATH), "MLLM_system_prompt.txt")
EVAL_INPUT_PATH    = os.path.join(os.path.dirname(PROMPT_TEMPLATE_PATH), "MLLM_eval_input.txt")

MAX_RETRIES = 3
RETRY_BACKOFF_SEC = 5.0

def list_images_sorted(dir_path: str) -> List[str]:
    def natural_key(s: str):
        import re
        return [int(text) if text.isdigit() else text.lower()
                for text in re.split(r'(\d+)', s)]
    exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    files = [f for f in os.listdir(dir_path)
             if os.path.splitext(f)[1].lower() in exts]
    files.sort(key=natural_key)
    return [os.path.join(dir_path, f) for f in files]

def load_lines(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]

def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def local_image_to_data_url(path: str) -> str:
    mime, _ = guess_type(path)
    mime = mime or "application/octet-stream"
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode()
    return f"data:{mime};base64,{data}"

def build_client() -> AzureOpenAI:
    return AzureOpenAI(
        api_version=API_VERSION,
        azure_endpoint=AZURE_ENDPOINT,
        api_key=AZURE_API_KEY
    )

def run_one(client: AzureOpenAI, filled_text: str, image_path: str) -> str:
    img_data_url = local_image_to_data_url(image_path)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = client.chat.completions.create(
                model=DEPLOYMENT_NAME,
                max_tokens=4096,
                temperature=1.0,
                top_p=1.0,
                messages=[
                    {"role": "system",
                     "content": ("You are a professional digital artist. "
                                 "You will evaluate the effectiveness of the AI-generated image(s) based on the given rules.")},
                    {"role": "user",
                     "content": [
                         {"type": "text", "text": filled_text},
                         {"type": "image_url", "image_url": {"url": img_data_url}}
                     ]}
                ],
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            print(f"[WARN] 呼叫失敗（第 {attempt}/{MAX_RETRIES} 次）：{e}", flush=True)
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF_SEC * attempt)
            else:
                raise

def main():
    print("=== 批次評分開始 ===", flush=True)
    client = build_client()

    template_text = read_text(PROMPT_TEMPLATE_PATH)
    prompt_lines  = load_lines(PROMPTS_LIST_PATH)
    image_paths   = list_images_sorted(IMAGES_DIR)

    n = min(len(prompt_lines), len(image_paths))
    ensure_dir(OUTPUT_DIR)

    print(f"[INFO] 將處理 {n} 筆", flush=True)
    for i in range(n):
        out_path = os.path.join(OUTPUT_DIR, f"{i}.txt")
        if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
            if i % PROGRESS_EVERY == 0:
                print(f"[SKIP] {i+1}/{n} 已存在 → {out_path}", flush=True)
            continue

        prompt_i = prompt_lines[i]

        # Step 1
        with open(SYSTEM_PROMPT_PATH, "w", encoding="utf-8") as f:
            f.write(prompt_i)

        # Step 2
        filled_text = re.sub(r"<\s*prompt\s*>", prompt_i, template_text, flags=re.IGNORECASE)
        with open(EVAL_INPUT_PATH, "w", encoding="utf-8") as f:
            f.write(filled_text)

        # Step 3
        img_i = image_paths[i]
        try:
            result = run_one(client, filled_text, img_i)
        except Exception as e:
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(f"__ERROR__ {str(e)}\n")
            print(f"[ERR ] {i+1}/{n} 失敗，已寫入錯誤：{out_path}", flush=True)
            continue

        with open(out_path, "w", encoding="utf-8") as f:
            f.write(result)

        if (i % PROGRESS_EVERY == 0) or (i == n - 1):
            print(f"[OK] 進度 {i+1}/{n} → {out_path}", flush=True)

    print("=== 全部完成 ===", flush=True)

if __name__ == "__main__":
    main()

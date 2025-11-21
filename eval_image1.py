#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
整合版流程：
1. 從 PROMPTS_FILE 逐條讀取 prompt。
2. 使用 GPT-4o 根據 TEMPLATE_FILE 強化 prompt。
3. 用 GPT-Image-1 生成影像 → resize 1024×1024 → 輸出 out1 / out2。
4. 執行 Agent_gpt.py。
"""
import os, re, requests, base64, shutil, subprocess, shlex, signal
from io import BytesIO
from PIL import Image
from openai import AzureOpenAI
from requests.exceptions import HTTPError, ReadTimeout, ConnectionError

# ===== API KEY & 端點設定 =====
GPT_API_KEY       = "Input your API KEY"
GPT_ENDPOINT      = "Input your ENDPOINT"
GPT_API_VERSION   = "Input your API Version"
GPT_DEPLOYMENT    = "gpt-4o"

IMAGE_API_KEY     = "Input your API KEY"
IMAGE_BASE        = "Input your ENDPOINT"
IMAGE_API_VERSION = "Input your API Version"
IMAGE_DEPLOYMENT  = "gpt-image-1"

# ===== 路徑設定 =====
PROMPTS_FILE       = "/home/pmd/Desktop/Alex/Datasets/MARIOEval/MARIOEval/LAIONEval4000/LAIONEval4000_GPT_USE.txt"
TEMPLATE_FILE      = "/home/pmd/Desktop/Alex/Datasets/Generate/Image-1/Enhance_system_prompt.txt"
LOG_FILE           = "/home/pmd/Desktop/Alex/Datasets/Generate/Image-1/bad_request_prompts.log"
SINGLE_PROMPT_PATH = "/home/pmd/Desktop/Alex/prompts/SD3_prompt.txt"
OUT1_DIR           = "/home/pmd/Desktop/Alex/images/SD3"
OUT2_DIR           = "/home/pmd/Desktop/Alex/images/Final_Output/SD3_1_1/SD"
AGENT_SCRIPT       = "Agent_gpt.py"

# 建立目錄
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
os.makedirs(OUT1_DIR, exist_ok=True)
os.makedirs(OUT2_DIR, exist_ok=True)
os.makedirs(os.path.dirname(SINGLE_PROMPT_PATH), exist_ok=True)

# ===== 函式：用 GPT-4o 強化 prompt =====
def enhance_prompt(raw_prompt: str, template_text: str) -> str:
    """根據模板與原始 prompt 呼叫 GPT-4o 生成增強版 prompt"""
    client = AzureOpenAI(api_version=GPT_API_VERSION,
                         azure_endpoint=GPT_ENDPOINT,
                         api_key=GPT_API_KEY)
    try:
        # 取出引號內的可渲染文字
        items = re.findall(r"[\"'](.*?)[\"']", raw_prompt)
        render_list = ", ".join(items) if items else "you can't render any text in the image"

        system_prompt = template_text.replace(
            '("Text")', f'({raw_prompt})'
        ).replace(
            "3. Focus on the text, The text only can be render is:",
            f"3. Focus on the text, The text only can be render is: {render_list}"
        )

        resp = client.chat.completions.create(
            model=GPT_DEPLOYMENT,
            temperature=1.0,
            max_tokens=500,
            messages=[{"role": "system", "content": system_prompt}]
        )
        return resp.choices[0].message.content.strip()

    except Exception as e:
        print(f"❌ GPT-4o 增強失敗: {e}")
        return raw_prompt  # fallback 為原始 prompt

# ===== 函式：呼叫 GPT-Image-1 生成圖片 =====
def generate_and_save_image(prompt: str) -> bool:
    """使用 GPT-Image-1 生成圖片並壓縮為 1024×1024"""
    out1 = os.path.join(OUT1_DIR, "SD3_1_1.png")
    out2 = os.path.join(OUT2_DIR, "SD3_1_1.png")

    url = (
        f"{IMAGE_BASE}/openai/deployments/{IMAGE_DEPLOYMENT}/images/generations"
        f"?api-version={IMAGE_API_VERSION}"
    )
    headers = {"Content-Type": "application/json", "api-key": IMAGE_API_KEY}
    payload = {"prompt": prompt, "n": 1, "size":"1024x1024", "quality":"medium"}

    try:
        r = requests.post(url, headers=headers, json=payload, timeout=(10, 180))
        r.raise_for_status()
        b64 = r.json()["data"][0]["b64_json"]
        img_data = base64.b64decode(b64)
    except Exception as e:
        print(f"❌ 生成圖片失敗: {e}")
        return False

    try:
        img = Image.open(BytesIO(img_data)).convert("RGB")
        img_resized = img.resize((1024, 1024), Image.LANCZOS)
        os.makedirs(os.path.dirname(out1), exist_ok=True)
        os.makedirs(os.path.dirname(out2), exist_ok=True)
        img_resized.save(out1, format="PNG", optimize=True)
        shutil.copy(out1, out2)
        print(f"✅ 已生成並壓縮輸出：{out1} 和 {out2}")
        return True
    except Exception as e:
        print(f"❌ 圖片寫入失敗: {e}")
        return False

# ===== 主流程 =====
def main():
    def _timeout_handler(signum, frame):
        raise TimeoutError("整包處理逾時，跳至下一張")
    signal.signal(signal.SIGALRM, _timeout_handler)

    # 讀取模板與 prompts
    try:
        with open(TEMPLATE_FILE, "r", encoding="utf-8") as f:
            template = f.read()
        with open(PROMPTS_FILE, "r", encoding="utf-8") as f:
            prompts = [line.strip() for line in f if line.strip()]
    except FileNotFoundError as e:
        print(f"❌ 找不到檔案: {e}")
        return

    for idx, raw_prompt in enumerate(prompts, start=1):
        signal.alarm(360)
        try:
            print(f"\n--- 開始處理 Prompt #{idx} ---")
            print(f"📝 原始 Prompt: {raw_prompt}")

            # 1️⃣ 寫入單一 prompt 給 Agent
            with open(SINGLE_PROMPT_PATH, "w", encoding="utf-8") as spf:
                spf.write(raw_prompt)

            # 2️⃣ GPT-4o 增強 prompt
            enhanced_prompt = raw_prompt #= enhance_prompt(raw_prompt, template)
            print(f"📈 增強後 Prompt:\n{enhanced_prompt}\n")

            # 3️⃣ GPT-Image-1 生成圖片
            if not generate_and_save_image(enhanced_prompt):
                continue

            # 4️⃣ 執行 Agent
            print(f"▶️ 執行 Agent_gpt.py for #{idx}")
            cmd = f"python {shlex.quote(AGENT_SCRIPT)} {idx}"
            subprocess.run(cmd, shell=True, check=True)

        except TimeoutError as e:
            print(f"⏲️ Prompt #{idx} {e}")
            continue
        except subprocess.CalledProcessError as e:
            print(f"❌ Agent_gpt.py 處理 Prompt #{idx} 失敗 (exit {e.returncode})，跳過")
            continue
        except Exception as e:
            print(f"❌ 處理 Prompt #{idx} 未預期錯誤: {e}")
            continue
        finally:
            signal.alarm(0)

    print("\n✅ 全部 prompts 處理完畢。")

if __name__ == "__main__":
    main()


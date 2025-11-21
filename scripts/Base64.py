import requests
import base64
import os
import time
import argparse
import sys


def file_to_base64(path):
    if not path or not os.path.isfile(path):
        raise FileNotFoundError(f"檔案不存在: {path}")
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def main():
    parser = argparse.ArgumentParser(
        description="使用 Flux Pro API 進行影像填充處理，接收輸入參數並支援預設目錄。"
    )
    parser.add_argument(
        "--image_path", "-i",
        default="/media/alexkuo3090/KC3000/MYWORK/images/SD3/SD3_1_1.png",
        help="要處理的原始影像路徑，預設為當前目錄下的 input.png"
    )
    parser.add_argument(
        "--mask_path", "-m",
        default="/media/alexkuo3090/KC3000/MYWORK/images/SD3/PRSPCT_Family_Album.png",
        help="遮罩影像路徑，若未指定將使用與 image 相同的影像作為遮罩"
    )
    parser.add_argument(
        "--output_file", "-o",
        default="/media/alexkuo3090/KC3000/MYWORK/images/SD3/output.png",
        help="下載後的輸出檔案名稱，預設為 output.png"
    )
    parser.add_argument(
        "--prompt", "-p",
        default="Generate the text 'PRSPCT Family Album'",
        help="API 生成文字的提示詞"
    )
    parser.add_argument(
        "--api-key", "-k",
        default="45a39127-b742-4b7d-b920-e776ae4efe8b",
        help="Flux Pro API 金鑰"
    )
    args = parser.parse_args()

    # 轉 Base64
    image_b64 = file_to_base64(args.image_path)
    mask_b64 = file_to_base64(args.mask_path) if args.mask_path else None

    # 初始化請求
    api_url = "your api key"
    payload = {
        "prompt": args.prompt,
        "steps": 50,
        "prompt_upsampling": True,
        "guidance": 60,
        "output_format": "png",
        "safety_tolerance": 2,
        "image": image_b64,
        "mask": mask_b64 or image_b64
    }
    headers = {
        "x-key": args.api_key,
        "Content-Type": "application/json"
    }

    # 首次請求
    response = requests.post(api_url, json=payload, headers=headers)
    response.raise_for_status()
    data = response.json()
    print("初次請求結果：", data)

    polling_url = data.get("polling_url")
    if not polling_url:
        raise ValueError("未取得 polling_url")

    # 開始計時
    start_time = time.time()
    timeout_seconds = 120

    # 輪詢直到完成或超時
    while True:
        # 檢查是否超時
        elapsed = time.time() - start_time
        if elapsed > timeout_seconds:
            print(f"⏰ 已超過 {timeout_seconds} 秒，停止輪詢。")
            sys.exit(1)

        resp = requests.get(polling_url)
        resp.raise_for_status()
        result = resp.json()
        status = result.get("status")
        print("狀態：", status)

        if status.lower() in ("succeeded", "completed", "ready"):
            sample_url = result.get("result", {}).get("sample")
            if not sample_url:
                raise ValueError("未在回傳結果中找到 sample URL")
            # 等待一段時間再下載，確保檔案可用
            time.sleep(5)
            img_resp = requests.get(sample_url, stream=True)
            img_resp.raise_for_status()
            out_path = args.output_file
            with open(out_path, "wb") as f:
                for chunk in img_resp.iter_content(8192):
                    f.write(chunk)
            print(f"✅ 圖片已下載並儲存為 {out_path}")
            break

        elif status.lower() in ("failed", "error"):
            print("❌ 處理失敗：", result)
            sys.exit(1)
        else:
            time.sleep(1)


if __name__ == "__main__":
    main()

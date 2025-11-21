import argparse
import warnings
warnings.filterwarnings('ignore', message='invalid value encountered in cast')
import torch
from diffusers import FluxFillPipeline, FluxTransformer2DModel
from diffusers.utils import load_image
from pathlib import Path

ations = 'Run FLUX text fill inference with full GPU (BF16) support and original logic.'

def clear_vram():
    """
    清空 GPU 記憶體緩存，避免 OOM。
    """
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

def main(image_path: str, mask_path: str, prompt: str, output_path: str):
    # 每次推論前清空 VRAM
    clear_vram()

    print('DEBUG - image_path:', image_path)
    print('DEBUG - mask_path: ', mask_path)
    print('DEBUG - prompt:    ', prompt)
    print('DEBUG - output_path:', output_path)

    image = load_image(image_path)
    mask  = load_image(mask_path)

    model_id   = 'black-forest-labs/FLUX.1-Fill-dev'


    # 使用 BF16，不採用 FP16/8bit/4bit 量化, 並將 device_map 設為 balanced
    model_fp16 = FluxTransformer2DModel.from_pretrained(
        model_id,
        subfolder='transformer',
        device_map='balanced',
        torch_dtype=torch.bfloat16
    )

    pipe = FluxFillPipeline.from_pretrained(
        model_id,
        transformer=model_fp16,
        device_map='balanced',
        torch_dtype=torch.bfloat16
    )

    # 推論
    output_image = pipe(
        prompt=prompt,
        image=image,
        mask_image=mask,
        height=1024,
        width=1024,
        guidance_scale=50,
        num_inference_steps=50,
        max_sequence_length=512,
        generator=torch.Generator(device='cuda').manual_seed(0)
    ).images[0]

    # 處理並儲存結果
    out_path = Path(output_path)
    if out_path.is_dir():
        name     = Path(image_path).stem + '_flux.png'
        out_file = out_path / name
    else:
        out_file = out_path if out_path.suffix else out_path.with_suffix('.png')
    out_file.parent.mkdir(parents=True, exist_ok=True)
    output_image.save(out_file)
    print('[INFO] Saved output to', out_file)

    # 推論後可選擇再次清空 VRAM
    clear_vram()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=ations)
    parser.add_argument(
        '--image_path', type=str,
        default='/home/pmd/Desktop/Alex/images/SD3/1/SD3_1_1.png',
        help='Path to the input image.'
    )
    parser.add_argument(
        '--mask_path', type=str,
        default='/home/pmd/Desktop/Alex/images/SD3/1/mask/SD3_1_1_mask.png',
        help='Path to the mask image.'
    )
    parser.add_argument(
        '--prompt', type=str,
        default='render the text "Congradulations" in mask',
        help='Text prompt for fill.'
    )
    parser.add_argument(
        '--output_path', type=str,
        default='./output.png',
        help='Output image filename or directory.'
    )
    args = parser.parse_args()
    print('DEBUG - parsed args:', args)
    main(args.image_path, args.mask_path, args.prompt, args.output_path)



# Text-Pilot: Intelligent Visual Text Planning and Manipulation via Multi-modal LLM as Agent

## 📘 Overview
Text-Pilot is a training-free, MLLM-based agent framework that automatically detects and corrects text errors in generated images.
It leverages the reasoning and perception abilities of multi-modal large language models (MLLMs) to evaluate visual text accuracy and autonomously decide which operation—edit, erase, or regenerate—should be applied.

Unlike text-focused diffusion models that often sacrifice visual fidelity, Text-Pilot acts as a post-processing layer, seamlessly integrating with any T2I model (e.g., Stable Diffusion 3.5, GPT-Image-1). It enhances OCR consistency while preserving image quality.

Key features

Agent-driven self-diagnosis and correction of visual text
Unified framework for editing, erasing, and rendering text
Plug-and-play with existing diffusion models, no training required
Significant improvement on OCR and text-fidelity metrics
→ Empower your diffusion model to “read and write” correctly.
<img width="2862" height="1592" alt="image" src="https://github.com/user-attachments/assets/d9b81874-54c6-494d-b946-28dcbb49b249" />



## 🔧 Environment Setup
0. git clone Text-Pilot

1. Manual setup all the tools that Text-pilot have use, and there have instructions in each tool folder, please reference it:

    a. OCR:<br>
           i. We did some revised on this file -> /Tools/OCR/Units_Detector/units/scripts/Units_inference.py, make sure this file is copy<br>
           ii. Download code from https://github.com/clovaai/units into path /Tools/OCR/Units_Detector/units/<br>
           iii. Make sure step 1 Units_inference.py replace original Units_inference.py<br>
           iv. conda create --name units python=3.10<br>
           v. Cotinue remain steps from https://github.com/clovaai/units into path /Tools/OCR/Units_Detector/units
   
    b. SD3:<br>
           i. Download code from https://huggingface.co/stabilityai/stable-diffusion-3.5-larges into path /Tools/SD3/stable-diffusion-3.5-large<br>
           ii. conda create --name sd3 python=3.11
           iii. Cotinue remain steps from https://huggingface.co/stabilityai/stable-diffusion-3.5-larges
    
    c. FLUX:<br>
           i. Download code from https://huggingface.co/black-forest-labs/FLUX.1-Fill-dev into path /Tools/STE/FLUX<br>
           ii. conda create --name Flux python=3.10
           iii. Cotinue remain steps from https://huggingface.co/black-forest-labs/FLUX.1-Fill-dev
   
    d. Textctrl:<br>
           i. Download code from https://github.com/weichaozeng/TextCtrl into path /Tools/STE/TextCtrl<br>
           ii. conda create --name textctrl python=3.11
           iii. Cotinue remain steps from https://github.com/weichaozeng/TextCtrl
   
    e. Lama:<br>
           i. We did some revised on this file -> /Tools/STR/Lama/bin/predict.py, make sure this file is copy<br>
           ii. Download code from https://github.com/advimman/lama into path /Tools/STR/Lama<br>
           iii. Make sure step 1 predict.py replace original predict.py<br>
           iv. conda create --name lama python=3.6<br>
           v. Cotinue remain steps from https://github.com/advimman/lama
   
    f. DiffTSR:<br>
           i. Download code from https://github.com/YuzheZhang-1999/DiffTSR into path /Tools/Super-resolution/DiffTSR<br>
           ii. conda create --name DiffTSR python=3.8<br>
           iii. Cotinue remain steps from https://github.com/YuzheZhang-1999/DiffTSR
   
    e. Base Python:<br>
           i. conda create --name base311 python=3.11<br>

3. Prepared GPT-4o and GPT-Image-1 API key and end-point filled in:
   
   a. GPT-4o -> eval.py, scripts/OpenAI_layout.py and scripts/OpenAI.py<br>
   b. GPT-Image-1 -> eval_image1.py

5. Prepared the datasets Mario-eval from https://github.com/microsoft/unilm/tree/master/textdiffuser into path /Datasets
   
    a. create folder iter_exp into path /Datasets/MARIOEval/MARIOEval/LAIONEval4000

6. ```
   conda activate base311

7. ```
   python eval.py
the pipeline uses Stable Diffusion 3.5 as its T2I model.

8. ```
   python eval_image1.py
the pipeline uses GPT-image-1 as its T2I model.

## 🔧 Eval

1. Prepare:
   
    a. python ./File_generate.py
   
    b. move all the image out of folder and copy the original and final image to /Datasets/Generate
        >> Use the microsoft image ai to ocr the image, and save in Datasets/Eval/OCR_results

2. OCR:
   
    a. python ./OCR_score.py
        >> Start evalution
   
3. FID
   
    a. python FID_score.py ../Datasets/MARIOEval/MARIOEval/LAIONEval4000/images ../Datasets/Generate/Edit_FID

4. Clip_score
   
    a. python Clip_scroe.py
   
    b. python Clip_file_make.py
        python Clip_score.py image2prompt.json ./Datasets/Generate/SD3_FID
        python Clip_score.py image2prompt_edit.json ./Datasets/Generate/Edit_FID

5. GPT-4o (prepare API Key) and LLava-LLama-3 (download from https://huggingface.co/xtuner/llava-llama-3-8b)

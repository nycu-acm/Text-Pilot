# Text-Pilot
Text-Pilot: Intelligent Visual Text Planning and Manipulation via Multi-modal LLM as Agent


📘 Overview
Text-Pilot is a training-free, MLLM-based agent framework that automatically detects and corrects text errors in generated images.
It leverages the reasoning and perception abilities of multi-modal large language models (MLLMs) to evaluate visual text accuracy and autonomously decide which operation—edit, erase, or regenerate—should be applied.

Unlike text-focused diffusion models that often sacrifice visual fidelity, Text-Pilot acts as a post-processing layer, seamlessly integrating with any T2I model (e.g., Stable Diffusion 3.5, GPT-Image-1). It enhances OCR consistency while preserving image quality.

Key features

Agent-driven self-diagnosis and correction of visual text
Unified framework for editing, erasing, and rendering text
Plug-and-play with existing diffusion models, no training required
Significant improvement on OCR and text-fidelity metrics
→ Empower your diffusion model to “read and write” correctly.


🔧 Environment Setup

1. Manual setup all the tools that Text-pilot is using:

    a. OCR: Download code from https://github.com/clovaai/units to path /Tools/OCR/Units_Detector/units/
    
    b. SD3: Download code from https://huggingface.co/stabilityai/stable-diffusion-3.5-larges to path /Tools/SD3/stable-diffusion-3.5-large  
    
    c. FLUX: Download code from https://huggingface.co/black-forest-labs/FLUX.1-Fill-dev to path /Tools/STE/FLUX 

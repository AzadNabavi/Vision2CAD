"""
cadcoder_runner.py
==================
This script runs INSIDE the cadcoder_env Python 3.10 virtual environment.
It is called as a subprocess from cad_coder_integration.py.

It loads the CAD-Coder model, runs inference on the given image,
and prints the generated CadQuery code to stdout.

Usage:
  cadcoder_env\Scripts\python.exe cadcoder_runner.py <image_path>
"""

import sys
import os
import torch

# Suppress symlink warnings on Windows
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
sys.stdout.reconfigure(encoding="utf-8")

HF_MODEL_ID = "CADCODER/CAD-Coder"


def run(image_path):
    from llava.model.builder import load_pretrained_model
    from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
    from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
    from llava.conversation import conv_templates
    from PIL import Image

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[CAD-Coder] Device: {device}", file=sys.stderr)
    print(f"[CAD-Coder] Loading model from {HF_MODEL_ID} ...", file=sys.stderr)

    model_name = get_model_name_from_path(HF_MODEL_ID)
    dtype = torch.float16

    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=HF_MODEL_ID,
        model_base=None,
        model_name=model_name,
        device_map=device,
        load_in_8bit=True,
    )
    model.eval()

    # If image_processor is None, load it directly from the vision tower
    if image_processor is None:
        print("[CAD-Coder] Loading image processor from vision tower...", file=sys.stderr)
        vision_tower = model.get_vision_tower()
        if not vision_tower.is_loaded:
            vision_tower.load_model(device_map=device)
        vision_tower.to(dtype=dtype)
        image_processor = vision_tower.image_processor

    # Ensure the mm_projector is the correct dtype
    model.get_model().mm_projector.to(dtype=dtype)

    print("[CAD-Coder] Model loaded. Running inference...", file=sys.stderr)

    image = Image.open(image_path).convert("RGB")
    image_tensor = process_images([image], image_processor, model.config)
    image_tensor = [t.to(dtype=dtype, device=device) for t in image_tensor]

    # LLaVA 1.5 conversation template
    conv = conv_templates["llava_v1"].copy()
    user_msg = (
        DEFAULT_IMAGE_TOKEN + "\n"
        "Generate complete CadQuery Python code to create the 3D part shown in this CAD drawing. "
        "Include all outer cylinders, steps, fillets, internal holes, and threads. "
        "The code must import cadquery and produce a valid result object."
    )
    conv.append_message(conv.roles[0], user_msg)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(
        prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0).to(device)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            image_sizes=[image.size],
            do_sample=False,
            max_new_tokens=1024,
        )

    # Only print the generated tokens (not the input prompt)
    generated = output_ids[0][input_ids.shape[1]:]
    code = tokenizer.decode(generated, skip_special_tokens=True).strip()

    # Print to stdout so the parent process can capture it
    print(code)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: cadcoder_runner.py <image_path>", file=sys.stderr)
        sys.exit(1)
    run(sys.argv[1])

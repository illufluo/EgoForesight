"""
Local Qwen3.5-9B VLM Client

Same interface as glm_client.py: call_vlm(images, prompt) -> str
Model is lazy-loaded on first call. Supports optional LoRA adapter for V4/V5.

Configuration:
    Environment variables (or call init_model() explicitly):
    - QWEN_MODEL_PATH:   path to base model (default: /root/sdp/models/qwen3.5-9b)
    - QWEN_ADAPTER_PATH: path to LoRA adapter (optional, empty = base model)
"""

import os
from typing import List, Optional

os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# Module-level lazy state
_model = None
_tokenizer = None
_initialized = False


def init_model(
    model_path: str = None,
    adapter_path: str = None,
    max_seq_length: int = 2048,
) -> None:
    """
    Explicitly initialize the model. Call this before call_vlm() if you need
    to specify a custom model path or adapter (e.g. for V4/V5 fine-tuned models).

    If not called, call_vlm() will auto-initialize from environment variables.
    """
    global _model, _tokenizer, _initialized

    if _initialized:
        return

    if model_path is None:
        model_path = os.getenv("QWEN_MODEL_PATH", "/root/sdp/models/qwen3.5-9b")
    if adapter_path is None:
        adapter_path = os.getenv("QWEN_ADAPTER_PATH", "")

    from unsloth import FastModel

    print(f"Loading Qwen model from {model_path}...")
    model, tokenizer = FastModel.from_pretrained(
        model_name=model_path,
        max_seq_length=max_seq_length,
        load_in_4bit=False,
        load_in_16bit=True,
        local_files_only=True,
    )

    if adapter_path:
        from peft import PeftModel

        print(f"Loading LoRA adapter from {adapter_path}...")
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()
        print("Adapter merged.")

    FastModel.for_inference(model)

    _model = model
    _tokenizer = tokenizer
    _initialized = True
    print("Model ready.")


def call_vlm(images: List[str], prompt: str) -> str:
    """
    Run inference on local Qwen3.5-9B with one or more images and a text prompt.

    Args:
        images: List of local image file paths
        prompt: Prompt text

    Returns:
        Final text response from the model (thinking tokens stripped)

    Raises:
        FileNotFoundError: If any image file does not exist
        RuntimeError: If model fails to generate
    """
    global _model, _tokenizer, _initialized

    if not _initialized:
        init_model()

    if not images:
        raise ValueError("No images provided.")

    for img in images:
        if not os.path.exists(img):
            raise FileNotFoundError(f"Image file not found: {img}")

    from PIL import Image

    # Load images
    pil_images = [Image.open(p).convert("RGB") for p in images]

    # Build message in Qwen chat format
    user_content = [{"type": "image", "image": img} for img in pil_images]
    user_content.append({"type": "text", "text": prompt})

    messages = [{"role": "user", "content": user_content}]

    # Apply chat template
    input_text = _tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
        enable_thinking=False,
    )

    # Tokenize with images
    inputs = _tokenizer(
        pil_images,
        input_text,
        add_special_tokens=False,
        return_tensors="pt",
    ).to("cuda")

    # Generate
    outputs = _model.generate(
        **inputs,
        max_new_tokens=256,
        temperature=0.3,
        do_sample=False,
        repetition_penalty=1.15,
    )

    # Decode — skip input tokens
    decoded = _tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )

    # Strip thinking tokens if present (Qwen3.5 may output <think>...</think>)
    if "</think>" in decoded:
        decoded = decoded.split("</think>")[-1].strip()

    if not decoded:
        raise RuntimeError("Model returned empty response.")

    return decoded

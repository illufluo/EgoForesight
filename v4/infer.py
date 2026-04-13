"""
V4 Quick Inference Test: load fine-tuned model, run on a few test samples.

Usage (on GPU server):
    cd /root/sdp
    python -m v4.infer
"""

import os
import json
from PIL import Image
from unsloth import FastModel
from peft import PeftModel

os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

MODEL_PATH   = "/root/sdp/models/qwen3.5-9b"
ADAPTER_PATH = "/root/sdp/outputs/v4/lora_adapter"
TEST_PATH    = "/root/sdp/data/training_v4/test.json"
IMAGE_BASE   = "/root/sdp"

model, tokenizer = FastModel.from_pretrained(
    model_name       = MODEL_PATH,
    max_seq_length   = 2048,
    load_in_4bit     = False,
    load_in_16bit    = True,
    local_files_only = True,
)

model = PeftModel.from_pretrained(model, ADAPTER_PATH)
model = model.merge_and_unload()
print("Adapter merged successfully")

FastModel.for_inference(model)

with open(TEST_PATH) as f:
    test_data = json.load(f)

for i, item in enumerate(test_data[:5]):
    print(f"\n{'='*60}")
    print(f"Sample {i+1}")

    images = [Image.open(os.path.join(IMAGE_BASE, p)).convert("RGB")
              for p in item["images"]]

    user_text = item["messages"][0]["content"].replace("<image>", "").strip()

    user_content = [{"type": "image", "image": img} for img in images]
    user_content.append({"type": "text", "text": user_text})

    messages = [{"role": "user", "content": user_content}]

    input_text = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )

    inputs = tokenizer(
        images,
        input_text,
        add_special_tokens=False,
        return_tensors="pt",
    ).to("cuda")

    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=1.0,
        do_sample=False,
    )

    decoded = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )

    if "</think>" in decoded:
        decoded = decoded.split("</think>")[-1].strip()

    print(f"[Ground Truth]\n{item['messages'][1]['content']}")
    print(f"\n[Model Output]\n{decoded}")

print("\nDone.")

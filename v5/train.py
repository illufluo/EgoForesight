"""
V5 Training: fine-tune Qwen3.5-9B on action prediction WITH history context.

Usage (on GPU server):
    cd /root/sdp
    python -m v5.train
"""

import os
import json
from PIL import Image
from unsloth import FastModel
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig

# ─── Config ───────────────────────────────────
MODEL_PATH  = "/root/sdp/models/qwen3.5-9b"
TRAIN_PATH  = "/root/sdp/data/training_v5/train.json"
VAL_PATH    = "/root/sdp/data/training_v5/val.json"
OUTPUT_DIR  = "/root/sdp/outputs/v5"
IMAGE_BASE  = "/root/sdp"
MAX_SEQ_LEN = 2048


# ─── Lazy Dataset ─────────────────────────────
class LazyVisionDataset:
    def __init__(self, path):
        with open(path) as f:
            self.raw = json.load(f)
        print(f"Loaded {len(self.raw)} samples from {path}")

    def __len__(self):
        return len(self.raw)

    def __getitem__(self, idx):
        item = self.raw[idx]
        images = [
            Image.open(os.path.join(IMAGE_BASE, p)).convert("RGB")
            for p in item["images"]
        ]
        user_text = item["messages"][0]["content"].replace("<image>", "").strip()
        user_content = [{"type": "image", "image": img} for img in images]
        user_content.append({"type": "text", "text": user_text})
        assistant_text = item["messages"][1]["content"]
        return {
            "messages": [
                {"role": "user",      "content": user_content},
                {"role": "assistant", "content": [{"type": "text", "text": assistant_text}]},
            ]
        }


# ─── Load model ───────────────────────────────
model, tokenizer = FastModel.from_pretrained(
    model_name     = MODEL_PATH,
    max_seq_length = MAX_SEQ_LEN,
    load_in_4bit   = False,
    load_in_16bit  = True,
)

model = FastModel.get_peft_model(
    model,
    finetune_vision_layers     = True,
    finetune_language_layers   = True,
    finetune_attention_modules = True,
    finetune_mlp_modules       = True,
    r              = 16,
    lora_alpha     = 16,
    lora_dropout   = 0,
    bias           = "none",
    random_state   = 3407,
    use_rslora     = False,
    target_modules = "all-linear",
)

# ─── Train ────────────────────────────────────
os.makedirs(OUTPUT_DIR, exist_ok=True)

train_dataset = LazyVisionDataset(TRAIN_PATH)
val_dataset   = LazyVisionDataset(VAL_PATH)

trainer = SFTTrainer(
    model         = model,
    tokenizer     = tokenizer,
    data_collator = UnslothVisionDataCollator(model, tokenizer),
    train_dataset = train_dataset,
    eval_dataset  = val_dataset,
    args = SFTConfig(
        output_dir                  = OUTPUT_DIR,
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        num_train_epochs            = 3,
        learning_rate               = 2e-4,
        warmup_ratio                = 0.05,
        lr_scheduler_type           = "cosine",
        bf16                        = True,
        fp16                        = False,
        logging_steps               = 10,
        save_steps                  = 200,
        eval_steps                  = 200,
        eval_strategy               = "steps",
        save_total_limit            = 3,
        max_seq_length              = MAX_SEQ_LEN,
        dataset_text_field          = "",
        dataset_kwargs              = {"skip_prepare_dataset": True},
        dataset_num_proc            = 1,
        remove_unused_columns       = False,
        seed                        = 3407,
    ),
)

trainer.train()

model.save_pretrained(os.path.join(OUTPUT_DIR, "lora_adapter"))
tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "lora_adapter"))
print("Training complete. Adapter saved to", OUTPUT_DIR)

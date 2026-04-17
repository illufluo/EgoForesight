# EgoForesight

VLM-based egocentric action prediction system using Ego4D FHO dataset. Analyzes video frames and predicts the next action in natural language. Built incrementally as V1 → V2 → V3 → V4 → V5, with an annotation pipeline for training data generation and an evaluation toolkit for quantitative comparison.

## Tech Stack

- **VLM (API)**: GLM-4.6V via zai-sdk (`ZhipuAiClient`), thinking enabled
- **VLM (Local)**: Qwen3.5-9B via unsloth `FastModel`, optional LoRA adapter, inference: temperature=0.3, repetition_penalty=1.15
- **Backend selector**: `shared/vlm.py` — `get_call_vlm("glm"|"qwen")` for V1/V2/V3
- **Fine-tuning**: unsloth + peft + trl (SFTTrainer), LoRA r=16, all-linear, bf16
- **Evaluation**: BERTScore + Semantic Similarity (sentence-transformers)
- **Frame extraction**: OpenCV (cv2)

## Dataset

- **Source**: Ego4D FHO (Forecasting Hands and Objects)
- **Narration CSV columns**: video_uid, pass, timestamp_sec, timestamp_frame, narration_text, annotation_uid

## Versions

| Version | Input | History | Model | Status |
|---------|-------|---------|-------|--------|
| V1 | Single frame | No | GLM / Qwen (base) | Done |
| V2 | 4 frames | No | GLM / Qwen (base) | Done |
| V3 | 4 frames | Sliding window (3 steps) | GLM / Qwen (base) | Done |
| V4 | 4 frames | No | Qwen + LoRA (run1 & run2) | Done |
| V5 | 4 frames | Yes (3 steps) | Qwen + LoRA (run1 & run2) | Done |

- **Explanation**: natural language description of current action (30-50 words)
- **Prediction**: natural language description of most likely next action (30-50 words)

## Training Data

| Dataset | Samples | Split | Notes |
|---------|---------|-------|-------|
| training_v4 | 4918 | 50/10/10 videos | No history context |
| training_v5 | 4918 | Same split | With history (3 steps) in prompt |
| training_v4_filtered | Subset | Same val/test | Jaccard > 0.5 samples removed |
| training_v5_filtered | Subset | Same val/test | Jaccard > 0.5 samples removed |

- 70/71 videos annotated via GLM-4.6V (single-pass, 30-50 word explanations)
- Splits by video (not window) to avoid data leakage, seed 42
- V4/V5 share same annotations; difference is prompt assembly in `build_training.py`

## Training Runs

| Run | Script | Data | LR | Epochs | Output |
|-----|--------|------|----|--------|--------|
| V4 run1 | `v4/train.py` | training_v4 | 2e-4 | 3 | outputs/v4 |
| V5 run1 | `v5/train.py` | training_v5 | 2e-4 | 3 | outputs/v5 |
| V4 run2 | `v4/train_v2.py` | training_v4_filtered | 5e-5 | 5 | outputs/v4_run2 |
| V5 run2 | `v5/train_v2.py` | training_v5_filtered | 5e-5 | 5 | outputs/v5_run2 |

## Directory Structure

```
sdp/
├── shared/
│   ├── glm_client.py          # GLM-4.6V API client
│   ├── qwen_client.py         # Local Qwen3.5-9B client (lazy loading, LoRA support)
│   ├── vlm.py                 # Backend selector: get_call_vlm("glm"|"qwen")
│   ├── video_frames.py        # Frame extraction (OpenCV)
│   └── utils.py               # JSON save, narration CSV load
│
├── v1/                        # Single-frame prediction (--backend glm|qwen)
│   ├── prompt.py
│   └── run.py
│
├── v2/                        # 4-frame window prediction (--backend glm|qwen)
│   ├── prompt.py
│   └── run.py
│
├── v3/                        # 4-frame window + history (--backend glm|qwen)
│   ├── history.py             # HistoryManager: deque(maxlen=3)
│   ├── prompt.py
│   └── run.py
│
├── v4/                        # Fine-tuned Qwen, no history
│   ├── prompt.py              # Anti-repetition rules for Prediction
│   ├── train.py               # Run 1: lr=2e-4, 3 epochs
│   ├── train_v2.py            # Run 2: lr=5e-5, 5 epochs, filtered data
│   ├── infer.py               # Quick test on 5 samples
│   └── run.py                 # Full inference pipeline
│
├── v5/                        # Fine-tuned Qwen, with history
│   ├── prompt.py              # Anti-repetition + anti-hallucination
│   ├── train.py               # Run 1
│   ├── train_v2.py            # Run 2
│   ├── infer.py
│   └── run.py
│
├── annotation/
│   ├── prompt.py              # Annotation prompt (30-50 words, anti-hedging)
│   ├── run_annotate.py        # Single-pass annotation, resumable (GLM-only)
│   └── build_training.py      # Annotations → LLaMA-Factory format
│
├── tools/
│   ├── filter_training_data.py  # Filter by Explanation/Prediction Jaccard overlap
│   └── evaluate.py              # BERTScore + Semantic Similarity evaluation
│
└── data/
    ├── videos/                # Raw videos (.mp4)
    ├── frames/                # Extracted frames
    ├── narrations/            # selected_narrations.csv
    ├── annotations/           # Per-video annotation JSONs
    ├── training_v4/           # V4 training data
    ├── training_v5/           # V5 training data
    ├── training_v4_filtered/  # Filtered V4 training data
    ├── training_v5_filtered/  # Filtered V5 training data
    ├── results/               # Inference output JSONs
    └── evaluation/            # Evaluation results
```

## Design Decisions

- V1/V2/V3 use `shared/vlm.py` backend selector; V4/V5 use `qwen_client` directly for explicit adapter control
- V4/V5 have dedicated `prompt.py` — fine-tuned models need anti-repetition rules (Prediction must differ from Explanation)
- V5 prompt adds anti-hallucination constraints for history context
- Qwen inference: `temperature=0.3`, `repetition_penalty=1.15`, `do_sample=False`
- Annotation uses 1s windows for flexibility; predictions constructed from neighboring explanations at training time
- Run 2 training: lower learning rate (5e-5) + more epochs (5) + filtered data to reduce overfitting
- Evaluation aligns predictions to GT narrations by temporal proximity (t_end + 1.0s, ±2s tolerance)

# EgoForesight

## Project Goal

Build a VLM-based action prediction system using egocentric video (Ego4D FHO dataset). The system analyzes video frames and predicts the next action in natural language. A separate annotation pipeline generates training data for fine-tuning.

## Tech Stack

- **VLM (API)**: GLM-4.6V via zai-sdk (`ZhipuAiClient`), thinking enabled
- **VLM (Local)**: Qwen3.5-9B via unsloth `FastModel`, optional LoRA adapter for V4/V5, inference params: temperature=0.3, repetition_penalty=1.15
- **Backend selector**: `shared/vlm.py` — `get_call_vlm("glm"|"qwen")` for V1/V2/V3
- **API key**: env var `ZHIPUAI_API_KEY` or `ZAI_API_KEY` (loaded via dotenv)
- **Frame extraction**: OpenCV (cv2)
- **Fine-tuning**: unsloth + peft + trl (SFTTrainer), LoRA r=16, all-linear, bf16
- **Python venv**: `.venv/` — always `source .venv/bin/activate` before running

## Dataset

- **Source**: Ego4D FHO (Forecasting Hands and Objects)
- **Narration CSV columns**: video_uid, pass, timestamp_sec, timestamp_frame, narration_text, annotation_uid


## Version Plan

| Version | Input | History | Fine-tune | Status |
|---------|-------|---------|-----------|--------|
| V1 | Single frame | No | No (GLM/Qwen) | ✅ Done |
| V2 | 4 frames (2s window) | No | No (GLM/Qwen) | ✅ Done |
| V3 | 4 frames | Sliding window (3 steps) | No (GLM/Qwen) | ✅ Done |
| V4 | 4 frames | No | Yes (Qwen LoRA) | ✅ Pipeline complete, needs GPU training |
| V5 | 4 frames | Yes (3 steps) | Yes (Qwen LoRA) | ✅ Pipeline complete, needs GPU training |

- **e** = explanation: natural language description of current action (30-50 words)
- **p** = prediction: natural language description of most likely next action

## Annotation & Training Data Pipeline

| Component | Status |
|-----------|--------|
| annotation/run_annotate.py | ✅ Done — single-pass, 30-50 word explanations directly |
| annotation/build_training.py | ✅ Done — supports both V4 (no history) and V5 (with history) |
| V4 training data (data/training_v4/) | ✅ 4918 samples (50 train / 10 val / 10 test videos) |
| V5 training data (data/training_v5/) | ✅ 4918 samples, same split, with history context in prompt |

### run_annotate.py
- Single-pass: directly generates 30-50 word explanations (no compression step)
- 1-second windows, 0.2s frame interval, 5 frames per window
- No prediction field — predictions constructed at training data build time
- Narration alignment: current ±1 window narrations as context anchors
- Resumable: saves after each window, detects partial output on restart

### build_training.py
- Reads all `*_annotation.json` files from annotations directory
- For each window t: explanation = window_t, prediction = window_(t + pred_horizon)
- Selects n_frames from 5 (evenly spaced, e.g. 4 from 5 → indices [0,1,3,4])
- `--with_history --history_steps 3`: appends previous N windows' explanations as context in user prompt (for V5)
- Splits by VIDEO (not window) for proper evaluation — seed 42 for reproducibility
- Outputs LLaMA-Factory format: train.json, val.json, test.json, split_info.json, config.json

## Directory Structure

```
sdp/
├── shared/                    # ✅ Shared modules
│   ├── video_frames.py        #   Frame extraction (default 0.2s, inference passes 0.5s)
│   ├── glm_client.py          #   GLM-4.6V API client (zai-sdk, multi-image, 1 retry)
│   ├── qwen_client.py         #   Local Qwen3.5-9B client (unsloth, optional LoRA adapter)
│   ├── vlm.py                 #   Backend selector: get_call_vlm("glm"|"qwen")
│   └── utils.py               #   JSON save, narration CSV load & cleanup
│
├── v1/                        # ✅ V1 single-frame prediction (--backend glm|qwen)
│   ├── prompt.py              #   Single-frame prompt (30-50 word prediction)
│   └── run.py                 #   Entry: extract frames → per-frame VLM → save JSON
│
├── v2/                        # ✅ V2 4-frame window prediction (--backend glm|qwen)
│   ├── prompt.py              #   Few-shot example, no hardcoded intervals
│   └── run.py                 #   Entry: extract frames → 4-frame windows → VLM → save JSON
│
├── v3/                        # ✅ V3 4-frame window + history context (--backend glm|qwen)
│   ├── history.py             #   HistoryManager: deque(maxlen=3), full text, recency labels
│   ├── prompt.py              #   V2 base + history section (current frames > history)
│   └── run.py                 #   Entry: same as V2 + history management & injection
│
├── v4/                        # ✅ V4 fine-tuned Qwen, no history
│   ├── prompt.py              #   V4 dedicated prompt (anti-repetition rules for Prediction)
│   ├── train.py               #   Fine-tune Qwen3.5-9B (LoRA, SFTTrainer)
│   ├── infer.py               #   Quick test on 5 test samples
│   └── run.py                 #   Full inference pipeline (qwen_client + v4 prompt)
│
├── v5/                        # ✅ V5 fine-tuned Qwen, with history
│   ├── prompt.py              #   V5 dedicated prompt (anti-repetition + anti-hallucination)
│   ├── train.py               #   Fine-tune Qwen3.5-9B (LoRA, SFTTrainer)
│   ├── infer.py               #   Quick test on 5 test samples
│   └── run.py                 #   Full inference pipeline (qwen_client + v5 prompt/history)
│
├── annotation/                # ✅ Annotation + training data pipeline
│   ├── prompt.py              #   Annotation prompt (30-50 words, anti-hedging, third person)
│   ├── run_annotate.py        #   Single-pass annotation, resumable, rate-limited (GLM-only)
│   └── build_training.py      #   Annotations → LLaMA-Factory format, supports --with_history
│
├── data/
│   ├── videos/                # 
│   ├── frames/                #   Extracted frames (subdirs by video name)
│   ├── narrations/            #   selected_narrations.csv (from Ego4D)
│   ├── results/               #   Inference output JSONs
│   ├── annotations/           #   annotation JSONs (per video)
│   ├── training_v4/           #   
│   └── training_v5/           #   
│
└── output_standard.md         # Explanation/prediction quality standard
```

## Module Details

### shared/glm_client.py
- `call_vlm(images: List[str], prompt: str) -> str`
- Images first in content, prompt last; 1 retry on failure

### shared/qwen_client.py
- `init_model(model_path, adapter_path, max_seq_length)` — explicit initialization with optional LoRA adapter
- `call_vlm(images: List[str], prompt: str) -> str` — same interface as glm_client
- Lazy loading: model loaded on first `call_vlm()` call or explicit `init_model()` call
- Env vars: `QWEN_MODEL_PATH` (default `/root/sdp/models/qwen3.5-9b`), `QWEN_ADAPTER_PATH` (optional)
- Strips `</think>` tokens from Qwen3.5 output
- Inference params: `temperature=0.3`, `repetition_penalty=1.15`, `do_sample=False`

### shared/vlm.py
- `get_call_vlm("glm"|"qwen")` — returns appropriate `call_vlm` function
- Used by V1/V2/V3 via `--backend` flag; V4/V5 use qwen_client directly for adapter control

### shared/utils.py
- `save_results(results, output_path)` — save dict as JSON
- `load_narrations(csv_path)` — load CSV, strip `#C C` prefix

### v2/prompt.py
- No hardcoded intervals, few-shot example, follows output_standard.md style

### v3/history.py
- `HistoryManager`: `add(e, p)`, `get_history()` → `[3 steps ago]` / `[2 steps ago]` / `[Previous step]`
- Full text history (not compressed), most recent last

### v4/prompt.py
- Dedicated prompt for fine-tuned Qwen, separates Explanation rules and Prediction rules
- Key rule: "Prediction must describe the NEXT action AFTER the current window ends, not a repetition"

### v5/prompt.py
- V4 prompt base + history context injection
- Anti-hallucination rules: only mention objects visible in current frames, Prediction follows current action trend not history predictions

### v4/run.py & v5/run.py
- V4: uses `shared/qwen_client` with V4 LoRA adapter + `v4/prompt.py` (no history)
- V5: uses `shared/qwen_client` with V5 LoRA adapter + `v5/prompt.py` + `v3/history.py` (with history)
- Both accept `--model` and `--adapter` args for custom paths

### annotation/build_training.py
- `--with_history --history_steps N`: appends `Step t-N ... Step t-1` explanations to user prompt
- Without flag: V4-style prompt (frames only); with flag: V5-style prompt (frames + history)
- Same annotation data serves both versions — difference is purely in prompt assembly

## Usage

### Local (Mac — V1/V2/V3 with GLM API)
```bash
cd ~/Desktop/sdp && source .venv/bin/activate

python -m v1.run --video data/videos/<video>.mp4 --output data/results/
python -m v2.run --video data/videos/<video>.mp4 --output data/results/
python -m v3.run --video data/videos/<video>.mp4 --output data/results/
```

### GPU Server (V4/V5 training & inference)
```bash
cd /root/sdp

# V4 Training
python -m v4.train

# V4 Quick inference test
python -m v4.infer

# V4 Full inference pipeline
python -m v4.run --video data/videos/<video>.mp4 --output data/results/

# V5 Training
python -m v5.train

# V5 Quick inference test
python -m v5.infer

# V5 Full inference pipeline
python -m v5.run --video data/videos/<video>.mp4 --output data/results/

# V1/V2/V3 with local Qwen (base model, no adapter)
python -m v1.run --video data/videos/<video>.mp4 --backend qwen
python -m v2.run --video data/videos/<video>.mp4 --backend qwen
python -m v3.run --video data/videos/<video>.mp4 --backend qwen
```

### Annotation & Training Data
```bash
# Annotate a single video (GLM API only)
python -m annotation.run_annotate \
    --video data/videos/<video_uid>.mp4 \
    --narration data/narrations/selected_narrations.csv \
    --output data/annotations/

# Batch annotate all videos
for video in data/videos/*.mp4; do
    python -m annotation.run_annotate \
        --video "$video" \
        --narration data/narrations/selected_narrations.csv \
        --output data/annotations/ --delay 0.5
done

# Build V4 training data (no history)
python -m annotation.build_training \
    --annotations data/annotations/ \
    --output data/training_v4/ \
    --n_frames 4 --pred_horizon 1

# Build V5 training data (with history)
python -m annotation.build_training \
    --annotations data/annotations/ \
    --output data/training_v5/ \
    --n_frames 4 --pred_horizon 1 \
    --with_history --history_steps 3
```

## Completed Iterations

1. **V1/V2 base pipeline** — shared modules + V1/V2 entry points
2. **glm_client.py refactor** — migrated from zhipuai SDK to zai-sdk, upgraded to glm-4.6v, enabled thinking
3. **V2 prompt improvement** — rewrote per output_standard.md, added few-shot example, fixed parse strip issue
4. **V3 development** — sliding window history (3 steps, full text), tested and verified
5. **Annotation pipeline (final)** — single-pass 30-50 word explanations, no compression step, 70 videos annotated
6. **build_training.py** — LLaMA-Factory format, video-level splits, `--with_history` flag for V5
7. **V4 + V5 training data generated** -ready for fine-tuning
8. **Qwen3.5-9B local inference support** — qwen_client.py, vlm.py backend selector, --backend flag for V1/V2/V3
9. **V4/V5 pipeline complete** — train.py, infer.py, run.py for both versions, organized in v4/ and v5/ directories
10. **V4/V5 dedicated prompts + inference tuning** — v4/prompt.py (anti-repetition), v5/prompt.py (anti-repetition + anti-hallucination), qwen_client temperature 1.0→0.3, added repetition_penalty=1.15

## Design Decisions
- V4/V5 have their own prompt.py rather than reusing V2/V3 prompts — fine-tuned models respond differently and need explicit anti-repetition rules for Prediction
- V5 history prompt adds anti-hallucination constraints (only mention visible objects, don't follow stale predictions from history)
- Qwen inference: `temperature=0.3` for stable output with `do_sample=False`, `repetition_penalty=1.15` to discourage Prediction from copying Explanation content

## TODO
- [ ] more trainings
- [ ] V4: run training on GPU server (`python -m v4.train`)
- [ ] V5: run training on GPU server (`python -m v5.train`)
- [ ] Run V4/V5 full inference and evaluate results
- [ ] Evaluation pipeline for fine-tuned models

# EgoForesight

## Project Goal

Build a VLM-based action prediction system using egocentric video (Ego4D FHO dataset). The system analyzes video frames and predicts the next action in natural language. A separate annotation pipeline generates training data for fine-tuning.

## Tech Stack

- **VLM**: GLM-4.6V via zai-sdk (`ZhipuAiClient`), thinking enabled
- **API key**: env var `ZHIPUAI_API_KEY` or `ZAI_API_KEY` (loaded via dotenv)
- **Frame extraction**: OpenCV (cv2)
- **Python venv**: `.venv/` — always `source .venv/bin/activate` before running

## Dataset

- **Source**: Ego4D FHO (Forecasting Hands and Objects)
- **Videos**: 71 clips, 30s–2min each
- **Narration CSV columns**: video_uid, pass, timestamp_sec, timestamp_frame, narration_text, annotation_uid
- **Narration format**: `#C C does something` — `#C C` prefix stripped by `load_narrations()`
- **Known issues**: narration density is uneven; some gaps >5s; sentences are short

## Version Plan

| Version | Input | History | Fine-tune | Status |
|---------|-------|---------|-----------|--------|
| V1 | Single frame | No | No | ✅ Done |
| V2 | 4 frames (2s window) | No | No | ✅ Done |
| V3 | 4 frames | Sliding window (3 steps) | No | ✅ Done (can be improved) |
| V4 | 4 frames | No | Yes | ⏳ Training data ready, needs GPU |
| V5 | 4 frames | Yes (3 steps) | Yes | ⏳ Training data ready, needs GPU |

- **e** = explanation: natural language description of current action (30-50 words)
- **p** = prediction: natural language description of most likely next action

## Annotation & Training Data Pipeline

| Component | Status |
|-----------|--------|
| annotation/run_annotate.py | ✅ Done — single-pass, 30-50 word explanations directly |
| annotation/build_training.py | ✅ Done — supports both V4 (no history) and V5 (with history) |
| 70/71 videos annotated | ✅ Done |
| V4 training data (data/training_v4/) | ✅ 4918 samples (50 train / 10 val / 10 test videos) |
| V5 training data (data/training_v5/) | ✅ 4918 samples, same split, with history context in prompt |

### run_annotate.py
- Single-pass: directly generates 30-50 word explanations (no compression step)
- 1-second windows, 0.2s frame interval, 5 frames per window
- No prediction field — predictions constructed at training data build time
- Narration alignment: current ±1 window narrations as context anchors
- Resumable: saves after each window, detects partial output on restart
- API delay: 0.5s between calls

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
│   ├── glm_client.py          #   VLM API client (zai-sdk, glm-4.6v, multi-image, 1 retry)
│   └── utils.py               #   JSON save, narration CSV load & cleanup
│
├── v1/                        # ✅ V1 single-frame prediction
│   ├── prompt.py              #   Single-frame prompt (30-50 word prediction)
│   └── run.py                 #   Entry: extract frames → per-frame VLM → save JSON
│
├── v2/                        # ✅ V2 4-frame window prediction
│   ├── prompt.py              #   Few-shot example, no hardcoded intervals
│   └── run.py                 #   Entry: extract frames → 4-frame windows → VLM → save JSON
│
├── v3/                        # ✅ V3 4-frame window + history context
│   ├── history.py             #   HistoryManager: deque(maxlen=3), full text, recency labels
│   ├── prompt.py              #   V2 base + history section (current frames > history)
│   └── run.py                 #   Entry: same as V2 + history management & injection
│
├── annotation/                # ✅ Annotation + training data pipeline
│   ├── prompt.py              #   Annotation prompt (30-50 words, anti-hedging, third person)
│   ├── run_annotate.py        #   Single-pass annotation, resumable, rate-limited
│   └── build_training.py      #   Annotations → LLaMA-Factory format, supports --with_history
│
├── data/
│   ├── videos/                #   71 raw videos (.mp4)
│   ├── frames/                #   Extracted frames (subdirs by video name)
│   ├── narrations/            #   selected_narrations.csv
│   ├── results/               #   Inference output JSONs (v1/v2/v3)
│   ├── annotations/           #   70 annotation JSONs (per video)
│   ├── training_v4/           #   V4 training data (no history): train/val/test.json
│   └── training_v5/           #   V5 training data (with history): train/val/test.json
│
└── output_standard.md         # Explanation/prediction quality standard
```

## Module Details

### shared/glm_client.py
- `call_vlm(images: List[str], prompt: str) -> str`
- Images first in content, prompt last; 1 retry on failure

### shared/utils.py
- `save_results(results, output_path)` — save dict as JSON
- `load_narrations(csv_path)` — load CSV, strip `#C C` prefix

### v2/prompt.py
- No hardcoded intervals, few-shot example, follows output_standard.md style

### v3/history.py
- `HistoryManager`: `add(e, p)`, `get_history()` → `[3 steps ago]` / `[2 steps ago]` / `[Previous step]`
- Full text history (not compressed), most recent last

### annotation/build_training.py
- `--with_history --history_steps N`: appends `Step t-N ... Step t-1` explanations to user prompt
- Without flag: V4-style prompt (frames only); with flag: V5-style prompt (frames + history)
- Same annotation data serves both versions — difference is purely in prompt assembly

## Usage

```bash
cd ~/Desktop/sdp && source .venv/bin/activate

# Inference (V1/V2/V3)
python -m v1.run --video data/videos/<video>.mp4 --output data/results/
python -m v2.run --video data/videos/<video>.mp4 --output data/results/
python -m v3.run --video data/videos/<video>.mp4 --output data/results/

# Annotate a single video
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
7. **V4 + V5 training data generated** — 4918 samples each, 50/10/10 train/val/test split

## TODO

- [ ] V4: fine-tune GLM-4.6V without history (GPU required)
- [ ] V5: fine-tune GLM-4.6V with history (GPU required)
- [ ] Evaluation pipeline for fine-tuned models
- [ ] Continued prompt iteration and optimization

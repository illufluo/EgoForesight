# Action Prediction Project — Progress Sync

> This file is used to sync project progress with Claude on the web. Last updated: 2026-03-26

## Project Goal

Build a VLM-based action prediction system using egocentric video (Ego4D FHO dataset). The system analyzes video frames and predicts the next action in natural language. A separate annotation pipeline generates training data for future fine-tuning.

## Tech Stack

- **VLM**: GLM-4.6V via zai-sdk (`ZhipuAiClient`), thinking enabled
- **API key**: env var `ZHIPUAI_API_KEY` or `ZAI_API_KEY` (loaded via dotenv)
- **Frame extraction**: OpenCV (cv2)
- **Python venv**: `.venv/` — always `source .venv/bin/activate` before running

## Dataset

- **Source**: Ego4D FHO (Forecasting Hands and Objects)
- **Videos**: ~70 clips, 30s–2min each
- **Narration CSV columns**: video_uid, pass, timestamp_sec, timestamp_frame, narration_text, annotation_uid
- **Narration format**: `#C C does something` — `#C C` prefix stripped by `load_narrations()`
- **Known issues**: narration density is uneven; some gaps >5s; sentences are short

## Version Plan

| Version | Input | History | Fine-tune | Status |
|---------|-------|---------|-----------|--------|
| V1 | Single frame | No | No | ✅ Done |
| V2 | 4 frames (2s window) | No | No | ✅ Done |
| V3 | 4 frames | Sliding window (3 steps) | No | ✅ Done (can be improved) |
| V4 | 4 frames | No | Yes | ❌ Not started |
| V5 | 4 frames | Yes | Yes | ❌ Not started |

- **e** = explanation: natural language description of current action (30-50 words)
- **p** = prediction: natural language description of most likely next action

## Annotation Pipeline

| Component | Status |
|-----------|--------|
| annotation/run_annotate.py (explanation_full + compression) | ✅ Done and tested |
| annotation/build_training.py (annotation → fine-tune data) | ❌ Placeholder only |

- **Window**: 1-second (not 2s), 0.2s frame interval, 5 frames per window
- **2-pass process**: Pass 1 generates explanation_full (80-100 words, with images), Pass 2 compresses to explanation_compact (30-50 words, text-only VLM call)
- **No prediction field** — predictions are constructed from neighboring explanations at training time
- **Narration alignment**: matches Ego4D narrations to windows by timestamp, feeds current ±1 window as context anchors
- **Resumable**: saves after each window, detects partial output on restart

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
├── annotation/                # ✅ Data annotation pipeline
│   ├── prompt.py              #   Annotation prompt + compression prompt
│   ├── compress.py            #   explanation_full → explanation_compact (text-only VLM)
│   ├── run_annotate.py        #   2-pass pipeline, resumable, rate-limited
│   └── build_training.py      #   Placeholder — annotations → fine-tune training data
│
├── data/
│   ├── videos/                #   Raw videos (.mp4)
│   ├── frames/                #   Extracted frames (subdirs by video name)
│   ├── narrations/            #   selected_narrations.csv (71 videos)
│   ├── results/               #   Inference output JSONs (v1/v2/v3)
│   └── annotations/           #   Annotation output JSONs
│
├── output_standard.md         # Explanation/prediction quality standard
├── glm46v_interface.py        # Legacy GLM interface (reference only)
└── video_frames.py            # Legacy frame extraction (reference only)
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

### annotation/run_annotate.py
- 1s windows, 5 frames each, 2-pass (annotate → compress), resumable
- Narration context from current ±1 windows as reference anchors

## Usage

```bash
cd ~/Desktop/sdp && source .venv/bin/activate

# Inference
python -m v1.run --video data/videos/test.mp4 --output data/results/
python -m v2.run --video data/videos/test.mp4 --output data/results/
python -m v3.run --video data/videos/test.mp4 --output data/results/

# Annotation
python -m annotation.run_annotate \
    --video data/videos/<video_uid>.mp4 \
    --narration data/narrations/selected_narrations.csv \
    --output data/annotations/
```

## Completed Iterations

1. **V1/V2 base pipeline** — shared modules + V1/V2 entry points
2. **glm_client.py refactor** — migrated from zhipuai SDK to zai-sdk, upgraded to glm-4.6v, enabled thinking
3. **V2 prompt improvement** — rewrote per output_standard.md, added few-shot example, fixed parse strip issue
4. **V3 development** — sliding window history (3 steps, full text), tested and verified
5. **Annotation pipeline** — 2-pass annotation (detailed → compressed), 1s windows, narration alignment, resumable, tested on 1 video (18 windows, all OK)

## TODO

- [ ] V4: no history + fine-tune
- [ ] V5: history + fine-tune
- [ ] build_training.py: convert annotation JSONs to fine-tune training data
- [ ] Batch annotation across all ~70 Ego4D videos
- [ ] Continued prompt iteration and optimization

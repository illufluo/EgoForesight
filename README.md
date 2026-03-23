# Action Prediction Project — Progress Sync

> This file is used to sync project progress with Claude on the web. Last updated: 2026-03-22

## Project Goal

A VLM-based (GLM-4.6V) action prediction system using egocentric video from the Ego4D dataset. The system analyzes video frames and predicts the next action in natural language.

## Version Plan

| Version | Input | History | Fine-tune | Status |
|---------|-------|---------|-----------|--------|
| V1 | Single frame | No | No | ✅ Done |
| V2 | 4 frames (2s window) | No | No | ✅ Done|
| V3 | 4 frames | Sliding window (3 steps) | No | ✅ Done |
| V4 | 4 frames | No | Yes | ❌ Not started |
| V5 | 4 frames | Yes | Yes | ❌ Not started |

## Current Directory Structure

```
sdp/
├── shared/                    # ✅ Shared modules
│   ├── video_frames.py        #   Frame extraction (default 0.2s interval, run.py passes 0.5s)
│   ├── glm_client.py          #   VLM API client (zai-sdk, glm-4.6v, multi-image, 1 retry)
│   └── utils.py               #   Utilities (JSON save, narration CSV load & cleanup)
│
├── v1/                        # ✅ V1 single-frame prediction
│   ├── prompt.py              #   Single-frame prompt (30-50 word prediction)
│   └── run.py                 #   Entry: extract frames → per-frame VLM call → save JSON
│
├── v2/                        # ✅ V2 4-frame window prediction (prompt improved)
│   ├── prompt.py              #   4-frame prompt, few-shot example, no hardcoded intervals
│   └── run.py                 #   Entry: extract frames → 4-frame windows → VLM call → save JSON
│
├── v3/                        # ✅ V3 4-frame window + history context
│   ├── history.py             #   HistoryManager: sliding window of last 3 e/p (full text)
│   ├── prompt.py              #   V3 prompt = V2 base prompt + history section
│   └── run.py                 #   Entry: same as V2 flow, plus history management & injection
│
├── data/
│   ├── videos/                #   Raw videos (test.mp4 available)
│   ├── frames/                #   Extracted frames (subdirs by video name)
│   ├── narrations/            #   Ego4D narration CSVs
│   └── results/               #   Output JSONs (test_v1/v2/v3.json available)
│
├── output_standard.md         # Explanation/prediction quality standard
├── project_overview.md        # Project overview document
├── claude_code_task_v1v2.md   # V1/V2 original task description
├── claude_code_tasks.md       # Task 1 (V2 prompt improvement) description
├── claude_code_task2_v3.md    # Task 2 (V3 development) description
├── glm46v_interface.py        # Legacy GLM interface (reference only, replaced)
└── video_frames.py            # Legacy frame extraction (reference only, copied to shared/)
```

## Module Details

### shared/glm_client.py
- Signature: `call_vlm(images: List[str], prompt: str) -> str`
- Uses `zai-sdk` (`ZhipuAiClient`), model `glm-4.6v`, thinking enabled
- API key from env var `ZHIPUAI_API_KEY` or `ZAI_API_KEY` (via dotenv)
- Images placed first in content, prompt last
- 1 automatic retry on failure, then raises exception

### shared/utils.py
- `save_results(results, output_path)` — save dict as JSON
- `load_narrations(csv_path)` — load narration CSV, strip `#C C` prefix from narration_text

### shared/video_frames.py
- `extract_frames(video_path, output_dir, interval=0.2)` — extract frames at fixed intervals
- `get_video_info(video_path)` — get basic video metadata

### v2/prompt.py (improved)
- No hardcoded seconds/frame counts, uses "consecutive frames at regular intervals"
- Includes few-shot example to stabilize output style
- Follows output_standard.md: action verbs first, specific objects, hand details, temporal progression

### v2/run.py, v3/run.py
- `_parse_response()` uses regex, handles Explanation/Prediction labels in any order
- Falls back to full raw response if parsing fails

### v3/history.py
- `HistoryManager` class, `deque(maxlen=3)` sliding window
- `add(explanation, prediction)` — stores full text (no compression)
- `get_history()` — formatted text block with recency labels: `[3 steps ago]`, `[2 steps ago]`, `[Previous step]`
- Most recent step appears last (closest to current context)

### v3/prompt.py
- `build_prompt(history=None)` — appends history section when available, otherwise identical to V2
- History section instructs model: current frames are primary, history is supplementary only, do not copy history content, trust current frames over conflicting history

## Usage

```bash
cd ~/Desktop/sdp
source .venv/bin/activate

# V1: single-frame prediction
python -m v1.run --video data/videos/test.mp4 --output data/results/

# V2: 4-frame window prediction
python -m v2.run --video data/videos/test.mp4 --output data/results/

# V3: 4-frame window + history context
python -m v3.run --video data/videos/test.mp4 --output data/results/
```

## Completed Iterations

1. **V1/V2 base pipeline** — shared modules + V1/V2 entry points
2. **glm_client.py refactor** — migrated from zhipuai SDK to zai-sdk, upgraded to glm-4.6v, enabled thinking
3. **V2 prompt improvement (Task 1)** — rewrote prompt per output_standard.md, added few-shot example, fixed parse strip issue
4. **V3 development (Task 2)** — added history.py + v3/prompt.py + v3/run.py, sliding window history mechanism, tested and verified

## TODO

- [ ] V4: no history + fine-tune
- [ ] V5: history + fine-tune
- [ ] Data annotation enhancement (use VLM to generate richer annotations as training data for V4/V5 and evaluation ground truth)
- [ ] Continued prompt iteration and optimization
- [ ] Batch testing on more Ego4D videos

# Task: Complete Annotation + Training Data Pipeline (Final Version)

> This task document is for Claude Code. It describes the FINAL pipeline that produces fine-tune-ready training data.
> This replaces the previous annotation pipeline. The key changes are:
> 1. Drop the compression step — generate 30-50 word explanations directly
> 2. Add build_training.py — converts annotations into training-ready JSON
> 3. Reduce API delay to 0.5s

## Overview

Two scripts, run in sequence:

```
run_annotate.py  →  annotation JSONs (per video, explanation only)
build_training.py  →  training-ready JSON (all videos combined, explanation + prediction)
```

## Part 1: Updated run_annotate.py

### What Changes From Current Version

1. **Remove the compression step entirely.** No Pass 2, no compress.py calls. The annotation prompt now directly asks for 30-50 words instead of 80-100 words.

2. **Remove the `explanation_full` / `explanation_compact` split.** Each window now has a single `explanation` field (30-50 words).

3. **Reduce API_DELAY from 1.0 to 0.5 seconds.**

4. **Update the annotation prompt** to:
   - Request 30-50 words (not 80-100)
   - Add explicit anti-hedging rule: "Do NOT use hedging words (possibly, likely, appears to, seems to) — describe only what you can clearly see, or omit uncertain details"
   - Add anti-meta rule for compression-style issues: "Always use third person, never use 'you' or 'your'"
   - Keep all existing good rules (temporal progression, specific objects, hand details, dynamic verbs, narration anchors)

### Updated Output Format

```json
{
  "video_uid": "783d36f8-...",
  "video_path": "/path/to/video.mp4",
  "window_duration_sec": 1.0,
  "frame_interval_sec": 0.2,
  "frames_per_window": 5,
  "total_windows": 27,
  "windows": [
    {
      "window_id": 0,
      "time_range": [0.0, 1.0],
      "frame_paths": [
        "data/frames/video_name/frame_0001.jpg",
        "data/frames/video_name/frame_0002.jpg",
        "data/frames/video_name/frame_0003.jpg",
        "data/frames/video_name/frame_0004.jpg",
        "data/frames/video_name/frame_0005.jpg"
      ],
      "explanation": "Right hand extends toward smartphone on wooden table, taps screen with index finger, scrolling through display. Blue water bottle and metal dish remain stationary as the hand adjusts content on screen.",
      "ego4d_narrations": [
        {"timestamp_sec": 0.63, "text": "scrolled the phone"},
        {"timestamp_sec": 0.63, "text": "operates phone"}
      ],
      "raw_vlm_response": "...",
      "annotation_status": "ok"
    }
  ]
}
```

Note: `explanation_full` and `explanation_compact` are replaced by a single `explanation` field.

### Updated Annotation Prompt

The prompt should be rewritten to produce 30-50 word explanations directly. Key rules:

```
- Describe the action progression across frames (first... then... finally...)
- Name specific objects (e.g. "white cup", "metal stirrer", "wooden board")
- Include hand details: which hand does what
- Start with action verbs, use dynamic verbs (reaches, grasps, lifts, places, stirs)
- Do NOT start with "The person is currently..."
- Do NOT reference the frames themselves (no "in frame 3", "across the images")
- Do NOT predict future actions — only describe what is visible
- Do NOT use hedging words (possibly, likely, appears to, seems to) — describe only what you can clearly see, or omit uncertain details
- Always use third person, never use "you" or "your"
- 30-50 words, concise but specific
```

Keep the narration context injection (current ±1 windows) as-is — it works correctly.

### Resumability

Keep the existing resumability logic, but adapt it to the new single-field format. Check if a window already has an `explanation` field (not None) to determine if it's already done.

### Files to Modify

- `annotation/prompt.py` — rewrite `build_annotation_prompt()`, remove `build_compression_prompt()`
- `annotation/run_annotate.py` — remove Pass 2 (compression), update output format, change delay to 0.5
- `annotation/compress.py` — can be deleted or left as unused

---

## Part 2: New build_training.py

This script converts annotation JSONs into fine-tune training data. It runs AFTER all videos are annotated.

### Usage

```bash
python -m annotation.build_training \
    --annotations data/annotations/ \
    --output data/training/ \
    --n_frames 4 \
    --pred_horizon 1 \
    --test_ratio 0.15 \
    --val_ratio 0.15
```

### Parameters

- `--annotations`: directory containing all `*_annotation.json` files
- `--output`: directory for output training files
- `--n_frames`: how many frames to select from each window's 5 frames (default: 4)
- `--pred_horizon`: prediction horizon h — predict h windows ahead (default: 1)
- `--test_ratio`: fraction of videos for test set (default: 0.15)
- `--val_ratio`: fraction of videos for validation set (default: 0.15)

### Core Logic

```python
for each annotation JSON:
    for each window t where (t + pred_horizon) < total_windows:
        
        # 1. Select n_frames from window t's 5 frame paths
        #    Evenly spaced selection. Examples:
        #    n=4 from 5: pick indices [0, 1, 3, 4] (skip middle)
        #    n=3 from 5: pick indices [0, 2, 4] (every other)
        #    n=5 from 5: pick all
        selected_frames = select_frames(window_t.frame_paths, n_frames)
        
        # 2. Explanation = this window's explanation
        explanation = window_t.explanation
        
        # 3. Prediction = explanation from window (t + pred_horizon)
        #    Direct copy, no VLM rewrite needed
        prediction = window_(t + pred_horizon).explanation
        
        # 4. Skip if either has annotation_status == "error"
        
        # 5. Assemble training sample
        sample = {
            "messages": [
                {
                    "role": "user", 
                    "content": build_inference_prompt(n_frames)
                },
                {
                    "role": "assistant",
                    "content": f"Explanation: {explanation}\nPrediction: {prediction}"
                }
            ],
            "images": selected_frames
        }
```

### Inference Prompt in Training Data

The user message in training samples should match the V2 inference prompt style — this is what the model will see at inference time after fine-tuning. Use a prompt like:

```
<image><image><image><image>
You are analyzing consecutive frames from a first-person (egocentric) video, shown in chronological order at regular intervals.

Task: Provide an Explanation of the current action and a Prediction of the next action.

Rules:
- Describe action progression across frames (first... then... finally...), not a single summary
- Name specific objects (e.g. "white cup", "wooden board"), never use vague words like "items" or "objects"
- Include hand details: which hand does what
- Start with action verbs, use dynamic verbs (reaches, grasps, lifts, places, stirs)
- Do NOT start with "The person is currently..."
- Do NOT reference the frames themselves (no "as seen in the frames", "across the images")
- No vague hedging ("possibly", "might", "could potentially")
- Prediction: one clear next action with brief reasoning linking to current action trend
- Each section: 30-50 words

Now analyze the provided frames. Respond with exactly two lines:
Explanation: <your explanation>
Prediction: <your prediction>
```

The number of `<image>` tokens must match `n_frames`.

### Dataset Split

Split by VIDEO, not by window. All windows from the same video go into the same split. This ensures the test set contains videos the model has never seen.

```python
videos = list of all video_uids
random.shuffle(videos, seed=42)  # reproducible split

n_test = int(len(videos) * test_ratio)
n_val = int(len(videos) * val_ratio)

test_videos = videos[:n_test]
val_videos = videos[n_test:n_test + n_val]
train_videos = videos[n_test + n_val:]
```

### Output Files

```
data/training/
├── train.json          # LLaMA-Factory format, training samples
├── val.json            # LLaMA-Factory format, validation samples  
├── test.json           # LLaMA-Factory format, test samples
├── split_info.json     # Which videos went into which split
└── config.json         # Parameters used (n_frames, pred_horizon, etc.)
```

`split_info.json`:
```json
{
  "n_frames": 4,
  "pred_horizon": 1,
  "total_videos": 70,
  "total_samples": 4100,
  "train_videos": ["uid1", "uid2", ...],
  "val_videos": ["uid3", ...],
  "test_videos": ["uid4", ...],
  "train_samples": 2870,
  "val_samples": 615,
  "test_samples": 615
}
```

### Frame Path Handling

Frame paths in annotation JSONs are relative (e.g., `data/frames/video_name/frame_0001.jpg`). In the training JSON, keep them relative — the fine-tuning framework will need a base path configured separately. Add a `--frame_base` argument (default: current directory) that can be prepended if needed.

---

## Execution Order

```bash
# Step 1: Annotate all videos (run for each video, or use a batch script)
for video in data/videos/*.mp4; do
    python -m annotation.run_annotate \
        --video "$video" \
        --narration data/narrations/selected_narrations.csv \
        --output data/annotations/ \
        --delay 0.5
done

# Step 2: Build training data from all annotations
python -m annotation.build_training \
    --annotations data/annotations/ \
    --output data/training/ \
    --n_frames 4 \
    --pred_horizon 1
```

## Important Notes

- Error windows (annotation_status == "error") should be skipped when building training data
- The last `pred_horizon` windows of each video cannot produce training samples (no future explanation to use as prediction) — this is expected
- All training samples from the same video MUST be in the same split (train/val/test)
- Use a fixed random seed (42) for reproducible splits
- Print summary statistics after building: total samples, per-split counts, samples per video average

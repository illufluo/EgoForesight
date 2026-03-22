# Task: Build V1 and V2 Action Prediction Pipeline

## Context
I'm building a VLM-based action prediction system using egocentric video (Ego4D dataset). The system analyzes video frames and predicts what the person will do next. See `project_overview.md` in project knowledge for full background.

## What to Build Now
**V1** (single frame → prediction) and **V2** (4 frames → prediction) only. Keep it simple and minimal.

## Project Structure

```
project/
├── shared/
│   ├── video_frames.py      # Frame extraction (already exists, copy from uploads)
│   ├── glm_client.py        # VLM API client, supports multi-image input
│   └── utils.py             # Save results JSON, load narration CSV, etc.
│
├── v1/
│   ├── run.py               # Entry point: extract frames → build prompt → call VLM → save results
│   └── prompt.py            # V1 prompt construction (single frame)
│
├── v2/
│   ├── run.py               # Entry point: same flow but with 4-frame windows
│   └── prompt.py            # V2 prompt construction (4 frames, temporal order)
│
└── data/
    ├── videos/              # Raw video files (.mp4)
    ├── frames/              # Extracted frames, organized as frames/{video_uid}/frame_0001.jpg
    ├── narrations/          # Narration CSV files
    └── results/             # Output JSON files
```

## Module Specifications

### shared/glm_client.py
- Function: `call_vlm(images: List[str], prompt: str) -> str`
- Input: list of image file paths (1 for V1, 4 for V2), prompt string
- Encodes images to base64, sends to GLM-4V API via ZhipuAI SDK
- API key from env var `ZHIPUAI_API_KEY`
- Returns raw response text
- Model: "glm-4v"

### shared/video_frames.py
- Already exists. Copy from uploaded file. No modifications needed.
- Used by each version's run.py to extract frames.

### shared/utils.py
- `save_results(results: dict, output_path: str)` — save to JSON
- `load_narrations(csv_path: str) -> List[dict]` — load narration CSV, clean `#C C` prefix from narration_text

### v1/prompt.py
- Function: `build_prompt() -> str`
- Returns a prompt asking VLM to analyze a single frame from egocentric video
- Output should request: prediction of the most likely next action (30-50 words)
- Simple first version is fine, we will iterate on prompts later

### v2/prompt.py
- Function: `build_prompt() -> str`
- Returns a prompt for 4 temporally ordered frames (0.5s interval, 2s window)
- Must tell VLM the frames are in chronological order with 0.5s intervals
- Output should request: explanation of current action + prediction of next action (each 30-50 words)

### v1/run.py
- Entry point with argparse: `python v1/run.py --video path/to/video.mp4 --output data/results/`
- Flow: extract frames (0.5s interval) → for each frame, call VLM with V1 prompt → collect results → save JSON
- For V1, every frame is an independent prediction (no windowing needed, just iterate)

### v2/run.py
- Entry point with argparse: `python v2/run.py --video path/to/video.mp4 --output data/results/`
- Flow: extract frames (0.5s interval) → group into 4-frame windows (frames 1-4, 5-8, 9-12...) → for each window, call VLM with V2 prompt → collect results → save JSON

## Output JSON Format

```json
{
  "video_path": "data/videos/xxx.mp4",
  "version": "V1",
  "frame_interval": 0.5,
  "predictions": [
    {
      "window_id": 1,
      "time_range": [0.0, 0.0],
      "frames": ["frame_0001.jpg"],
      "prediction": "The person will likely reach for...",
      "raw_response": "full VLM response text"
    }
  ]
}
```

For V2, time_range would be [0.0, 1.5], frames would have 4 entries, and there would also be an "explanation" field.

## Important Notes
- Keep code simple and clean. No over-engineering.
- Prompts are placeholders — we will refine them later. Just make them reasonable.
- Later we will add V3 (history), V4/V5 (fine-tune) as separate folders. The shared/ modules should be general enough to support this, but don't build for it now.
- Add basic error handling for API failures (retry once, then log and skip).
- Print progress to console (e.g., "Processing window 3/15...").

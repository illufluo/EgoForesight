# Action Prediction Project Overview

## Project Goal
Build a VLM-based action prediction system using egocentric video. The system analyzes video frames and predicts what the person will do next, using natural language descriptions.

## Version Definitions

| Version | Input | History | Fine-tune | Purpose |
|---------|-------|---------|-----------|---------|
| V1 | Single frame | No | No | Simplest baseline, sanity check |
| V2 | 4 frames (2s window, 0.5s interval) | No | No | Multi-frame baseline |
| V3 | 4 frames | Yes (sliding window of past e/p) | No | Core method, proves history helps |
| V4 | 4 frames | No | Yes | Fine-tune baseline |
| V5 | 4 frames | Yes | Yes | Final version |

- **e** = explanation: natural language description of current action (30-50 words for human, compressed to ~10 keywords for history input)
- **p** = prediction: natural language description of most likely next action

## V3 History Mechanism
- Uses past 3 time steps: e(t-3, t-2, t-1) and p(t-3, t-2, t-1)
- History is compressed to keywords (~10-15 words) to avoid error accumulation
- Current frames always take priority over history

## Dataset
- **Source**: Ego4D FHO (Forecasting Hands and Objects)
- **Videos**: ~70 clips, 30s–2min each, more available if needed
- **Frame extraction**: 0.5s interval → 4 frames per 2s window
- **Narration format** (CSV columns):
  - `video_uid`: video identifier
  - `pass`: narration_pass_1 or narration_pass_2 (two annotators per video)
  - `timestamp_sec`: precise timestamp
  - `timestamp_frame`: frame number
  - `narration_text`: format is `#C C does something` (needs `#C C` prefix removed)
  - `annotation_uid`: unique annotation ID
- **Known issues**: narration density is uneven; some gaps >5s exist; narration sentences are short

## Data Annotation Enhancement (Separate Workflow)
- Use VLM to generate richer, more uniform annotations for all 2s windows
- Combine VLM output with existing Ego4D narrations as anchor points
- Output serves as: evaluation ground truth for V1-V3, training labels for V4-V5
- This is an **offline data preparation step**, not part of the inference pipeline

## Tech Stack
- **VLM**: GLM-4.6V (via ZhipuAI API now; local deployment for fine-tune later)
- **API**: ZhipuAI SDK, env var `ZHIPUAI_API_KEY`
- **Frame extraction**: OpenCV (`cv2`)
- **Dev tools**: Claude Code for development

## Current Progress
- Frame extraction module: done and working (`video_frames.py`)
- GLM-4.6V API interface: prototype done (`glm46v_interface.py`)
- Narration data: downloaded and formatted

## Active Workstreams
1. **V1/V2/V3 pipeline development** — building the inference pipeline
2. **Data annotation enhancement** — using VLM to enrich ground truth labels

# Task 2: Build V3 Pipeline (do this after Task 1 is done and tested)

## Context
V3 adds history to V2: past explanations and predictions are included in the prompt as context. See `output_standard.md` in project knowledge for explanation/prediction quality standards.

Key design decision: history is passed as **full text** (not compressed to keywords). The prompt must emphasize that current frames take priority over history.

## Project Structure
```
v3/
├── run.py          # Entry point, similar to v2 but with history management
├── prompt.py       # V3 prompt construction (frames + history context)
└── history.py      # History manager: sliding window of past e/p
```

## Module Specifications

### v3/history.py

**`class HistoryManager`**
- Maintains a sliding window of past 3 time steps
- Stores full explanation and prediction text at each step (no compression)
- Methods:
  - `add(explanation: str, prediction: str)` — store the full text
  - `get_history() -> str` — format history as text block for prompt injection, labeled by recency (e.g., "2 steps ago:", "1 step ago:", "Previous step:")
  - `clear()` — reset history (for new video)
- When fewer than 3 past steps exist (beginning of video), just use whatever is available
- Most recent step should appear last (closest to current context)

### v3/prompt.py

**`build_prompt(history: str = None) -> str`**
- Base prompt is same as V2 (use the improved version from Task 1)
- When history is provided (not None, not empty), append a history section
- The history section must clearly state:
  - "Below is context from recent time steps for reference"
  - "Your analysis of the CURRENT FRAMES is the primary basis for your response"
  - "Use history only as supplementary context — do NOT copy or repeat history content"
  - "If history conflicts with what you observe in current frames, trust the current frames"
- When no history (first window), just use base V2 prompt

### v3/run.py
- Similar to v2/run.py but:
  - Initialize HistoryManager at start
  - After each window: parse explanation and prediction from response → add to history
  - Before each window: get_history() and pass to build_prompt()
  - First window has no history (same as V2)

## Output JSON Format
Same as V2 but add a `history_input` field to each window:
```json
{
  "window_id": 3,
  "time_range": [4.0, 5.5],
  "frames": ["frame_0009.jpg", ...],
  "explanation": "...",
  "prediction": "...",
  "history_input": "2 steps ago: e: ... p: ... | 1 step ago: e: ... p: ...",
  "raw_response": "..."
}
```
First window's history_input should be null.

## Important Notes
- No keyword compression — store and pass full text
- The sliding window is always max 3 past steps
- History text must not overwhelm the prompt — current frames are always primary
- If explanation/prediction parsing fails for a window, store the raw_response as-is in history (don't skip it)

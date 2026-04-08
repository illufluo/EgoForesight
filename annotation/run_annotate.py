"""
Annotation Pipeline: video → annotation JSON (single-pass, 30-50 word explanations).

Usage:
    python -m annotation.run_annotate \
        --video data/videos/<video_uid>.mp4 \
        --narration data/narrations/selected_narrations.csv \
        --output data/annotations/
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.video_frames import extract_frames
from shared.glm_client import call_vlm
from shared.utils import load_narrations
from annotation.prompt import build_annotation_prompt

WINDOW_DURATION = 1.0   # seconds per window
FRAME_INTERVAL = 0.2    # seconds between frames
FRAMES_PER_WINDOW = 5   # 1.0 / 0.2
API_DELAY = 0.5          # seconds between VLM calls


def main():
    parser = argparse.ArgumentParser(description="Annotation pipeline: generate action annotations")
    parser.add_argument("--video", required=True, help="Path to input video (.mp4)")
    parser.add_argument("--narration", required=True, help="Path to narration CSV")
    parser.add_argument("--output", default="data/annotations/", help="Output directory")
    parser.add_argument("--delay", type=float, default=API_DELAY, help="Delay between API calls (seconds)")
    args = parser.parse_args()

    video_path = os.path.abspath(args.video)
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_file = os.path.join(args.output, f"{video_name}_annotation.json")

    # --- Step 1: Extract frames ---
    print(f"Extracting frames from {video_path} (interval={FRAME_INTERVAL}s)...")
    frames_dir = os.path.join("data", "frames", video_name)
    frame_list = extract_frames(video_path, frames_dir, interval=FRAME_INTERVAL)
    print(f"Extracted {len(frame_list)} frames.")

    # --- Step 2: Group into 1-second windows ---
    windows = _group_into_windows(frame_list)
    total_windows = len(windows)
    print(f"Created {total_windows} windows ({WINDOW_DURATION}s each, {FRAMES_PER_WINDOW} frames).")

    # --- Step 3: Load & align narrations ---
    narrations = _load_video_narrations(args.narration, video_name)
    print(f"Loaded {len(narrations)} narrations for this video.")
    aligned = _align_narrations(narrations, windows)

    # --- Step 4: Check for resumable state ---
    completed = _load_partial(output_file)
    if completed:
        print(f"Resuming: {len(completed)} windows already annotated, continuing from window {len(completed)}.")

    # --- Step 5: Annotation pass ---
    print(f"\n=== Generating explanations (30-50 words) ===")
    for i, window in enumerate(windows):
        if i < len(completed):
            continue  # already done

        print(f"  Annotating window {i + 1}/{total_windows} [{window['t_start']:.1f}s - {window['t_end']:.1f}s]...")

        # Build narration context (current window ± 1)
        narration_ctx = _format_narration_context(aligned, i, total_windows)
        prompt = build_annotation_prompt(narration_ctx)
        img_paths = [f["image_path"] for f in window["frames"]]

        try:
            response = call_vlm(img_paths, prompt)
            explanation = _parse_description(response)
            status = "ok"
        except Exception as e:
            print(f"    ERROR: {e}")
            response = f"ERROR: {e}"
            explanation = response
            status = "error"

        entry = {
            "window_id": i,
            "time_range": [window["t_start"], window["t_end"]],
            "frame_paths": [os.path.relpath(f["image_path"]) for f in window["frames"]],
            "explanation": explanation,
            "ego4d_narrations": aligned[i],
            "raw_vlm_response": response,
            "annotation_status": status,
        }
        completed.append(entry)

        # Save after each window for resumability
        _save_partial(output_file, video_path, video_name, completed, total_windows)

        if i < total_windows - 1:
            time.sleep(args.delay)

    # --- Final save ---
    _save_partial(output_file, video_path, video_name, completed, total_windows)
    ok_count = sum(1 for w in completed if w["annotation_status"] == "ok")
    print(f"\nDone. {ok_count}/{total_windows} windows annotated successfully → {output_file}")


# ─── Helpers ───────────────────────────────────────────────────────────

def _group_into_windows(frame_list):
    """Group frames into non-overlapping 1-second windows (5 frames each)."""
    windows = []
    for i in range(0, len(frame_list), FRAMES_PER_WINDOW):
        chunk = frame_list[i : i + FRAMES_PER_WINDOW]
        if len(chunk) < FRAMES_PER_WINDOW:
            break  # discard incomplete tail
        windows.append({
            "frames": chunk,
            "t_start": chunk[0]["timestamp"],
            "t_end": chunk[0]["timestamp"] + WINDOW_DURATION,
        })
    return windows


def _load_video_narrations(csv_path, video_name):
    """Load narrations from CSV, filtered to this video's uid."""
    all_narrations = load_narrations(csv_path)
    video_narrations = []
    for n in all_narrations:
        if n.get("video_uid", "") == video_name:
            try:
                ts = float(n["timestamp_sec"])
            except (ValueError, KeyError):
                continue
            video_narrations.append({
                "timestamp_sec": ts,
                "text": n["narration_text"],
            })
    video_narrations.sort(key=lambda x: x["timestamp_sec"])
    return video_narrations


def _align_narrations(narrations, windows):
    """For each window, find narrations whose timestamp falls within [t_start, t_end)."""
    aligned = []
    for w in windows:
        matched = [
            n for n in narrations
            if w["t_start"] <= n["timestamp_sec"] < w["t_end"]
        ]
        aligned.append(matched)
    return aligned


def _format_narration_context(aligned, window_idx, total_windows):
    """Format narrations from current window and ±1 neighbors."""
    entries = []

    # Previous window
    if window_idx > 0:
        for n in aligned[window_idx - 1]:
            entries.append(f"  [{n['timestamp_sec']:.1f}s] {n['text']}")

    # Current window
    for n in aligned[window_idx]:
        entries.append(f"  [{n['timestamp_sec']:.1f}s] {n['text']} (current)")

    # Next window
    if window_idx < total_windows - 1:
        for n in aligned[window_idx + 1]:
            entries.append(f"  [{n['timestamp_sec']:.1f}s] {n['text']}")

    return "\n".join(entries)


def _parse_description(response):
    """Extract description text from VLM response."""
    text = response.strip()
    # If model prefixed with "Description:", strip it
    lower = text.lower()
    if lower.startswith("description:"):
        text = text[len("description:"):].strip()
    return text


def _load_partial(output_file):
    """Load already-completed windows from a partial annotation file."""
    if not os.path.exists(output_file):
        return []
    try:
        with open(output_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Only count windows that have a non-None explanation
        windows = data.get("windows", [])
        done = []
        for w in windows:
            if w.get("explanation") is not None:
                done.append(w)
            else:
                break  # stop at first incomplete
        return done
    except (json.JSONDecodeError, KeyError):
        return []


def _save_partial(output_file, video_path, video_name, windows, total_windows):
    """Save current annotation progress to JSON."""
    result = {
        "video_uid": video_name,
        "video_path": video_path,
        "window_duration_sec": WINDOW_DURATION,
        "frame_interval_sec": FRAME_INTERVAL,
        "frames_per_window": FRAMES_PER_WINDOW,
        "total_windows": total_windows,
        "windows": windows,
    }
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()

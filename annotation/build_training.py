"""
Build fine-tune training data from annotation JSONs.

Reads all *_annotation.json files, constructs (explanation, prediction) pairs
using temporal offset, splits by video, and outputs LLaMA-Factory format JSONs.

Usage:
    python -m annotation.build_training \
        --annotations data/annotations/ \
        --output data/training/ \
        --n_frames 4 \
        --pred_horizon 1 \
        --test_ratio 0.15 \
        --val_ratio 0.15
"""

import argparse
import json
import os
import random
import sys
from typing import List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    parser = argparse.ArgumentParser(description="Build fine-tune training data from annotations")
    parser.add_argument("--annotations", default="data/annotations/", help="Directory containing annotation JSONs")
    parser.add_argument("--output", default="data/training/", help="Output directory for training files")
    parser.add_argument("--n_frames", type=int, default=4, help="Number of frames to select per window (from 5)")
    parser.add_argument("--pred_horizon", type=int, default=1, help="Predict h windows ahead")
    parser.add_argument("--test_ratio", type=float, default=0.15, help="Fraction of videos for test set")
    parser.add_argument("--val_ratio", type=float, default=0.15, help="Fraction of videos for validation set")
    parser.add_argument("--frame_base", default="", help="Base path to prepend to frame paths (optional)")
    parser.add_argument("--with_history", action="store_true", help="Include history context in prompt (for V5)")
    parser.add_argument("--history_steps", type=int, default=3, help="Number of past windows as history (default: 3)")
    args = parser.parse_args()

    # --- Step 1: Load all annotation files ---
    annotation_files = sorted([
        f for f in os.listdir(args.annotations)
        if f.endswith("_annotation.json")
    ])
    if not annotation_files:
        print(f"No annotation files found in {args.annotations}")
        return

    print(f"Found {len(annotation_files)} annotation files.")

    all_annotations = {}
    for fname in annotation_files:
        path = os.path.join(args.annotations, fname)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        video_uid = data["video_uid"]
        all_annotations[video_uid] = data

    # --- Step 2: Split videos into train/val/test ---
    video_uids = sorted(all_annotations.keys())
    random.seed(42)
    random.shuffle(video_uids)

    n_test = int(len(video_uids) * args.test_ratio)
    n_val = int(len(video_uids) * args.val_ratio)

    test_videos = video_uids[:n_test]
    val_videos = video_uids[n_test:n_test + n_val]
    train_videos = video_uids[n_test + n_val:]

    print(f"Split: {len(train_videos)} train, {len(val_videos)} val, {len(test_videos)} test videos.")

    # --- Step 3: Generate training samples ---
    train_samples = []
    val_samples = []
    test_samples = []

    for video_uid, data in all_annotations.items():
        windows = data["windows"]
        samples = _build_samples_for_video(
            windows, args.n_frames, args.pred_horizon, args.frame_base,
            args.with_history, args.history_steps,
        )

        if video_uid in test_videos:
            test_samples.extend(samples)
        elif video_uid in val_videos:
            val_samples.extend(samples)
        else:
            train_samples.extend(samples)

    # --- Step 5: Save outputs ---
    os.makedirs(args.output, exist_ok=True)

    _save_json(os.path.join(args.output, "train.json"), train_samples)
    _save_json(os.path.join(args.output, "val.json"), val_samples)
    _save_json(os.path.join(args.output, "test.json"), test_samples)

    # split_info.json
    split_info = {
        "n_frames": args.n_frames,
        "pred_horizon": args.pred_horizon,
        "total_videos": len(video_uids),
        "total_samples": len(train_samples) + len(val_samples) + len(test_samples),
        "train_videos": train_videos,
        "val_videos": val_videos,
        "test_videos": test_videos,
        "train_samples": len(train_samples),
        "val_samples": len(val_samples),
        "test_samples": len(test_samples),
    }
    _save_json(os.path.join(args.output, "split_info.json"), split_info)

    # config.json
    config = {
        "n_frames": args.n_frames,
        "pred_horizon": args.pred_horizon,
        "test_ratio": args.test_ratio,
        "val_ratio": args.val_ratio,
        "frame_base": args.frame_base,
        "with_history": args.with_history,
        "history_steps": args.history_steps if args.with_history else 0,
        "annotations_dir": os.path.abspath(args.annotations),
        "random_seed": 42,
    }
    _save_json(os.path.join(args.output, "config.json"), config)

    # --- Summary ---
    total = len(train_samples) + len(val_samples) + len(test_samples)
    print(f"\n=== Training Data Summary ===")
    print(f"Total videos: {len(video_uids)}")
    print(f"Total samples: {total}")
    print(f"  Train: {len(train_samples)} samples ({len(train_videos)} videos)")
    print(f"  Val:   {len(val_samples)} samples ({len(val_videos)} videos)")
    print(f"  Test:  {len(test_samples)} samples ({len(test_videos)} videos)")
    if video_uids:
        print(f"  Avg samples/video: {total / len(video_uids):.1f}")
    print(f"\nOutput saved to {os.path.abspath(args.output)}")


# ─── Helpers ───────────────────────────────────────────────────────────

def _build_samples_for_video(
    windows: List[dict],
    n_frames: int,
    pred_horizon: int,
    frame_base: str,
    with_history: bool = False,
    history_steps: int = 3,
) -> List[dict]:
    """Build training samples for one video's annotation windows."""
    samples = []
    total = len(windows)

    for t in range(total - pred_horizon):
        w_current = windows[t]
        w_future = windows[t + pred_horizon]

        # Skip error windows
        if w_current.get("annotation_status") != "ok":
            continue
        if w_future.get("annotation_status") != "ok":
            continue

        explanation = w_current["explanation"]
        prediction = w_future["explanation"]

        # Select frames
        frame_paths = w_current["frame_paths"]
        selected = _select_frames(frame_paths, n_frames)

        if frame_base:
            selected = [os.path.join(frame_base, p) for p in selected]

        # Build prompt — with or without history
        if with_history:
            history_text = _build_history_context(windows, t, history_steps)
            prompt = _build_inference_prompt(n_frames, history_text)
        else:
            prompt = _build_inference_prompt(n_frames)

        sample = {
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                },
                {
                    "role": "assistant",
                    "content": f"Explanation: {explanation}\nPrediction: {prediction}",
                },
            ],
            "images": selected,
        }
        samples.append(sample)

    return samples


def _select_frames(frame_paths: List[str], n: int) -> List[str]:
    """
    Select n frames evenly spaced from the available frame_paths.

    Examples (from 5 frames):
        n=5: [0, 1, 2, 3, 4]
        n=4: [0, 1, 3, 4]
        n=3: [0, 2, 4]
        n=2: [0, 4]
        n=1: [2]
    """
    total = len(frame_paths)
    if n >= total:
        return list(frame_paths)
    if n == 1:
        return [frame_paths[total // 2]]

    indices = [round(i * (total - 1) / (n - 1)) for i in range(n)]
    return [frame_paths[i] for i in indices]


def _build_history_context(windows: List[dict], current_idx: int, history_steps: int) -> str:
    """Build history context string from previous windows' explanations."""
    entries = []
    start = max(0, current_idx - history_steps)
    for i in range(start, current_idx):
        w = windows[i]
        if w.get("annotation_status") != "ok":
            continue
        offset = current_idx - i
        entries.append(f"- Step t-{offset}: {w['explanation']}")

    if not entries:
        return ""
    return "\n".join(entries)


def _build_inference_prompt(n_frames: int, history_text: str = "") -> str:
    """Build the inference prompt with correct <image> tokens and optional history."""
    image_tokens = "<image>" * n_frames

    parts = [
        f"{image_tokens}\n"
        "You are analyzing consecutive frames from a first-person (egocentric) video, "
        "shown in chronological order at regular intervals.\n\n"
    ]

    if history_text:
        parts.append(
            "Recent context from previous time steps:\n"
            f"{history_text}\n\n"
            "Your analysis of the CURRENT FRAMES is the primary basis for your response. "
            "Use the above context only as supplementary reference.\n\n"
        )

    parts.append(
        "Task: Provide an Explanation of the current action and a Prediction of the next action.\n\n"
        "Rules:\n"
        "- Describe action progression across frames (first... then... finally...), not a single summary\n"
        "- Name specific objects (e.g. \"white cup\", \"wooden board\"), never use vague words like \"items\" or \"objects\"\n"
        "- Include hand details: which hand does what\n"
        "- Start with action verbs, use dynamic verbs (reaches, grasps, lifts, places, stirs)\n"
        "- Do NOT start with \"The person is currently...\"\n"
        "- Do NOT reference the frames themselves (no \"as seen in the frames\", \"across the images\")\n"
        "- No vague hedging (\"possibly\", \"might\", \"could potentially\")\n"
        "- Prediction: one clear next action with brief reasoning linking to current action trend\n"
        "- Each section: 30-50 words\n\n"
        "Now analyze the provided frames. Respond with exactly two lines:\n"
        "Explanation: <your explanation>\n"
        "Prediction: <your prediction>"
    )

    return "".join(parts)


def _save_json(path: str, data) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()

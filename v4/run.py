"""
V4 Action Prediction: 4-frame window, fine-tuned Qwen3.5-9B (no history).

Usage (on GPU server):
    cd /root/sdp
    python -m v4.run --video data/videos/<video>.mp4 --output data/results/

    # With custom adapter path:
    python -m v4.run --video data/videos/<video>.mp4 --adapter /path/to/adapter
"""

import argparse
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.video_frames import extract_frames
from shared.qwen_client import init_model, call_vlm
from shared.utils import save_results
from v4.prompt import build_prompt

WINDOW_SIZE = 4


def main():
    parser = argparse.ArgumentParser(description="V4: Fine-tuned 4-frame window action prediction")
    parser.add_argument("--video", required=True, help="Path to input video (.mp4)")
    parser.add_argument("--output", default="data/results/", help="Output directory")
    parser.add_argument("--interval", type=float, default=0.5, help="Frame extraction interval (seconds)")
    parser.add_argument("--model", default="/root/sdp/models/qwen3.5-9b", help="Base model path")
    parser.add_argument("--adapter", default="/root/sdp/outputs/v4/lora_adapter", help="LoRA adapter path")
    args = parser.parse_args()

    # Initialize fine-tuned model
    init_model(model_path=args.model, adapter_path=args.adapter)

    video_path = os.path.abspath(args.video)
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    # Extract frames
    print(f"Extracting frames from {video_path} (interval={args.interval}s)...")
    frames_dir = os.path.join("data", "frames", video_name)
    frame_list = extract_frames(video_path, frames_dir, interval=args.interval)
    print(f"Extracted {len(frame_list)} frames.")

    # Group into non-overlapping 4-frame windows
    windows = [
        frame_list[i : i + WINDOW_SIZE]
        for i in range(0, len(frame_list), WINDOW_SIZE)
        if i + WINDOW_SIZE <= len(frame_list)
    ]
    print(f"Created {len(windows)} windows of {WINDOW_SIZE} frames each.")

    # Build prompt (same as V2)
    prompt = build_prompt()

    # Process each window
    predictions = []
    total = len(windows)

    for i, window in enumerate(windows):
        print(f"Processing window {i + 1}/{total}...")
        img_paths = [f["image_path"] for f in window]
        t_start = window[0]["timestamp"]
        t_end = window[-1]["timestamp"]

        try:
            response = call_vlm(img_paths, prompt)
        except Exception as e:
            print(f"  Skipping window {i + 1}: {e}")
            response = f"ERROR: {e}"

        explanation, prediction = _parse_response(response)

        predictions.append({
            "window_id": i + 1,
            "time_range": [t_start, t_end],
            "frames": [os.path.basename(f["image_path"]) for f in window],
            "explanation": explanation,
            "prediction": prediction,
            "raw_response": response,
        })

    # Save results
    results = {
        "video_path": video_path,
        "version": "V4",
        "frame_interval": args.interval,
        "window_size": WINDOW_SIZE,
        "adapter": args.adapter,
        "predictions": predictions,
    }

    output_file = os.path.join(args.output, f"{video_name}_v4.json")
    save_results(results, output_file)
    print(f"Done. {len(predictions)} predictions saved.")


def _parse_response(response: str) -> tuple:
    """Extract explanation and prediction from VLM response."""
    text = response.strip()

    exp_match = re.search(r"(?i)explanation:\s*", text)
    pred_match = re.search(r"(?i)prediction:\s*", text)

    if exp_match and pred_match:
        exp_start = exp_match.end()
        pred_start = pred_match.end()

        if exp_match.start() < pred_match.start():
            explanation = text[exp_start:pred_match.start()].strip()
            prediction = text[pred_start:].strip()
        else:
            prediction = text[pred_start:exp_match.start()].strip()
            explanation = text[exp_start:].strip()
    elif exp_match:
        explanation = text[exp_match.end():].strip()
        prediction = explanation
    elif pred_match:
        prediction = text[pred_match.end():].strip()
        explanation = prediction
    else:
        explanation = text
        prediction = text

    return explanation, prediction


if __name__ == "__main__":
    main()

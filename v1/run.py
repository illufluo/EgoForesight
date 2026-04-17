"""
V1 Action Prediction: single frame → prediction.

Usage:
    python -m v1.run --video data/videos/example.mp4 --output data/results/
"""

import argparse
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.video_frames import extract_frames
from shared.vlm import get_call_vlm
from shared.utils import save_results
from v1.prompt import build_prompt


def main():
    parser = argparse.ArgumentParser(description="V1: Single-frame action prediction")
    parser.add_argument("--video", required=True, help="Path to input video (.mp4)")
    parser.add_argument("--output", default="data/results/", help="Output directory")
    parser.add_argument("--interval", type=float, default=0.5, help="Frame extraction interval (seconds)")
    parser.add_argument("--backend", default="glm", choices=["glm", "qwen"], help="VLM backend")
    args = parser.parse_args()

    call_vlm = get_call_vlm(args.backend)

    video_path = os.path.abspath(args.video)
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    # Extract frames
    print(f"Extracting frames from {video_path} (interval={args.interval}s)...")
    frames_dir = os.path.join("data", "frames", video_name)
    frame_list = extract_frames(video_path, frames_dir, interval=args.interval)
    print(f"Extracted {len(frame_list)} frames.")

    # Build prompt
    prompt = build_prompt()

    # Process each frame
    predictions = []
    total = len(frame_list)

    for i, frame_info in enumerate(frame_list):
        print(f"Processing frame {i + 1}/{total}...")
        img_path = frame_info["image_path"]
        ts = frame_info["timestamp"]

        try:
            response = call_vlm([img_path], prompt)
        except Exception as e:
            print(f"  Skipping frame {i + 1}: {e}")
            response = f"ERROR: {e}"

        explanation, prediction = _parse_response(response)

        predictions.append({
            "window_id": i + 1,
            "time_range": [ts, ts],
            "frames": [os.path.basename(img_path)],
            "explanation": explanation,
            "prediction": prediction,
            "raw_response": response,
        })

    # Save results
    results = {
        "video_path": video_path,
        "version": "V1",
        "frame_interval": args.interval,
        "predictions": predictions,
    }

    output_file = os.path.join(args.output, f"{video_name}_v1.json")
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

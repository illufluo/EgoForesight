"""
Evaluate prediction quality with BERTScore against Ego4D narrations as GT.

For each prediction window [t_start, t_end], find the narration closest to
(t_end + offset) within ±tolerance seconds as the ground-truth "next action",
then score prediction against GT with BERTScore F1.

Dependencies:
    pip install bert-score
    # (also requires a network connection the first time to download the
    # microsoft/deberta-xlarge-mnli model, or a pre-cached HuggingFace cache)

Usage:
    python -m tools.evaluate --versions v1 v2 v3 v4 v5
    python -m tools.evaluate --versions v4 v5 --results_dir data/results
"""

import argparse
import json
import os
import sys
from typing import List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.utils import load_narrations


def _load_test_videos(split_info_path: str) -> List[str]:
    with open(split_info_path) as f:
        info = json.load(f)
    return info.get("test_videos", [])


def _group_narrations_by_video(narrations: List[dict]) -> dict:
    """Return {video_uid: [(timestamp_sec, text), ...]} sorted by timestamp."""
    by_video = {}
    for row in narrations:
        uid = row.get("video_uid", "")
        text = (row.get("narration_text") or "").strip()
        try:
            ts = float(row.get("timestamp_sec", ""))
        except (TypeError, ValueError):
            continue
        if not uid or not text:
            continue
        by_video.setdefault(uid, []).append((ts, text))
    for uid in by_video:
        by_video[uid].sort(key=lambda x: x[0])
    return by_video


def _find_gt(narrations: List[tuple], target_time: float, tolerance: float) -> Optional[str]:
    """Find the narration whose timestamp is closest to target_time within ±tolerance."""
    best_text = None
    best_diff = float("inf")
    for ts, text in narrations:
        diff = abs(ts - target_time)
        if diff <= tolerance and diff < best_diff:
            best_diff = diff
            best_text = text
    return best_text


def _align_video(
    result_path: str,
    narrations_by_video: dict,
    video_uid: str,
    offset: float,
    tolerance: float,
) -> List[dict]:
    """Return list of alignment dicts with prediction, gt, timestamps, window_id."""
    with open(result_path) as f:
        result = json.load(f)

    video_narrations = narrations_by_video.get(video_uid, [])
    if not video_narrations:
        return []

    pairs = []
    for p in result.get("predictions", []):
        prediction = (p.get("prediction") or "").strip()
        if not prediction or prediction.startswith("ERROR:"):
            continue
        tr = p.get("time_range") or [None, None]
        if len(tr) < 2 or tr[1] is None:
            continue
        t_end = float(tr[1])
        target = t_end + offset
        gt = _find_gt(video_narrations, target, tolerance)
        if not gt:
            continue
        pairs.append({
            "video_uid": video_uid,
            "window_id": p.get("window_id"),
            "time_range": tr,
            "target_time": target,
            "prediction": prediction,
            "ground_truth": gt,
        })
    return pairs


def _score_pairs(pairs: List[dict], model_type: str, batch_size: int) -> List[dict]:
    """Run BERTScore in one batch; return pairs with P/R/F1 fields added."""
    if not pairs:
        return []

    import bert_score

    cands = [p["prediction"] for p in pairs]
    refs = [p["ground_truth"] for p in pairs]

    print(f"  Running BERTScore on {len(pairs)} pairs (model={model_type})...")
    P, R, F1 = bert_score.score(
        cands,
        refs,
        model_type=model_type,
        lang="en",
        batch_size=batch_size,
        verbose=False,
        rescale_with_baseline=False,
    )

    scored = []
    for p, pr, rc, f1 in zip(pairs, P.tolist(), R.tolist(), F1.tolist()):
        p_out = dict(p)
        p_out["bertscore_precision"] = pr
        p_out["bertscore_recall"] = rc
        p_out["bertscore_f1"] = f1
        scored.append(p_out)
    return scored


def evaluate_version(
    version: str,
    test_videos: List[str],
    narrations_by_video: dict,
    results_dir: str,
    output_dir: str,
    model_type: str,
    offset: float,
    tolerance: float,
    batch_size: int,
) -> dict:
    print(f"\n=== Evaluating {version} ===")

    all_pairs = []
    missing = []
    for i, uid in enumerate(test_videos, 1):
        fname = f"{uid}_{version}.json"
        result_path = os.path.join(results_dir, fname)
        if not os.path.exists(result_path):
            print(f"  [{i}/{len(test_videos)}] {uid}: missing {fname}, skipping")
            missing.append(uid)
            continue

        pairs = _align_video(result_path, narrations_by_video, uid, offset, tolerance)
        print(f"  [{i}/{len(test_videos)}] {uid}: {len(pairs)} aligned pairs")
        all_pairs.extend(pairs)

    if not all_pairs:
        print(f"  No aligned pairs for {version} — skipping BERTScore.")
        summary = {
            "version": version,
            "n_pairs": 0,
            "avg_f1": None,
            "avg_precision": None,
            "avg_recall": None,
            "videos_evaluated": len(test_videos) - len(missing),
            "videos_missing": missing,
        }
        return {"summary": summary, "pairs": []}

    scored = _score_pairs(all_pairs, model_type, batch_size)

    n = len(scored)
    avg_f1 = sum(p["bertscore_f1"] for p in scored) / n
    avg_p = sum(p["bertscore_precision"] for p in scored) / n
    avg_r = sum(p["bertscore_recall"] for p in scored) / n

    summary = {
        "version": version,
        "n_pairs": n,
        "avg_f1": avg_f1,
        "avg_precision": avg_p,
        "avg_recall": avg_r,
        "videos_evaluated": len(test_videos) - len(missing),
        "videos_missing": missing,
        "model_type": model_type,
        "offset": offset,
        "tolerance": tolerance,
    }

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"bertscore_{version}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "pairs": scored}, f, ensure_ascii=False, indent=2)
    print(f"  Saved detailed results to {out_path}")

    return {"summary": summary, "pairs": scored}


def _print_summary_table(summaries: List[dict]) -> None:
    print("\n" + "=" * 70)
    print(f"{'Version':<10}{'#Pairs':>10}{'Avg F1':>12}{'Avg P':>12}{'Avg R':>12}")
    print("-" * 70)
    for s in summaries:
        f1 = f"{s['avg_f1']:.4f}" if s["avg_f1"] is not None else "N/A"
        pr = f"{s['avg_precision']:.4f}" if s["avg_precision"] is not None else "N/A"
        rc = f"{s['avg_recall']:.4f}" if s["avg_recall"] is not None else "N/A"
        print(f"{s['version']:<10}{s['n_pairs']:>10}{f1:>12}{pr:>12}{rc:>12}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="BERTScore evaluation for V1-V5 predictions")
    parser.add_argument("--versions", nargs="+", default=["v1", "v2", "v3", "v4", "v5"],
                        help="Versions to evaluate")
    parser.add_argument("--results_dir", default="data/results", help="Directory of result JSONs")
    parser.add_argument("--narration", default="data/narrations/selected_narrations.csv",
                        help="Narration CSV path")
    parser.add_argument("--split_info", default="data/training_v4/split_info.json",
                        help="split_info.json providing test_videos list")
    parser.add_argument("--video_list", nargs="+", default=None,
                        help="Explicit video uids (overrides split_info)")
    parser.add_argument("--output_dir", default="data/evaluation", help="Output directory")
    parser.add_argument("--model_type", default="microsoft/deberta-xlarge-mnli",
                        help="BERTScore model")
    parser.add_argument("--offset", type=float, default=1.0,
                        help="Target = t_end + offset (seconds)")
    parser.add_argument("--tolerance", type=float, default=2.0,
                        help="Max |narration_ts - target| (seconds)")
    parser.add_argument("--batch_size", type=int, default=32, help="BERTScore batch size")
    args = parser.parse_args()

    # Resolve test videos
    if args.video_list:
        test_videos = args.video_list
    else:
        test_videos = _load_test_videos(args.split_info)
    print(f"Evaluating {len(test_videos)} test videos: {test_videos}")

    # Load narrations once
    print(f"Loading narrations from {args.narration}...")
    narrations = load_narrations(args.narration)
    narrations_by_video = _group_narrations_by_video(narrations)
    print(f"Loaded {len(narrations)} narrations across {len(narrations_by_video)} videos.")

    # Evaluate each version
    summaries = []
    for version in args.versions:
        result = evaluate_version(
            version=version,
            test_videos=test_videos,
            narrations_by_video=narrations_by_video,
            results_dir=args.results_dir,
            output_dir=args.output_dir,
            model_type=args.model_type,
            offset=args.offset,
            tolerance=args.tolerance,
            batch_size=args.batch_size,
        )
        summaries.append(result["summary"])

    _print_summary_table(summaries)

    # Save combined summary
    os.makedirs(args.output_dir, exist_ok=True)
    combined_path = os.path.join(args.output_dir, "bertscore_summary.json")
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump({"summaries": summaries}, f, ensure_ascii=False, indent=2)
    print(f"\nCombined summary saved to {combined_path}")


if __name__ == "__main__":
    main()

"""
Evaluate prediction quality against Ego4D narrations as GT.

Metrics:
  - bertscore: BERTScore P/R/F1 (pip install bert-score)
  - semantic:  Cosine similarity via sentence-transformers (pip install sentence-transformers)

For each prediction window [t_start, t_end], find the narration closest to
(t_end + offset) within ±tolerance seconds as the ground-truth "next action".

Usage:
    # BERTScore only (default):
    python -m tools.evaluate --versions v1 v2 v3

    # Semantic similarity only:
    python -m tools.evaluate --versions v1 v2 --metrics semantic

    # Both metrics:
    python -m tools.evaluate --metrics bertscore semantic

    # Multi-directory mode:
    python -m tools.evaluate \
      --version_dirs \
        glm_v1:report_data/glm_v1:v1 \
        v4_run1:report_data/results_run1:v4 \
      --metrics bertscore semantic \
      --model_type roberta-large
"""

import argparse
import json
import os
import sys
from typing import List, Optional, Set

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


# ─── Scoring functions ────────────────────────


def _bertscore_pairs(pairs: List[dict], model_type: str, batch_size: int) -> List[dict]:
    """Run BERTScore in one batch; return pairs with P/R/F1 fields added."""
    if not pairs:
        return pairs

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

    for p, pr, rc, f1 in zip(pairs, P.tolist(), R.tolist(), F1.tolist()):
        p["bertscore_precision"] = pr
        p["bertscore_recall"] = rc
        p["bertscore_f1"] = f1
    return pairs


def _semantic_score_pairs(pairs: List[dict], model_name: str, batch_size: int) -> List[dict]:
    """Compute cosine similarity via sentence-transformers."""
    if not pairs:
        return pairs

    from sentence_transformers import SentenceTransformer, util

    model = SentenceTransformer(model_name)

    cands = [p["prediction"] for p in pairs]
    refs = [p["ground_truth"] for p in pairs]

    print(f"  Running Semantic Similarity on {len(pairs)} pairs ({model_name})...")
    cand_embeddings = model.encode(cands, batch_size=batch_size, show_progress_bar=False)
    ref_embeddings = model.encode(refs, batch_size=batch_size, show_progress_bar=False)

    sims = util.cos_sim(cand_embeddings, ref_embeddings)
    for i, p in enumerate(pairs):
        p["semantic_sim"] = sims[i][i].item()
    return pairs


# ─── Main evaluation logic ────────────────────


def evaluate_version(
    version: str,
    test_videos: List[str],
    narrations_by_video: dict,
    results_dir: str,
    file_suffix: str,
    output_dir: str,
    metrics: Set[str],
    model_type: str,
    semantic_model: str,
    offset: float,
    tolerance: float,
    batch_size: int,
) -> dict:
    """Evaluate one version. file_suffix is the vN part in {uid}_{suffix}.json."""
    print(f"\n=== Evaluating {version} (dir={results_dir}, suffix={file_suffix}) ===")

    all_pairs = []
    missing = []
    for i, uid in enumerate(test_videos, 1):
        fname = f"{uid}_{file_suffix}.json"
        result_path = os.path.join(results_dir, fname)
        if not os.path.exists(result_path):
            print(f"  [{i}/{len(test_videos)}] {uid}: missing {fname}, skipping")
            missing.append(uid)
            continue

        pairs = _align_video(result_path, narrations_by_video, uid, offset, tolerance)
        print(f"  [{i}/{len(test_videos)}] {uid}: {len(pairs)} aligned pairs")
        all_pairs.extend(pairs)

    if not all_pairs:
        print(f"  No aligned pairs for {version} — skipping scoring.")
        summary = {
            "version": version,
            "n_pairs": 0,
            "videos_evaluated": len(test_videos) - len(missing),
            "videos_missing": missing,
        }
        return {"summary": summary, "pairs": []}

    # Run selected metrics
    if "bertscore" in metrics:
        _bertscore_pairs(all_pairs, model_type, batch_size)
    if "semantic" in metrics:
        _semantic_score_pairs(all_pairs, semantic_model, batch_size)

    # Build summary
    n = len(all_pairs)
    summary = {
        "version": version,
        "n_pairs": n,
        "videos_evaluated": len(test_videos) - len(missing),
        "videos_missing": missing,
        "offset": offset,
        "tolerance": tolerance,
    }

    if "bertscore" in metrics:
        summary["avg_f1"] = sum(p["bertscore_f1"] for p in all_pairs) / n
        summary["avg_precision"] = sum(p["bertscore_precision"] for p in all_pairs) / n
        summary["avg_recall"] = sum(p["bertscore_recall"] for p in all_pairs) / n
        summary["bertscore_model"] = model_type

    if "semantic" in metrics:
        summary["avg_semantic_sim"] = sum(p["semantic_sim"] for p in all_pairs) / n
        summary["semantic_model"] = semantic_model

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"eval_{version}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "pairs": all_pairs}, f, ensure_ascii=False, indent=2)
    print(f"  Saved detailed results to {out_path}")

    return {"summary": summary, "pairs": all_pairs}


def _print_summary_table(summaries: List[dict], metrics: Set[str]) -> None:
    max_name = max((len(s["version"]) for s in summaries), default=7)
    col = max(max_name + 2, 12)

    # Build header
    header = f"{'Version':<{col}}{'#Pairs':>10}"
    if "bertscore" in metrics:
        header += f"{'Avg F1':>12}{'Avg P':>12}{'Avg R':>12}"
    if "semantic" in metrics:
        header += f"{'Sem Sim':>12}"
    width = len(header)

    print("\n" + "=" * width)
    print(header)
    print("-" * width)
    for s in summaries:
        line = f"{s['version']:<{col}}{s['n_pairs']:>10}"
        if "bertscore" in metrics:
            f1 = f"{s['avg_f1']:.4f}" if s.get("avg_f1") is not None else "N/A"
            pr = f"{s['avg_precision']:.4f}" if s.get("avg_precision") is not None else "N/A"
            rc = f"{s['avg_recall']:.4f}" if s.get("avg_recall") is not None else "N/A"
            line += f"{f1:>12}{pr:>12}{rc:>12}"
        if "semantic" in metrics:
            sm = f"{s['avg_semantic_sim']:.4f}" if s.get("avg_semantic_sim") is not None else "N/A"
            line += f"{sm:>12}"
        print(line)
    print("=" * width)


def _parse_version_dir(spec: str) -> dict:
    """Parse 'version_name:directory:file_suffix' into a dict."""
    parts = spec.split(":")
    if len(parts) != 3:
        raise ValueError(
            f"Invalid --version_dirs format: '{spec}'. "
            "Expected 'version_name:directory:file_suffix'"
        )
    return {
        "version": parts[0],
        "results_dir": os.path.expanduser(parts[1]),
        "file_suffix": parts[2],
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate action predictions against GT narrations")
    parser.add_argument("--versions", nargs="+", default=None,
                        help="Versions to evaluate (simple mode: file suffix = version name)")
    parser.add_argument("--results_dir", default="data/results",
                        help="Directory of result JSONs (used with --versions)")
    parser.add_argument("--version_dirs", nargs="+", default=None,
                        help="Multi-dir mode: version_name:directory:file_suffix entries")
    parser.add_argument("--metrics", nargs="+", default=["bertscore"],
                        choices=["bertscore", "semantic"],
                        help="Metrics to compute (default: bertscore)")
    parser.add_argument("--narration", default="data/narrations/selected_narrations.csv",
                        help="Narration CSV path")
    parser.add_argument("--split_info", default="data/training_v4/split_info.json",
                        help="split_info.json providing test_videos list")
    parser.add_argument("--video_list", nargs="+", default=None,
                        help="Explicit video uids (overrides split_info)")
    parser.add_argument("--output_dir", default="data/evaluation", help="Output directory")
    parser.add_argument("--model_type", default="microsoft/deberta-xlarge-mnli",
                        help="BERTScore model")
    parser.add_argument("--semantic_model", default="all-MiniLM-L6-v2",
                        help="Sentence-transformers model for semantic similarity")
    parser.add_argument("--offset", type=float, default=1.0,
                        help="Target = t_end + offset (seconds)")
    parser.add_argument("--tolerance", type=float, default=2.0,
                        help="Max |narration_ts - target| (seconds)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for scoring")
    args = parser.parse_args()

    metrics = set(args.metrics)

    # Build evaluation specs
    eval_specs = []
    if args.version_dirs:
        for spec in args.version_dirs:
            eval_specs.append(_parse_version_dir(spec))
    elif args.versions:
        for v in args.versions:
            eval_specs.append({
                "version": v,
                "results_dir": args.results_dir,
                "file_suffix": v,
            })
    else:
        for v in ["v1", "v2", "v3", "v4", "v5"]:
            eval_specs.append({
                "version": v,
                "results_dir": args.results_dir,
                "file_suffix": v,
            })

    # Resolve test videos
    if args.video_list:
        test_videos = args.video_list
    else:
        test_videos = _load_test_videos(args.split_info)
    print(f"Evaluating {len(test_videos)} test videos")
    print(f"Versions: {[s['version'] for s in eval_specs]}")
    print(f"Metrics: {sorted(metrics)}")

    # Load narrations once
    print(f"Loading narrations from {args.narration}...")
    narrations = load_narrations(args.narration)
    narrations_by_video = _group_narrations_by_video(narrations)
    print(f"Loaded {len(narrations)} narrations across {len(narrations_by_video)} videos.")

    output_dir = os.path.expanduser(args.output_dir)

    # Evaluate each version
    summaries = []
    for spec in eval_specs:
        result = evaluate_version(
            version=spec["version"],
            test_videos=test_videos,
            narrations_by_video=narrations_by_video,
            results_dir=spec["results_dir"],
            file_suffix=spec["file_suffix"],
            output_dir=output_dir,
            metrics=metrics,
            model_type=args.model_type,
            semantic_model=args.semantic_model,
            offset=args.offset,
            tolerance=args.tolerance,
            batch_size=args.batch_size,
        )
        summaries.append(result["summary"])

    _print_summary_table(summaries, metrics)

    # Save combined summary
    os.makedirs(output_dir, exist_ok=True)
    combined_path = os.path.join(output_dir, "eval_summary.json")
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump({"metrics": sorted(metrics), "summaries": summaries}, f, ensure_ascii=False, indent=2)
    print(f"\nCombined summary saved to {combined_path}")


if __name__ == "__main__":
    main()

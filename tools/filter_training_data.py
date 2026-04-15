"""
Filter training data: drop samples where Prediction substantially repeats Explanation.

Computes Jaccard similarity over tokenized words between the Explanation and
Prediction sections of each sample's assistant content. Samples with similarity
above the threshold (default 0.5) are discarded to reduce overfitting on
repetitive outputs.

Usage:
    python -m tools.filter_training_data
    python -m tools.filter_training_data --threshold 0.4
"""

import argparse
import json
import os
import re
import shutil
import string


STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "of", "in", "on", "at", "to", "for",
    "with", "by", "from", "as", "is", "are", "was", "were", "be", "been",
    "being", "this", "that", "these", "those", "it", "its", "he", "she", "they",
    "them", "his", "her", "their", "then", "also", "which", "who", "while",
    "into", "onto", "over", "under", "up", "down", "off", "out", "will", "would",
    "can", "could", "should", "may", "might", "has", "have", "had", "do", "does",
    "did", "not", "no", "so", "there", "here",
}


def _split_explanation_prediction(text: str) -> tuple:
    """
    Extract Explanation and Prediction sections from assistant content.
    Returns (explanation, prediction) as strings. Empty string if not found.
    """
    exp_match = re.search(r"(?i)explanation:\s*", text)
    pred_match = re.search(r"(?i)prediction:\s*", text)

    if not (exp_match and pred_match):
        return "", ""

    if exp_match.start() < pred_match.start():
        explanation = text[exp_match.end():pred_match.start()].strip()
        prediction = text[pred_match.end():].strip()
    else:
        prediction = text[pred_match.end():exp_match.start()].strip()
        explanation = text[exp_match.end():].strip()

    return explanation, prediction


def _tokenize(text: str) -> set:
    """Lowercase, strip punctuation, split on whitespace, drop stopwords."""
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    tokens = {w for w in text.split() if w and w not in STOPWORDS}
    return tokens


def jaccard_similarity(a: str, b: str) -> float:
    """Jaccard similarity of tokenized word sets. Returns 0.0 if either is empty."""
    sa = _tokenize(a)
    sb = _tokenize(b)
    if not sa or not sb:
        return 0.0
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union if union else 0.0


def filter_split(
    input_path: str,
    output_path: str,
    threshold: float,
) -> tuple:
    """Filter train.json, writing kept samples to output_path. Returns (kept, dropped)."""
    with open(input_path) as f:
        samples = json.load(f)

    kept = []
    dropped = 0
    for sample in samples:
        assistant_text = sample["messages"][1]["content"]
        explanation, prediction = _split_explanation_prediction(assistant_text)
        sim = jaccard_similarity(explanation, prediction)
        if sim > threshold:
            dropped += 1
            continue
        kept.append(sample)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(kept, f, ensure_ascii=False, indent=2)

    return len(kept), dropped


def copy_other_splits(src_dir: str, dst_dir: str) -> None:
    """Copy val.json, test.json, split_info.json, config.json unchanged if present."""
    os.makedirs(dst_dir, exist_ok=True)
    for name in ("val.json", "test.json", "split_info.json", "config.json"):
        src = os.path.join(src_dir, name)
        dst = os.path.join(dst_dir, name)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f"  Copied {name}")


def process_version(version: str, threshold: float, base_dir: str) -> None:
    """Filter training data for one version (v4 or v5)."""
    src_dir = os.path.join(base_dir, f"training_{version}")
    dst_dir = os.path.join(base_dir, f"training_{version}_filtered")

    print(f"\n=== Filtering {src_dir} → {dst_dir} ===")

    train_src = os.path.join(src_dir, "train.json")
    train_dst = os.path.join(dst_dir, "train.json")

    with open(train_src) as f:
        total_before = len(json.load(f))

    kept, dropped = filter_split(train_src, train_dst, threshold)
    print(f"  Before: {total_before} samples")
    print(f"  Kept:   {kept} samples")
    print(f"  Dropped: {dropped} samples (Jaccard > {threshold})")

    copy_other_splits(src_dir, dst_dir)


def main():
    parser = argparse.ArgumentParser(description="Filter training data by Explanation/Prediction overlap")
    parser.add_argument("--data_dir", default="data", help="Base data directory")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Drop samples where Jaccard(Explanation, Prediction) > threshold")
    parser.add_argument("--versions", nargs="+", default=["v4", "v5"],
                        help="Which versions to filter (v4, v5)")
    args = parser.parse_args()

    for version in args.versions:
        process_version(version, args.threshold, args.data_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()

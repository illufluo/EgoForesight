"""
Shared utilities: result saving and narration loading.
"""

import csv
import json
import os
import re
from typing import List


def save_results(results: dict, output_path: str) -> None:
    """Save results dict to a JSON file."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Results saved to {output_path}")


def load_narrations(csv_path: str) -> List[dict]:
    """
    Load narration CSV and clean the '#C C' prefix from narration_text.

    Returns:
        List of dicts with keys: video_uid, pass, timestamp_sec,
        timestamp_frame, narration_text, annotation_uid
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Narration CSV not found: {csv_path}")

    narrations = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Clean '#C C ' prefix from narration text
            text = row.get("narration_text", "")
            text = re.sub(r"^#C\s+C\s+", "", text)
            row["narration_text"] = text
            narrations.append(row)

    return narrations

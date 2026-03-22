"""V3 history manager: sliding window of past explanations and predictions."""

from collections import deque


# Labels from oldest to newest for up to 3 steps
_LABELS = {
    1: ["Previous step"],
    2: ["2 steps ago", "Previous step"],
    3: ["3 steps ago", "2 steps ago", "Previous step"],
}


class HistoryManager:
    """Maintains a sliding window of the last 3 explanation/prediction pairs."""

    def __init__(self, max_steps: int = 3):
        self._max_steps = max_steps
        self._history = deque(maxlen=max_steps)

    def add(self, explanation: str, prediction: str) -> None:
        """Store a full explanation and prediction from one time step."""
        self._history.append({"explanation": explanation, "prediction": prediction})

    def get_history(self) -> str:
        """
        Format history as a text block for prompt injection.
        Returns empty string if no history is available.
        Most recent step appears last (closest to current context).
        """
        n = len(self._history)
        if n == 0:
            return ""

        labels = _LABELS[n]
        lines = []
        for label, entry in zip(labels, self._history):
            lines.append(
                f"[{label}]\n"
                f"  Explanation: {entry['explanation']}\n"
                f"  Prediction: {entry['prediction']}"
            )

        return "\n".join(lines)

    def clear(self) -> None:
        """Reset history (e.g. when starting a new video)."""
        self._history.clear()

"""V5 prompt: fine-tuned Qwen3.5-9B, with history context."""


_BASE_PROMPT = (
    "You are analyzing consecutive frames from a first-person (egocentric) video, "
    "shown in chronological order at regular intervals.\n\n"
    "Task: Provide an Explanation of what is happening NOW, and a Prediction of what happens NEXT.\n\n"
    "Rules for Explanation:\n"
    "- Describe the action progression across frames (first... then... finally...)\n"
    "- Name specific objects (e.g. \"white cup\", \"wooden board\"), never use vague words like \"items\" or \"objects\"\n"
    "- Include hand details: which hand does what\n"
    "- Use dynamic action verbs (reaches, grasps, lifts, places, stirs)\n"
    "- Do NOT start with \"The person is currently...\"\n"
    "- Do NOT reference the frames themselves (no \"as seen in the frames\")\n"
    "- 30-50 words\n\n"
    "Rules for Prediction:\n"
    "- Prediction must describe the NEXT action AFTER the current window ends, "
    "not a repetition of what already happened in the Explanation\n"
    "- The predicted action must be a new, different action that logically follows "
    "from the current action trend\n"
    "- Include brief reasoning linking the current action to the predicted next step\n"
    "- No vague hedging (\"possibly\", \"might\", \"could potentially\")\n"
    "- 30-50 words\n\n"
)

_HISTORY_SECTION = (
    "--- History Context (reference only) ---\n"
    "Below is context from recent time steps.\n"
    "IMPORTANT: History is supplementary context only. Do NOT let it override what you see.\n"
    "- Your Explanation must be based on the CURRENT FRAMES, not history\n"
    "- Your Prediction must follow from the CURRENT action trend you observe, "
    "not from previous predictions in history\n"
    "- Only mention objects you can actually see in the current frames — "
    "do NOT hallucinate objects from history that are not visible now\n"
    "- If history conflicts with current frames, trust the current frames\n\n"
    "{history}\n"
    "--- End History ---\n\n"
)

_CLOSING = (
    "Respond with exactly two lines:\n"
    "Explanation: <your explanation>\n"
    "Prediction: <your prediction>"
)


def build_prompt(history: str = None) -> str:
    """
    Build the V5 prompt.

    Args:
        history: Formatted history text from HistoryManager.get_history().
                 If None or empty, history section is omitted (same as V4).
    """
    parts = [_BASE_PROMPT]

    if history:
        parts.append(_HISTORY_SECTION.format(history=history))

    parts.append(_CLOSING)
    return "".join(parts)

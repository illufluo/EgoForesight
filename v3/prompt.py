"""V3 prompt: multi-frame temporal window + history context."""


_BASE_PROMPT = (
    "You are analyzing consecutive frames from a first-person (egocentric) video, "
    "shown in chronological order at regular intervals.\n\n"
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
    "Example of good output:\n"
    "Explanation: Reaches for a thin stirrer with the right hand and stirs inside the white cup "
    "on the round wooden board. Then releases the stirrer and grasps the cup, beginning to lift it "
    "from the surface.\n"
    "Prediction: Lifts the white cup toward the mouth with the right hand and takes a sip, "
    "continuing the drinking motion initiated by grasping and raising the cup from the board.\n\n"
)

_HISTORY_SECTION = (
    "--- History Context ---\n"
    "Below is context from recent time steps for reference.\n"
    "Your analysis of the CURRENT FRAMES is the primary basis for your response.\n"
    "Use history only as supplementary context — do NOT copy or repeat history content.\n"
    "If history conflicts with what you observe in current frames, trust the current frames.\n\n"
    "{history}\n"
    "--- End History ---\n\n"
)

_CLOSING = (
    "Now analyze the provided frames. Respond with exactly two lines:\n"
    "Explanation: <your explanation>\n"
    "Prediction: <your prediction>"
)


def build_prompt(history: str = None) -> str:
    """
    Build the V3 prompt.

    Args:
        history: Formatted history text from HistoryManager.get_history().
                 If None or empty, history section is omitted (same as V2).
    """
    parts = [_BASE_PROMPT]

    if history:
        parts.append(_HISTORY_SECTION.format(history=history))

    parts.append(_CLOSING)
    return "".join(parts)

"""V2 prompt: multi-frame temporal window action prediction."""


def build_prompt() -> str:
    return (
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
        "Now analyze the provided frames. Respond with exactly two lines:\n"
        "Explanation: <your explanation>\n"
        "Prediction: <your prediction>"
    )

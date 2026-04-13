"""V4 prompt: fine-tuned Qwen3.5-9B, no history."""


def build_prompt() -> str:
    return (
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
        "Respond with exactly two lines:\n"
        "Explanation: <your explanation>\n"
        "Prediction: <your prediction>"
    )

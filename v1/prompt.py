"""V1 prompt: single frame action prediction (Explanation + Prediction)."""


def build_prompt() -> str:
    return (
        "Explain what is currently happening in this first-person video frame, "
        "then predict the next action.\n\n"
        "Rules:\n"
        "- Explanation: describe the current action in 30-50 words using first/then/finally structure\n"
        "- Prediction: describe the most likely next action in 30-50 words\n"
        "- Name specific objects, include hand details\n"
        "- No vague hedging, no references to the frame itself\n"
        "- Start with action verbs\n\n"
        "Respond with exactly two lines:\n"
        "Explanation: <your explanation>\n"
        "Prediction: <your prediction>"
    )

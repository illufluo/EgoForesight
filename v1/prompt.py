"""V1 prompt: single frame action prediction."""


def build_prompt() -> str:
    return (
        "You are analyzing a single frame from an egocentric (first-person) video.\n"
        "The camera is mounted on the person's head, so you are seeing what they see.\n\n"
        "Based on this frame, predict the most likely next action the person will perform.\n"
        "Consider the objects visible, hand positions, and the current scene context.\n\n"
        "Respond with ONLY a prediction of the next action in 30-50 words.\n"
        "Format: Prediction: <your prediction>"
    )

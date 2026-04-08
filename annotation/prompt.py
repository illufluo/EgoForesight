"""Annotation-specific prompt for direct 30-50 word explanations."""


def build_annotation_prompt(narration_context: str = "") -> str:
    """
    Build the annotation prompt for generating a concise explanation (30-50 words).

    Args:
        narration_context: Formatted narration text from nearby windows.
                           Empty string if no narrations available.
    """
    parts = [
        "You are analyzing consecutive frames from a first-person (egocentric) video, "
        "shown in chronological order at regular intervals within a 1-second segment.\n\n"
        "Task: Describe what the person is doing in this segment.\n\n"
        "Rules:\n"
        "- Describe the action progression across frames (first... then... finally...)\n"
        "- Name specific objects (e.g. \"white cup\", \"metal stirrer\", \"wooden board\")\n"
        "- Include hand details: which hand does what\n"
        "- Start with action verbs, use dynamic verbs (reaches, grasps, lifts, places, stirs)\n"
        "- Do NOT start with \"The person is currently...\"\n"
        "- Do NOT reference the frames themselves (no \"in frame 3\", \"across the images\")\n"
        "- Do NOT predict future actions — only describe what is visible\n"
        "- Do NOT use hedging words (possibly, likely, appears to, seems to) — describe only what you can clearly see, or omit uncertain details\n"
        "- Always use third person, never use \"you\" or \"your\"\n"
        "- 30-50 words, concise but specific\n"
    ]

    if narration_context:
        parts.append(
            "\n--- Reference Narrations ---\n"
            "The following are human-written narrations near this time segment. "
            "Use them only to resolve ambiguity about objects or actions. "
            "Do NOT copy them — describe what you actually see in the frames.\n\n"
            f"{narration_context}\n"
            "--- End Narrations ---\n"
        )

    parts.append(
        "\nNow describe what is happening in this segment.\n"
        "Description:"
    )

    return "".join(parts)

"""Annotation-specific prompts for detailed video description."""


def build_annotation_prompt(narration_context: str = "") -> str:
    """
    Build the annotation prompt for generating explanation_full.

    Args:
        narration_context: Formatted narration text from nearby windows.
                           Empty string if no narrations available.
    """
    parts = [
        "You are analyzing consecutive frames from a first-person (egocentric) video, "
        "shown in chronological order at regular intervals within a 1-second segment.\n\n"
        "Task: Describe in detail what the person is doing in this segment.\n\n"
        "Rules:\n"
        "- Describe the action progression across frames (first... then... finally...)\n"
        "- Name specific objects (e.g. \"white cup\", \"metal stirrer\", \"wooden board\")\n"
        "- Include hand details: which hand does what, finger positions, grip changes\n"
        "- Start with action verbs, use dynamic verbs (reaches, grasps, lifts, places, stirs)\n"
        "- Do NOT start with \"The person is currently...\"\n"
        "- Do NOT reference the frames themselves (no \"in frame 3\", \"across the images\")\n"
        "- Do NOT predict future actions — only describe what is visible\n"
        "- 80-100 words, be detailed and precise\n"
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


def build_compression_prompt(explanation_full: str) -> str:
    """
    Build a text-only prompt to compress explanation_full into explanation_compact.
    """
    return (
        "Compress the following detailed action description into 30-50 words.\n\n"
        "Rules:\n"
        "- Keep the temporal progression (first... then... finally...)\n"
        "- Keep specific object names and hand details\n"
        "- Start with action verbs, use dynamic verbs\n"
        "- Do NOT add new information not in the original\n"
        "- Do NOT start with \"The person...\"\n\n"
        f"Original:\n{explanation_full}\n\n"
        "Compressed:"
    )

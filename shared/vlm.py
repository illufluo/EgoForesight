"""Backend selector: returns the appropriate call_vlm function."""


def get_call_vlm(backend: str = "glm"):
    """
    Return the call_vlm function for the specified backend.

    Args:
        backend: "glm" for GLM-4.6V API, "qwen" for local Qwen3.5-9B

    Returns:
        call_vlm(images: List[str], prompt: str) -> str
    """
    if backend == "qwen":
        from shared.qwen_client import call_vlm
    elif backend == "glm":
        from shared.glm_client import call_vlm
    else:
        raise ValueError(f"Unknown backend: {backend}. Use 'glm' or 'qwen'.")
    return call_vlm

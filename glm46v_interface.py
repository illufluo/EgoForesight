"""
GLM-4.6V Low-Level Perception Interface Module

What this module does:
- Receives an image file path
- Calls the GLM-4.6V multimodal model
- Returns raw text description of the image

What this module does NOT do:
- Does not perform action recognition or prediction
- Does not extract fixed structured fields (e.g., hand_distance, grasp, action label)
- Does not assume or enforce any intermediate state representation schema
- Does not participate in temporal reasoning or decision-making
- Does not apply structured parsing (e.g., JSON constraints) to outputs

Design Intent:
This module serves as a replaceable perception backend. It can be replaced with
other visual perception modules (e.g., MediaPipe, other multimodal models) in the
future, as long as they maintain the same input/output interface.
"""

import os
from typing import Optional


def query_glm_46v(image_path: str, prompt: Optional[str] = None) -> str:
    """
    Call GLM-4.6V to generate a perceptual description of a given image

    Args:
        image_path: Path to the input image (absolute or relative)
        prompt: Optional custom prompt. If None, uses default prompt

    Returns:
        str: Raw text description returned by GLM-4.6V model

    Raises:
        FileNotFoundError: When the image file does not exist
        ValueError: When API key is not configured or API call fails
    """

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    # Read API configuration from environment variables
    api_key = os.getenv("ZHIPUAI_API_KEY")
    if not api_key:
        raise ValueError(
            "ZHIPUAI_API_KEY environment variable not set. "
            "Please configure it before using this module."
        )

    # Default prompt: guide model to provide open-ended visual description without enforcing structured output
    if prompt is None:
        prompt = (
            "Please describe the visual content of this image in detail, including:\n"
            "1. Visible objects in the scene and their approximate locations\n"
            "2. Hand posture and position (if visible)\n"
            "3. Relative relationship between hand and objects (distance, contact, movement, etc.)\n"
            "4. Any obvious motion or state changes\n\n"
            "Please use natural language to describe freely, without using fixed formats or fields."
        )

    # Call GLM-4.6V API
    try:
        from zhipuai import ZhipuAI

        # Create client
        client = ZhipuAI(api_key=api_key)

        # Build multimodal message
        image_base64 = _encode_image_to_base64(image_path)

        response = client.chat.completions.create(
            model="glm-4v",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": image_base64}}
                    ]
                }
            ]
        )

        # Return model response
        return response.choices[0].message.content

    except ImportError:
        raise ValueError(
            "zhipuai SDK not installed. "
            "Please install it: pip install zhipuai"
        )
    except Exception as e:
        raise ValueError(f"API call error: {str(e)}")


def _encode_image_to_base64(image_path: str) -> str:
    """
    Helper function: encode image file to base64 format

    Args:
        image_path: Path to image file

    Returns:
        str: Base64-encoded image data (with data URI prefix)
    """
    import base64

    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode("utf-8")

    # Determine MIME type based on file extension
    ext = os.path.splitext(image_path)[1].lower()
    mime_type = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
    }.get(ext, "image/jpeg")

    return f"data:{mime_type};base64,{image_data}"

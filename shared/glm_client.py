"""
GLM-4.6V API Client

Supports single-image (V1) and multi-image (V2) input for VLM calls via zai-sdk.
"""

import os
import base64
import time
from typing import List
from dotenv import load_dotenv
from zai import ZhipuAiClient

# Load .env once when module is imported
load_dotenv()


def call_vlm(images: List[str], prompt: str) -> str:
    """
    Call GLM-4.6V with one or more local images and a text prompt.

    Args:
        images: List of local image file paths
        prompt: Prompt text

    Returns:
        Final text response from the model

    Raises:
        ValueError: If API key is missing or API call fails after retry
        FileNotFoundError: If any image file does not exist
    """
    api_key = os.getenv("ZHIPUAI_API_KEY") or os.getenv("ZAI_API_KEY")
    if not api_key:
        raise ValueError("API key not found. Set ZHIPUAI_API_KEY or ZAI_API_KEY.")

    if not images:
        raise ValueError("No images provided.")

    for img in images:
        if not os.path.exists(img):
            raise FileNotFoundError(f"Image file not found: {img}")

    client = ZhipuAiClient(api_key=api_key)

    # Build content: images first, then text prompt
    content = []
    for img_path in images:
        data_url = _encode_image(img_path)
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": data_url},
            }
        )

    content.append({"type": "text", "text": prompt})

    messages = [{"role": "user", "content": content}]

    # Try up to 2 times
    last_error = None
    for attempt in range(2):
        try:
            response = client.chat.completions.create(
                model="glm-4.6v",
                messages=messages,
                thinking={"type": "enabled"},
            )

            msg = response.choices[0].message

            # Return only final answer, not the whole object
            if hasattr(msg, "content") and msg.content:
                return msg.content

            raise ValueError("Model returned empty content.")

        except Exception as e:
            last_error = e
            if attempt == 0:
                print(f"  API call failed, retrying in 2s... ({e})")
                time.sleep(2)

    raise ValueError(f"API call failed after retry: {last_error}")


def _encode_image(image_path: str) -> str:
    """
    Encode a local image file into a data URL.
    Supports jpg/jpeg/png.
    """
    with open(image_path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")

    ext = os.path.splitext(image_path)[1].lower()
    mime = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
    }.get(ext)

    if mime is None:
        raise ValueError(f"Unsupported image extension: {ext}")

    return f"data:{mime};base64,{data}"
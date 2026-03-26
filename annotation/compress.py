"""Compress explanation_full to explanation_compact via text-only VLM call."""

import os
import time
from dotenv import load_dotenv
from annotation.prompt import build_compression_prompt

load_dotenv()


def compress_explanation(explanation_full: str) -> str:
    """
    Compress a detailed explanation into 30-50 words using a text-only VLM call.

    Args:
        explanation_full: The detailed explanation text (80-100 words)

    Returns:
        Compressed explanation (30-50 words)
    """
    from zai import ZhipuAiClient

    api_key = os.getenv("ZHIPUAI_API_KEY") or os.getenv("ZAI_API_KEY")
    if not api_key:
        raise ValueError("API key not found. Set ZHIPUAI_API_KEY or ZAI_API_KEY.")

    client = ZhipuAiClient(api_key=api_key)
    prompt = build_compression_prompt(explanation_full)

    messages = [{"role": "user", "content": prompt}]

    last_error = None
    for attempt in range(2):
        try:
            response = client.chat.completions.create(
                model="glm-4.6v",
                messages=messages,
            )
            content = response.choices[0].message.content
            if content:
                return content.strip()
            raise ValueError("Model returned empty content.")
        except Exception as e:
            last_error = e
            if attempt == 0:
                print(f"  Compression failed, retrying in 2s... ({e})")
                time.sleep(2)

    raise ValueError(f"Compression failed after retry: {last_error}")

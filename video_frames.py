"""
Video Frame Extraction Module

What this module does:
- Extract video frames from MP4 video files at fixed time intervals
- Save extracted frames as image files (jpg format)
- Output filenames reflect temporal order (frame_0001.jpg, frame_0002.jpg, ...)

What this module does NOT do:
- Does not perform any visual analysis or content understanding
- Does not call any machine learning models
- Does not filter or screen frame content
- Does not perform video compression or encoding conversion

Design Intent:
This module serves as a low-level data preprocessing tool, providing simple and
reliable video frame extraction functionality.
"""

import os
from typing import List, Dict
import cv2


def extract_frames(
    video_path: str,
    output_dir: str,
    interval: float = 0.2,
    output_format: str = "jpg"
) -> List[str]:
    """
    Extract frames from video at fixed time intervals

    Args:
        video_path: Path to input video file (mp4 format)
        output_dir: Output directory path, will be created if it doesn't exist
        interval: Time interval for frame extraction (seconds), default 0.2 seconds
        output_format: Output image format, supports "jpg" or "png"

    Returns:
        List[Dict]: List of frame information, each element contains {"image_path": str, "timestamp": float}

    Raises:
        FileNotFoundError: When video file does not exist
        ValueError: When video cannot be opened or parameters are invalid
    """

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    if interval <= 0:
        raise ValueError(f"Interval must be positive, got {interval}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Failed to open video file: {video_path}")

    # Get video frame rate
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        cap.release()
        raise ValueError(f"Invalid video FPS: {fps}")

    # Calculate frame interval (in number of frames)
    frame_interval = int(fps * interval)
    if frame_interval < 1:
        frame_interval = 1

    saved_frames = []
    frame_index = 0
    saved_count = 0

    # Iterate through video frames
    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # Extract frames at specified interval
        if frame_index % frame_interval == 0:
            # Calculate timestamp (seconds)
            timestamp = frame_index / fps

            # Generate filename: frame_0001.jpg, frame_0002.jpg, ...
            filename = f"frame_{saved_count + 1:04d}.{output_format}"
            output_path = os.path.join(output_dir, filename)

            # Save frame
            cv2.imwrite(output_path, frame)

            # Save frame information (path + timestamp)
            saved_frames.append({
                "image_path": output_path,
                "timestamp": round(timestamp, 2)
            })
            saved_count += 1

        frame_index += 1

    cap.release()

    return saved_frames


def get_video_info(video_path: str) -> dict:
    """
    Get basic information about a video

    Args:
        video_path: Path to video file

    Returns:
        dict: Dictionary containing video information, including:
            - frame_count: Total number of frames
            - fps: Frame rate
            - duration: Video duration (seconds)
            - width: Video width (pixels)
            - height: Video height (pixels)

    Raises:
        FileNotFoundError: When video file does not exist
        ValueError: When video cannot be opened
    """

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Failed to open video file: {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    cap.release()

    duration = frame_count / fps if fps > 0 else 0

    return {
        "frame_count": frame_count,
        "fps": fps,
        "duration": duration,
        "width": width,
        "height": height,
    }

"""Utility functions for calibration data preparation."""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict
from shutil import copy2, rmtree


def load_images_from_dir(input_dir: Path, extensions: List[str] = None) -> List[Path]:
    """
    Find all image files in directory.

    Args:
        input_dir: Directory to search
        extensions: List of file extensions (default: .jpg, .png, .jpeg)

    Returns:
        List of image file paths
    """
    if extensions is None:
        extensions = [".jpg", ".png", ".jpeg"]

    images = []
    for ext in extensions:
        images.extend(input_dir.glob(f"*{ext}"))
    return sorted(images)


def load_images_from_dirs(
    input_dir: Path,
    expected_languages: List[str],
    extensions: List[str] = None,
) -> Dict[str, List[Path]]:
    """
    Load images from language-specific subdirectories.

    Args:
        input_dir: Root directory containing language subdirs
        expected_languages: List of expected language codes
        extensions: File extensions to search (.jpg, .png, .jpeg)

    Returns:
        Dict mapping language code to list of image paths
    """
    if extensions is None:
        extensions = [".jpg", ".png", ".jpeg"]

    language_images: Dict[str, List[Path]] = {}

    for lang_code in expected_languages:
        lang_dir = input_dir / lang_code
        images = []

        if not lang_dir.exists():
            images = []
        else:
            for ext in extensions:
                images.extend(lang_dir.glob(f"*{ext}"))

        language_images[lang_code] = sorted(images)

    return language_images


def load_image(image_path: Path) -> np.ndarray:
    """
    Load image using OpenCV.

    Args:
        image_path: Path to image file

    Returns:
        Image as numpy array (BGR format)
    """
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")
    return img


def resize_image(
    image: np.ndarray, target_width: int, target_height: int
) -> np.ndarray:
    """
    Resize image maintaining aspect ratio.

    Args:
        image: Input image
        target_width: Target width
        target_height: Target height

    Returns:
        Resized image
    """
    h, w = image.shape[:2]
    scale = min(target_width / w, target_height / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)


def copy_folder(src: Path, dest: Path):
    """
    Recursively copy source directory to destination.

    Args:
        src: Source directory
        dest: Destination directory
    """
    dest.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        dest_item = dest / item.name
        if item.is_dir():
            copy_folder(item, dest_item)
        else:
            copy2(item, dest_item)


def remove_folder(dir_path: Path):
    """
    Remove directory and all contents.

    Args:
        dir_path: Directory to remove
    """
    if dir_path.exists():
        rmtree(dir_path)

"""CLI interface for preparing calibration data."""

import os
import argparse
from typing import List, Dict
from pathlib import Path
from huggingface_hub.constants import default_cache_path
from datetime import datetime as dt, timezone
import numpy as np

from .typing import CropResult, CalibrationFiles
from .detection import load_detector, detect_text_regions
from .cropping import crop_text_regions, normalize_for_inference, resize_for_detection
from .convert_tensors import images_to_onnx_tensors, prepare_calibration_manifest
from .metadata import generate_calibration_metadata, MetadataSerializer
from .language_utils import get_detected_languages
from .utils import (
    load_images_from_dir,
    load_images_from_dirs,
    load_image,
    remove_folder,
)


with open(Path(__file__).parent / ".." / "version.txt", "r") as version_file:
    version = version_file.read().strip()


def parse_args():
    parser = argparse.ArgumentParser(
        os.path.basename(__file__),
        description="""Prepare calibration data for AWQ quantization.

This module prepares calibration data for AWQ (Activation-aware Weight Quantization)
quantization of PaddleOCR models. It detects text regions in source images using
a PaddleOCR detector, crops and preprocesses the regions, and saves them in ONNX
tensor format for AWQ usage.

Workflow:
1. Load source images from input directory
2. Load PaddleOCR detector model
3. Detect text regions in each image
4. Collect and validate sample count (256-512 samples recommended)
5. For recognizer: crop, resize to 48x320, grayscale, normalize
6. For detector: use resized images directly
7. Convert to ONNX tensor format
8. Generate calibration manifest and metadata

The module supports both detector and recognizer calibration.
""",
    )

    parser.add_argument(
        "--input_dir",
        type=Path,
        required=True,
        help="Directory containing source calibration images",
    )

    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("calibration"),
        help="Output directory for calibration data",
    )

    parser.add_argument(
        "--recognizer",
        type=str,
        required=True,
        help="HuggingFace model ID or local path for target model to calibrate",
    )

    parser.add_argument(
        "--detector",
        type=str,
        required=True,
        help="HuggingFace model ID or local path for detector model (e.g., PaddlePaddle/PP-OCRv5_mobile_det)",
    )

    parser.add_argument(
        "--module",
        type=str,
        required=True,
        choices=["detector", "recognizer"],
        help="Module type: 'detector' or 'recognizer'",
    )

    parser.add_argument(
        "--detect_width",
        type=int,
        default=960,
        help="Width for detection resizing (default: 960)",
    )

    parser.add_argument(
        "--detect_height",
        type=int,
        default=960,
        help="Height for detection resizing (default: 960)",
    )

    parser.add_argument(
        "--min_confidence",
        type=float,
        default=0.5,
        help="Minimum confidence score for text detection (default: 0.5)",
    )

    parser.add_argument(
        "--min_samples",
        type=int,
        default=256,
        help="Minimum number of calibration samples needed (default: 256)",
    )

    parser.add_argument(
        "--max_samples",
        type=int,
        default=512,
        help="Maximum number of calibration samples to keep (default: 512)",
    )

    parser.add_argument(
        "--clear_cache",
        action="store_true",
        default=False,
        help="Clear HuggingFace cache after download",
    )

    return parser.parse_args()


def main(
    input_dir: Path,
    output_dir: Path,
    recognizer: str,
    detector: str,
    module: str,
    detect_width: int = 960,
    detect_height: int = 960,
    min_confidence: float = 0.5,
    min_samples: int = 256,
    max_samples: int = 512,
    clear_cache: bool = False,
):
    # Auto-detect languages from model name
    print(f"\n{'=' * 70}")
    print(f"Auto-Detecting Languages")
    print(f"{'=' * 70}")
    print(f"Model: {recognizer}")

    detected_languages = get_detected_languages(recognizer, input_dir)

    print(f"{'=' * 70}")
    print(f"Detected languages:")
    print(f"{'=' * 70}")
    print(f"{', '.join(detected_languages)}\n")

    # Step 1: Load images from language subdirectories
    print(f"\n{'=' * 70}")
    print(f"STEP 1: Loading Input Images by Language")
    print(f"{'=' * 70}")
    print(f"Input directory: {input_dir}")
    print(f"Languages: {', '.join(detected_languages)}")

    language_images = load_images_from_dirs(input_dir, detected_languages)

    missing_languages = [
        lang for lang, imgs in language_images.items() if len(imgs) == 0
    ]
    if missing_languages:
        print(
            f"\n⚠️  WARNING: No images found for languages: {', '.join(missing_languages)}"
        )

    total_input_images = sum(len(imgs) for imgs in language_images.values())
    print(f"\nFound {total_input_images} images across all languages")

    # Step 2: Load detector
    print(f"\n{'=' * 70}")
    print(f"STEP 2: Loading Detector Model")
    print(f"{'=' * 70}")

    detector = load_detector(detector)

    # Step 3-7: Process each language separately
    print(f"\n{'=' * 70}")
    print(f"STEP 3-7: Process Each Language")
    print(f"{'=' * 70}\n")

    output_dir.mkdir(parents=True, exist_ok=True)

    all_crops_by_lang: Dict[str, List[CropResult]] = {
        lang: [] for lang in detected_languages
    }
    all_image_paths_by_lang: Dict[str, List[str]] = {
        lang: [] for lang in detected_languages
    }

    for lang_code in detected_languages:
        print(f"\n{'=' * 70}")
        print(f"Processing language: {lang_code}")
        print(f"{'=' * 70}")

        lang_output_dir = output_dir / lang_code
        lang_output_dir.mkdir(parents=True, exist_ok=True)

        calibration_dir = lang_output_dir / "calibration"
        calibration_dir.mkdir(parents=True, exist_ok=True)

        images = language_images[lang_code]

        if len(images) == 0:
            print(f"SKIP: No images for {lang_code}")
            continue

        print(f"Processing {len(images)} images for {lang_code}...")

        for img_path in images:
            image = load_image(img_path)
            resized = resize_for_detection(image, detect_width, detect_height)

            boxes, scores = detect_text_regions(
                img_path, resized, detector, min_confidence
            )

            if len(boxes) == 0:
                continue

            detection_result = {
                "boxes": boxes,
                "scores": scores,
                "image_path": str(img_path),
            }

            crops = crop_text_regions(image, detection_result)

            for crop in crops:
                all_crops_by_lang[lang_code].append(crop)
                all_image_paths_by_lang[lang_code].append(str(img_path))

        crops = all_crops_by_lang[lang_code]

        if len(crops) == 0:
            print(f"SKIP: No crops detected for {lang_code}")
            continue

        print(f"Total regions for {lang_code}: {len(crops)}")

        # Sample validation (can exceed min_sample per language)
        if len(crops) > max_samples:
            print(f"⚠️  INFO: {len(crops)} samples (max: {max_samples})")

            import random

            random.seed(42)

            crops = crops[:max_samples]
            all_crops_by_lang[lang_code] = crops

        # Preprocess
        if module == "recognizer":
            preprocessed_samples = [
                normalize_for_inference(
                    crop["image"], target_width=320, target_height=48
                )
                for crop in crops
            ]
        else:
            preprocessed_samples = []
            for crop in crops:
                resized = resize_for_detection(
                    crop["image"], detect_width, detect_height
                )
                resized = resized[np.newaxis, np.newaxis, :, :]
                preprocessed_samples.append(resized)

        print(f"Preprocessed {len(preprocessed_samples)} samples for {lang_code}")

        # Save ONNX tensors
        tensor_path = calibration_dir / "calibration.onnx"
        images_to_onnx_tensors(
            preprocessed_samples, tensor_path, tensor_name="x", batch_axis=True
        )

        # Create language-specific manifest
        manifest_path = calibration_dir / "manifest.json"
        prepare_calibration_manifest(
            crops,
            all_image_paths_by_lang[lang_code],
            [lang_code] * len(crops),
            manifest_path,
        )

        # Generate metadata
        calibration_files: CalibrationFiles = {
            "tensors": str(tensor_path.relative_to(lang_output_dir)),
            "manifest": str(manifest_path.relative_to(lang_output_dir)),
        }

        tensor_info = None
        if module == "recognizer":
            tensor_info = {
                "name": "x",
                "dtype": "float32",
                "shape": preprocessed_samples[0].shape,
                "preprocessing": {
                    "resize": [48, 320],
                    "grayscale": True,
                    "normalize": {"scale": 1.0 / 255.0, "mean": 0.5, "std": 0.5},
                },
            }
        else:
            tensor_info = {
                "name": "x",
                "dtype": "float16",
                "shape": preprocessed_samples[0].shape,
                "preprocessing": {"resize": [detect_height, detect_width]},
            }

        avg_score = (
            sum(c["score"] for c in crops) / len(crops) if len(crops) > 0 else 0.0
        )
        language_stats = {lang_code: {"avg_confidence": avg_score}}

        generate_calibration_metadata(
            version=version,
            output_dir=lang_output_dir,
            detector_model=detector,
            target_model=recognizer,
            module=module,
            num_samples=len(preprocessed_samples),
            calibration_files=calibration_files,
            tensor_info=tensor_info,
            language_stats=language_stats,
        )

        print(f"✓ {lang_code} complete: {len(preprocessed_samples)} samples\n")

    # Validate total samples across all languages
    total_samples = sum(len(all_crops_by_lang[lang]) for lang in detected_languages)

    print(f"\n{'=' * 70}")
    print(f"Total Samples Validation")
    print(f"{'=' * 70}")
    print(f"Total regions detected: {total_samples}")
    print(f"Recommended sample range: {min_samples}-{max_samples}")

    if total_samples < min_samples:
        print(
            f"\n⚠️  WARNING: Only {total_samples} samples detected (min: {min_samples})"
        )
        user_input = input("Continue anyway? (y/N): ")
        if user_input.lower() != "y":
            raise ValueError("Insufficient calibration samples")

    if total_samples > max_samples:
        print(f"\n⚠️  INFO: {total_samples} samples detected (max: {max_samples})")
        print(
            f"   Recommendation: Ensure each language stays within {max_samples} samples"
        )

    # Create master calibration index
    print(f"\n{'=' * 70}")
    print(f"Creating Master Calibration Index")
    print(f"{'=' * 70}\n")

    language_calibrations: Dict[str, Dict] = {}
    for lang_code in detected_languages:
        crops = all_crops_by_lang[lang_code]
        avg_score = (
            sum(c["score"] for c in crops) / len(crops) if len(crops) > 0 else 0.0
        )
        language_calibrations[lang_code] = {
            "num_samples": len(crops),
            "avg_confidence": avg_score,
        }

    master_index = {
        "format": "multi_language_calibration",
        "version": version,
        "target_model": recognizer,
        "detector_model": detector,
        "module": module,
        "total_samples": total_samples,
        "languages_by_code": language_calibrations,
        "expected_languages": detected_languages,
        "created_at": dt.now(timezone.utc).isoformat(),
    }

    master_index_path = output_dir / "calibration_index.json"
    print(f"Saving master index to: {master_index_path}")

    import json

    with open(master_index_path, "w", encoding="utf-8") as f:
        json.dump(master_index, f, indent=2, ensure_ascii=False, cls=MetadataSerializer)

    print(f"✓ Master index saved\n")

    # Step 8: Cleanup
    print(f"\n{'=' * 70}")
    print(f"STEP 8: Cleanup")
    print(f"{'=' * 70}")

    if clear_cache:
        cache_path = Path(default_cache_path) / "models" / detector.replace("/", "--")
        remove_folder(cache_path)
        print("HuggingFace cache cleared")

    print(f"\n{'=' * 70}")
    print(f"✓ Calibration data preparation complete!")
    print(f"{'=' * 70}")
    print(f"\nOutput directory: {output_dir}")
    print(f"Master index: {master_index_path}")
    print(f"Total samples across all languages: {total_samples}")
    print(f"Languages processed: {', '.join(detected_languages)}")
    print(f"\n{'=' * 70}\n")


if __name__ == "__main__":
    args = parse_args()
    main(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        recognizer=args.recognizer,
        detector=args.detector,
        module=args.module,
        detect_width=args.detect_width,
        detect_height=args.detect_height,
        min_confidence=args.min_confidence,
        min_samples=args.min_samples,
        max_samples=args.max_samples,
        clear_cache=args.clear_cache,
    )

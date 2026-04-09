"""
CLI for OmniDocBench-aligned OCR evaluation.

Usage:
    # Run PaddleOCR (default)
    python -m versta.evaluate --output-dir ./output --model-type paddleocr
    
    # Run ONNX
    python -m versta.evaluate --output-dir ./output --model-type onnx --detector path/detector.ort --recognizer path/recognizer.ort
    
    # Evaluation only (use existing .md files)
    python -m versta.evaluate --output-dir ./output --eval-only
"""
import os
import sys
import traceback
from argparse import ArgumentParser
from pathlib import Path

from math import floor

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

os.environ['FLAGS_use_mkldnn'] = '0'
os.environ['OMNIDOCBENCH_PDFLATEX'] = 'pdftex'


def parse_args():
    parser = ArgumentParser(
        description="OmniDocBench-aligned OCR evaluation"
    )
    
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for predictions and results"
    )
    
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["paddleocr", "onnx", "both"],
        default="both",
        help="Model type to evaluate (default: both)"
    )
    
    parser.add_argument(
        "--detector",
        type=Path,
        help="Path to ONNX detector model (required for ONNX mode)"
    )
    
    parser.add_argument(
        "--recognizer",
        type=Path,
        help="Path to ONNX recognizer model (required for ONNX mode)"
    )
    
    parser.add_argument(
        "--model-name-detector",
        type=str,
        default="PP-OCRv5_mobile_det",
        help="PaddleOCR detector name (for PaddleOCR mode)"
    )
    
    parser.add_argument(
        "--model-name-recognizer",
        type=str,
        default="PP-OCRv5_mobile_rec",
        help="PaddleOCR recognizer name (for PaddleOCR mode)"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for inference (default: 8)"
    )
    
    parser.add_argument(
        "--num-threads",
        type=int,
        default=floor(os.cpu_count() / 2),
        help="Number of CPU threads for inference (default: 0 for all cores)"
    )

    parser.add_argument(
        "--subset",
        type=int,
        help="Number of samples to evaluate (default: all)"
    )
    
    parser.add_argument(
        "--inference-only",
        action="store_true",
        default=False,
        help="Only run inference, skip evaluation"
    )
    
    parser.add_argument(
        "--eval-only",
        action="store_true",
        default=False,
        help="Only run evaluation, skip inference"
    )
    
    parser.add_argument(
        "--match-method",
        type=str,
        choices=["quick_match", "simple_match", "no_split"],
        default="quick_match",
        help="Matching algorithm (default: quick_match)"
    )
    
    parser.add_argument(
        "--match-workers",
        type=int,
        default=4,
        help="Number of workers for matching (default: 4)"
    )
    
    return parser.parse_args()


def run_inference_and_eval(args):
    """Run the full inference + evaluation pipeline."""
    from .dataset import get_omnidocbench_images, download_omnidocbench_json
    from .inference import run_inference
    from .run_eval import run_evaluation
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.eval_only:
        print("Downloading OmniDocBench annotations...")
        try:
            gt_json_path = download_omnidocbench_json()
            images_dir = None
        except Exception as e:
            print(f"Error downloading annotations: {e}")
            return
    else:
        print(f"Downloading OmniDocBench dataset (this may take a while)...")
        try:
            images_dir, gt_json_path = get_omnidocbench_images()
        except Exception as e:
            print(f"Error downloading dataset: {e}")
            return

    model_types = [args.model_type] if args.model_type != "both" else ["paddleocr", "onnx"]
    
    for model_type in model_types:
        print(f"\n{'='*60}")
        print(f"Evaluating model: {model_type}")
        print(f"{'='*60}")
        
        pred_dir = output_dir / "predictions" / model_type
        pred_dir.mkdir(parents=True, exist_ok=True)
        
        if not args.eval_only:
            print(f"\nRunning inference...")
            
            inference_success = False
            
            if model_type == "onnx":
                if not args.detector or not args.recognizer:
                    print(f"Error: --detector and --recognizer required for ONNX mode")
                    continue
                
                try:
                    run_inference(
                        image_dir=images_dir,
                        output_dir=pred_dir,
                        model_type="onnx",
                        detector_path=args.detector,
                        recognizer_path=args.recognizer,
                        batch_size=args.batch_size,
                        num_threads=args.num_threads,
                    )
                    inference_success = True
                except Exception as e:
                    print(f"Error running ONNX inference: {e}")
                    print(f"  -> You may need to re-export your ONNX models")
                    traceback.print_exc()
            else:
                try:
                    run_inference(
                        image_dir=images_dir,
                        output_dir=pred_dir,
                        model_type="paddleocr",
                        model_name_det=args.model_name_detector,
                        model_name_rec=args.model_name_recognizer,
                        batch_size=args.batch_size,
                        num_threads=args.num_threads,
                    )
                    inference_success = True
                except Exception as e:
                    print(f"Error running PaddleOCR inference: {e}")
                    traceback.print_exc()
            
            if not inference_success:
                print(f"Skipping evaluation for {model_type} due to inference failure")
                continue
        
        if not args.inference_only:
            print(f"\nRunning evaluation...")
            
            try:
                result = run_evaluation(
                    predictions_dir=pred_dir,
                    gt_json_path=gt_json_path,
                    output_dir=output_dir / "eval" / model_type,
                    match_method=args.match_method,
                    match_workers=args.match_workers,
                )
                
                result_dir = output_dir / "eval" / model_type / "result"
                result_files = list(result_dir.glob("*_metric_result.json"))
                if result_files:
                    from .generate_score import generate_result_tables
                    score_result = generate_result_tables(
                        result_files[0],
                        model_name=f"{model_type}_{args.match_method}",
                    )
                    print(f"\n{score_result['summary']}")
                else:
                    print(f"\nResults for {model_type}:")
                    print(f"  {result}")
            except Exception as e:
                print(f"Error running evaluation: {e}")
                traceback.print_exc()


def main():
    args = parse_args()
    
    run_inference_and_eval(args)
    
    print(f"\nDone! Output in: {args.output_dir}")


if __name__ == "__main__":
    main()

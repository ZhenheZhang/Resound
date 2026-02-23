#!/usr/bin/env python3
"""
WavBench inference runner — Gemini 3.1 Pro.

Usage:
    # Single dataset
    python run_inference.py --data basic_creative --wavbench_dir D:/WavBench_download

    # All datasets
    python run_inference.py --data all --wavbench_dir D:/WavBench_download

    # Dry-run (validate paths, no API calls)
    python run_inference.py --data basic_creative --wavbench_dir D:/WavBench_download --dry_run

    # Custom output dir, more workers
    python run_inference.py --data all --wavbench_dir /data/WavBench_download \\
        --output_dir ./predictions --max_workers 4
"""

import os
import sys
import json
import logging
import argparse
import traceback
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from src.models.gemini_model import GeminiWavBenchModel
from src.data.dataset_loader import (
    DATASET_NAME_MAP,
    ALL_DATASETS,
    WavBenchSample,
    load_from_local,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_inference")


# ── Single-sample inference ───────────────────────────────────────────────────

def infer_sample(
    model: GeminiWavBenchModel,
    sample: WavBenchSample,
    dry_run: bool = False,
) -> dict:
    """Run Gemini inference on one sample and return a result dict."""
    task_type    = sample.metadata.get("task_type", "colloquial")
    implicit_type = sample.metadata.get("implicit_type")

    base = {
        "sample_id":   sample.sample_id,
        "audio_path":  sample.audio_path,
        "task_type":   task_type,
        "panel":       sample.panel,
        "domain":      sample.domain,
        "attribute":   sample.attribute,
        "class":       sample.metadata.get("class"),
        "question":    sample.question,
        "reference":   sample.reference_answer,
    }

    if dry_run:
        return {**base, "prediction": "[DRY RUN]", "error": None}

    try:
        prediction = model.run(
            audio_path=sample.audio_path,
            task_type=task_type,
            attribute=sample.attribute,
            question=sample.question,
            domain=sample.domain,
            target_value=sample.target_value,
            history=sample.history,
            implicit_type=implicit_type,
        )
        return {**base, "prediction": prediction, "error": None}

    except Exception as e:
        logger.error(f"Error on {sample.sample_id}: {e}")
        return {**base, "prediction": None, "error": str(e)}


# ── Dataset runner ────────────────────────────────────────────────────────────

def run_dataset(
    dataset_name: str,
    model: GeminiWavBenchModel,
    output_dir: Path,
    wavbench_dir: str,
    max_workers: int = 4,
    dry_run: bool = False,
    resume: bool = True,
) -> list[dict]:
    """
    Run inference on all samples in one dataset and stream results to JSONL.

    Args:
        dataset_name : e.g. "basic_creative", "acoustic_explicit_understanding_emotion"
        model        : Initialised GeminiWavBenchModel
        output_dir   : Directory where {dataset_name}_predictions.jsonl is written
        wavbench_dir : Root of WavBench_download (passed to load_from_local)
        max_workers  : ThreadPoolExecutor workers (keep ≤4 to respect rate limits)
        dry_run      : Skip API calls; write placeholder predictions
        resume       : Skip already-completed sample_ids from a previous run
    """
    output_file = output_dir / f"{dataset_name}_predictions.jsonl"

    # Resume: collect sample_ids already written
    done_ids: set[str] = set()
    if resume and output_file.exists():
        with open(output_file, encoding="utf-8", errors="replace") as f:
            for line in f:
                try:
                    done_ids.add(json.loads(line)["sample_id"])
                except Exception:
                    pass
        if done_ids:
            logger.info(f"  Resuming '{dataset_name}': {len(done_ids)} samples already done")

    # Load samples from the local WavBench_download directory
    samples = [
        s for s in load_from_local(dataset_name, wavbench_dir)
        if s.sample_id not in done_ids
    ]
    logger.info(f"  Running {len(samples)} samples for '{dataset_name}'")

    if not samples:
        logger.info(f"  Nothing to do for '{dataset_name}'")
        return []

    results: list[dict] = []

    with open(output_file, "a", encoding="utf-8") as fout:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(infer_sample, model, s, dry_run): s
                for s in samples
            }
            for i, future in enumerate(as_completed(futures)):
                sample = futures[future]
                try:
                    result = future.result()
                except Exception as e:
                    logger.error(
                        f"Unexpected error on {sample.sample_id}:\n"
                        + traceback.format_exc()
                    )
                    result = {
                        "sample_id": sample.sample_id,
                        "audio_path": sample.audio_path,
                        "prediction": None,
                        "error": str(e),
                    }

                results.append(result)
                fout.write(json.dumps(result, ensure_ascii=False) + "\n")
                fout.flush()

                if (i + 1) % 10 == 0 or i == 0:
                    errors = sum(1 for r in results if r.get("error"))
                    logger.info(
                        f"  [{dataset_name}] {i+1}/{len(samples)} "
                        f"({'DRY RUN' if dry_run else f'{errors} errors'})"
                    )

    logger.info(f"  Done '{dataset_name}' → {output_file}")
    return results


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Run WavBench inference with Gemini 3.1 Pro",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_inference.py --data basic_creative \\
      --wavbench_dir D:/Zhenhe/Coding/Projects/WavBench/WavBench_download

  python run_inference.py --data all \\
      --wavbench_dir /data/WavBench_download --max_workers 2

  python run_inference.py --data acoustic_explicit_understanding_emotion \\
      --wavbench_dir D:/WavBench_download --dry_run
        """,
    )

    parser.add_argument(
        "--data",
        required=True,
        metavar="DATASET",
        help=(
            "Dataset to run. Use 'all' for the full benchmark, or any name from:\n  "
            + ", ".join(ALL_DATASETS)
        ),
    )
    parser.add_argument(
        "--wavbench_dir",
        required=True,
        metavar="PATH",
        help=(
            "Path to the WavBench_download root directory.\n"
            "Example: D:/Zhenhe/Coding/Projects/WavBench/WavBench_download"
        ),
    )
    parser.add_argument(
        "--output_dir",
        default="./predictions",
        metavar="PATH",
        help="Directory for prediction JSONL files (default: ./predictions)",
    )
    parser.add_argument(
        "--model_id",
        default="[次]gemini-3.1-pro-preview",
        help="Gemini model ID (default: [次]gemini-3.1-pro-preview)",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=4,
        help="Parallel API workers — keep ≤4 to avoid rate limits (default: 4)",
    )
    parser.add_argument(
        "--self_consistency_k",
        type=int,
        default=3,
        help="Votes for classification self-consistency (default: 3)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature (default: 0.2)",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Validate paths and print samples without calling the Gemini API",
    )
    parser.add_argument(
        "--no_resume",
        action="store_true",
        help="Reprocess all samples even if a partial output file already exists",
    )

    args = parser.parse_args()

    # Resolve dataset list
    if args.data == "all":
        datasets = ALL_DATASETS
    elif args.data in DATASET_NAME_MAP:
        datasets = [args.data]
    else:
        parser.error(
            f"Unknown dataset: '{args.data}'\n"
            f"Valid options: all, " + ", ".join(ALL_DATASETS)
        )

    # Validate wavbench_dir exists
    wavbench_root = Path(args.wavbench_dir).resolve()
    if not wavbench_root.exists():
        parser.error(f"--wavbench_dir not found: {wavbench_root}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialise model (skipped for dry-run)
    if not args.dry_run:
        api_key = os.environ.get("GOOGLE_API_KEY", "")
        if not api_key:
            parser.error(
                "GOOGLE_API_KEY environment variable is not set.\n"
                "Run: export GOOGLE_API_KEY='your-key'"
            )
        model = GeminiWavBenchModel(
            model_id=args.model_id,
            temperature=args.temperature,
            self_consistency_k=args.self_consistency_k,
        )
    else:
        model = None

    # Run
    start = datetime.now()
    total_samples = 0
    total_errors = 0

    logger.info(f"WavBench_download : {wavbench_root}")
    logger.info(f"Output dir        : {output_dir}")
    logger.info(f"Datasets          : {len(datasets)}")
    logger.info(f"Model             : {args.model_id}")
    logger.info(f"Dry run           : {args.dry_run}")

    for dataset_name in datasets:
        logger.info(f"\n{'='*60}")
        logger.info(f"  {dataset_name}")
        logger.info(f"{'='*60}")
        results = run_dataset(
            dataset_name=dataset_name,
            model=model,
            output_dir=output_dir,
            wavbench_dir=str(wavbench_root),
            max_workers=args.max_workers,
            dry_run=args.dry_run,
            resume=not args.no_resume,
        )
        total_samples += len(results)
        total_errors  += sum(1 for r in results if r.get("error"))

    elapsed = (datetime.now() - start).total_seconds()
    logger.info(f"\n{'='*60}")
    logger.info(
        f"DONE — {total_samples} samples processed, "
        f"{total_errors} errors, {elapsed:.1f}s elapsed"
    )
    logger.info(f"Predictions written to: {output_dir}/")


if __name__ == "__main__":
    main()

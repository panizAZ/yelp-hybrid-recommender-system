# system.py
"""
CLI entry point and pipeline orchestrator for Assignment 3.

This script runs the full pipeline:

1. text_processor.py
   -> HW3_data/train_reviews.csv
   -> HW3_data/dev_reviews.csv
   -> HW3_data/dev_qrels.tsv
   -> HW3_data/dev_users.csv
   -> HW3_data/preprocessing_summary.json

2. sentiment.py
   -> HW3_data/train_df_with_aspects.csv
   -> HW3_data/user_attention_matrix_U.csv
   -> HW3_data/item_quality_matrix_V.csv
   -> results/aspect_summary.json

3. hybrid_model.py
   -> results/dev_predictions_*.csv
   -> results/dev_recommendations_*.csv
   -> results/test_recommendations.csv
   -> results/trace.jsonl
   -> results/baseline_manifest.json
   -> results/summary.json

4. evaluator.py
   -> results/dev_metrics.json

example: python system.py --skip-preprocessing --use-transformer --transformer-model "distilbert-base-uncased-finetuned-sst-2-english" --alpha 0.9
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path


def run_command(command: list[str], step_name: str) -> None:
    """
    Run one pipeline command and fail immediately if it exits with non-zero status.
    """
    print("\n" + "=" * 80)
    print(f"Starting step: {step_name}")
    print("=" * 80)
    print("Command:")
    print(" ".join(command))
    print("-" * 80, flush=True)

    start_time = time.time()

    subprocess.run(command, check=True)

    elapsed = time.time() - start_time
    print("-" * 80)
    print(f"Finished step: {step_name} in {elapsed:.2f} seconds")
    print("=" * 80 + "\n", flush=True)


def check_file_exists(path: str, description: str) -> None:
    """
    Validate that a required output file exists after a pipeline step.
    """
    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Expected {description} was not created: {file_path}"
        )

    print(f"[OK] {description}: {file_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Assignment 3 CLI pipeline orchestrator."
    )

    # ------------------------------------------------------------------
    # Raw data paths
    # ------------------------------------------------------------------
    parser.add_argument(
        "--business-path",
        type=str,
        default="HW3_data/Yelp-JSON/Yelp JSON/yelp_dataset/yelp_academic_dataset_business.json",
        help="Path to Yelp business JSON file.",
    )

    parser.add_argument(
        "--review-path",
        type=str,
        default="HW3_data/Yelp-JSON/Yelp JSON/yelp_dataset/yelp_academic_dataset_review.json",
        help="Path to Yelp review JSON file.",
    )

    # ------------------------------------------------------------------
    # Directories
    # ------------------------------------------------------------------
    parser.add_argument(
        "--data-dir",
        type=str,
        default="HW3_data",
        help="Directory for processed data files.",
    )

    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Directory for final outputs.",
    )

    # ------------------------------------------------------------------
    # Pipeline control
    # ------------------------------------------------------------------
    parser.add_argument(
        "--skip-preprocessing",
        action="store_true",
        help="Skip text_processor.py if preprocessing outputs already exist.",
    )

    parser.add_argument(
        "--skip-sentiment",
        action="store_true",
        help="Skip sentiment.py if aspect matrices already exist.",
    )

    parser.add_argument(
        "--skip-hybrid",
        action="store_true",
        help="Skip hybrid_model.py if recommendation/model outputs already exist.",
    )

    parser.add_argument(
        "--skip-evaluation",
        action="store_true",
        help="Skip evaluator.py.",
    )

    # ------------------------------------------------------------------
    # Sentiment options
    # ------------------------------------------------------------------
    parser.add_argument(
        "--spacy-model",
        type=str,
        default="en_core_web_md",
        help="spaCy model name used by sentiment.py if supported.",
    )

    parser.add_argument(
        "--n-process",
        type=int,
        default=4,
        help="Number of processes used by sentiment.py if supported.",
    )

    # ------------------------------------------------------------------
    # Hybrid model hyperparameters
    # ------------------------------------------------------------------
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=0.7,
        help="Hybrid weight. alpha=1.0 means SVD only; alpha=0.0 means aspect only.",
    )

    parser.add_argument(
        "--latent-dim",
        type=int,
        default=32,
        help="Latent dimension for matrix factorization.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=80,
        help="Maximum number of training epochs.",
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.005,
        help="Learning rate.",
    )

    parser.add_argument(
        "--regularization",
        type=float,
        default=0.02,
        help="Regularization strength.",
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=8,
        help="Early stopping patience.",
    )

    parser.add_argument(
        "--min-train-reviews",
        type=int,
        default=5,
        help="Minimum user train reviews before non-fallback recommendation is allowed.",
    )

    parser.add_argument(
        "--tau",
        type=float,
        default=0.30,
        help="Abstention threshold for maximum hybrid score.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of recommendations per user and K for metrics.",
    )

    parser.add_argument(
        "--use-transformer",
        action="store_true",
        help="Use Transformer-based sentiment scoring instead of VADER.",
    )

    parser.add_argument(
        "--transformer-model",
        type=str,
        default="distilbert-base-uncased-finetuned-sst-2-english",
        help="Hugging Face checkpoint for sentiment scoring.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    data_dir = Path(args.data_dir)
    results_dir = Path(args.results_dir)

    data_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Standard paths
    # ------------------------------------------------------------------
    train_reviews_path = data_dir / "train_reviews.csv"
    dev_reviews_path = data_dir / "dev_reviews.csv"
    dev_qrels_path = data_dir / "dev_qrels.tsv"
    dev_users_path = data_dir / "dev_users.csv"
    test_users_path = data_dir / "test_users.csv"

    preprocessing_summary_path = data_dir / "preprocessing_summary.json"

    user_attention_path = data_dir / "user_attention_matrix_U.csv"
    item_quality_path = data_dir / "item_quality_matrix_V.csv"
    train_with_aspects_path = data_dir / "train_df_with_aspects.csv"

    aspect_summary_path = data_dir / "aspect_summary.json"

    dev_recommendations_path = results_dir / "dev_recommendations.csv"
    test_recommendations_path = results_dir / "test_recommendations.csv"
    trace_path = results_dir / "trace.jsonl"
    summary_path = results_dir / "summary.json"
    baseline_manifest_path = results_dir / "baseline_manifest.json"
    dev_metrics_path = results_dir / "dev_metrics.json"

    pipeline_start = time.time()

    # ------------------------------------------------------------------
    # Step 1: Preprocessing
    # ------------------------------------------------------------------
    if not args.skip_preprocessing:
        run_command(
            [
                sys.executable,
                "text_processor.py",
                "--business-path",
                args.business_path,
                "--review-path",
                args.review_path,
                "--outdir",
                str(data_dir),
            ],
            step_name="Preprocessing / Phase 0",
        )

        check_file_exists(str(train_reviews_path), "train reviews")
        check_file_exists(str(dev_reviews_path), "dev reviews")
        check_file_exists(str(dev_qrels_path), "dev qrels")
        check_file_exists(str(dev_users_path), "dev users")
        check_file_exists(str(preprocessing_summary_path), "preprocessing summary")
    else:
        print("[SKIP] Preprocessing step skipped.")

    # ------------------------------------------------------------------
    # Step 2: Sentiment / aspect extraction
    # ------------------------------------------------------------------
    if not args.skip_sentiment:
        # Your current sentiment.py uses hard-coded defaults in __main__:
        # csv_path="HW3_data/train_reviews.csv"
        # model="en_core_web_md"
        # n_process=4
        #
        # If you later add argparse to sentiment.py, system.py can pass
        # --csv-path, --model-name, --n-process, etc.
        run_command(
            [
                sys.executable,
                "sentiment.py",
                "--spacy-model",
                args.spacy_model,
                "--n-process",
                str(args.n_process),
                "--use-transformer",
                "--transformer-model",
                args.transformer_model,
            ],
            step_name="Aspect extraction and Transformer sentiment",
        )

        check_file_exists(str(train_with_aspects_path), "train dataframe with aspects")
        check_file_exists(str(user_attention_path), "user attention matrix U")
        check_file_exists(str(item_quality_path), "item quality matrix V")
        check_file_exists(str(aspect_summary_path), "aspect summary")
    else:
        print("[SKIP] Sentiment/aspect extraction step skipped.")

    # ------------------------------------------------------------------
    # Step 3: Hybrid model, recommendations, trace, summary, manifest
    # ------------------------------------------------------------------
    if not args.skip_hybrid:
        run_command(
            [
                sys.executable,
                "hybrid_model.py",
                "--seed",
                str(args.seed),
                "--alpha",
                str(args.alpha),
                "--latent_dim",
                str(args.latent_dim),
                "--epochs",
                str(args.epochs),
                "--learning_rate",
                str(args.learning_rate),
                "--regularization",
                str(args.regularization),
                "--patience",
                str(args.patience),
                "--min_train_reviews",
                str(args.min_train_reviews),
                "--tau",
                str(args.tau),
                "--dev_users_path",
                str(dev_users_path),
                "--test_users_path",
                str(test_users_path),
                "--output_dev_path",
                str(dev_recommendations_path),
                "--output_test_path",
                str(test_recommendations_path),
                "--output_trace_path",
                str(trace_path),
                "--summary_path",
                str(summary_path),
                "--preprocessing_summary_path",
                str(preprocessing_summary_path),
                "--aspect_summary_path",
                str(aspect_summary_path),
            ],
            step_name="Hybrid recommendation model",
        )

        check_file_exists(str(dev_recommendations_path), "dev recommendations")
        check_file_exists(str(test_recommendations_path), "test recommendations")
        check_file_exists(str(trace_path), "trace jsonl")
        check_file_exists(str(summary_path), "final summary")
        check_file_exists(str(baseline_manifest_path), "baseline manifest")
    else:
        print("[SKIP] Hybrid model step skipped.")

    # ------------------------------------------------------------------
    # Step 4: Evaluation
    # ------------------------------------------------------------------
    if not args.skip_evaluation:
        run_command(
            [
                sys.executable,
                "evaluator.py",
                "--train-reviews",
                str(train_reviews_path),
                "--dev-users",
                str(dev_users_path),
                "--dev-qrels",
                str(dev_qrels_path),
                "--baseline-manifest",
                str(baseline_manifest_path),
                "--output",
                str(dev_metrics_path),
                "--k",
                str(args.top_k),
            ],
            step_name="Evaluation / dev_metrics.json",
        )

        check_file_exists(str(dev_metrics_path), "dev metrics")
    else:
        print("[SKIP] Evaluation step skipped.")

    total_elapsed = time.time() - pipeline_start

    print("\n" + "=" * 80)
    print("Pipeline completed successfully.")
    print(f"Total runtime: {total_elapsed:.2f} seconds")
    print("=" * 80)

    print("\nFinal expected outputs:")
    print(f"- {dev_recommendations_path}")
    print(f"- {test_recommendations_path}")
    print(f"- {trace_path}")
    print(f"- {baseline_manifest_path}")
    print(f"- {dev_metrics_path}")
    print(f"- {summary_path}")


if __name__ == "__main__":
    main()
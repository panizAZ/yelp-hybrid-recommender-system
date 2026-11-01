import argparse
import json
import math
from pathlib import Path
from typing import Dict, Set, Tuple, Any

import numpy as np
import pandas as pd

# python evaluator.py `
#   --train-reviews HW3_data/train_reviews.csv `
#   --dev-users HW3_data/dev_users.csv `
#   --dev-qrels HW3_data/dev_qrels.tsv `
#   --baseline-manifest results/baseline_manifest.json `
#   --output results/dev_metrics.json


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Root Mean Squared Error.
    """
    if len(y_true) == 0:
        return 0.0

    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def load_eval_users(dev_users_path: str) -> Set[str]:
    """
    Load user IDs that should be evaluated.
    """
    df = pd.read_csv(
        dev_users_path,
        dtype={"user_id": str},
    )

    if "user_id" not in df.columns:
        raise ValueError(
            f"{dev_users_path} must contain a user_id column. "
            f"Available columns: {list(df.columns)}"
        )

    return set(df["user_id"].astype(str).dropna().unique())


def load_training_catalogue(train_reviews_path: str) -> Set[str]:
    """
    Training catalogue = businesses with at least one training review.
    """
    train_df = pd.read_csv(
        train_reviews_path,
        dtype={"user_id": str, "business_id": str},
    )

    if "business_id" not in train_df.columns:
        raise ValueError(
            f"{train_reviews_path} must contain a business_id column."
        )

    return set(train_df["business_id"].astype(str).dropna().unique())


def load_qrels(
    qrels_path: str,
    eval_users: Set[str],
    training_catalogue: Set[str],
) -> Dict[str, Set[str]]:
    """
    Load binary relevance labels from dev_qrels.tsv.

    Expected columns:
        user_id, business_id, relevance

    If relevance is missing, every row is treated as relevant.
    """

    qrels = pd.read_csv(
        qrels_path,
        sep="\t",
        dtype={"user_id": str, "business_id": str},
    )

    required = {"user_id", "business_id"}
    missing = required - set(qrels.columns)
    if missing:
        raise ValueError(
            f"{qrels_path} is missing columns: {missing}. "
            f"Available columns: {list(qrels.columns)}"
        )

    qrels["user_id"] = qrels["user_id"].astype(str)
    qrels["business_id"] = qrels["business_id"].astype(str)

    if "relevance" not in qrels.columns:
        qrels["relevance"] = 1

    qrels["relevance"] = qrels["relevance"].astype(float)

    # Catalogue and eval-user constraints.
    qrels = qrels[
        qrels["user_id"].isin(eval_users)
        & qrels["business_id"].isin(training_catalogue)
    ].copy()

    # If duplicates exist, keep maximum relevance.
    qrels = (
        qrels.groupby(["user_id", "business_id"], as_index=False)["relevance"]
        .max()
    )

    # Binary relevance for HR/NDCG.
    qrels["relevance"] = (qrels["relevance"] > 0).astype(int)

    relevant_items = (
        qrels[qrels["relevance"] == 1]
        .groupby("user_id")["business_id"]
        .apply(set)
        .to_dict()
    )

    return relevant_items


def load_recommendations(
    recommendations_path: str,
    eval_users: Set[str],
    training_catalogue: Set[str],
    k: int = 10,
) -> pd.DataFrame:
    """
    Load and filter recommendation file.

    Expected columns:
        user_id, business_id, rank, score, is_fallback
    """

    recs = pd.read_csv(
        recommendations_path,
        dtype={"user_id": str, "business_id": str},
    )

    required = {"user_id", "business_id", "rank"}
    missing = required - set(recs.columns)
    if missing:
        raise ValueError(
            f"{recommendations_path} is missing columns: {missing}. "
            f"Available columns: {list(recs.columns)}"
        )

    recs["user_id"] = recs["user_id"].astype(str)
    recs["business_id"] = recs["business_id"].astype(str)
    recs["rank"] = recs["rank"].astype(int)

    recs = recs[
        recs["user_id"].isin(eval_users)
        & recs["business_id"].isin(training_catalogue)
        & (recs["rank"] <= k)
    ].copy()

    recs = recs.sort_values(
        ["user_id", "rank", "business_id"],
        ascending=[True, True, True],
        kind="mergesort",
    ).reset_index(drop=True)

    return recs


def calculate_hr_at_k(
    recommendations: pd.DataFrame,
    relevant_items: Dict[str, Set[str]],
    eval_users: Set[str],
    k: int = 10,
) -> float:
    """
    HR@K:
    For each eval user, 1 if at least one relevant item appears in top-K.
    Average over all eval users.
    """

    grouped_recs = (
        recommendations[recommendations["rank"] <= k]
        .sort_values(["user_id", "rank", "business_id"])
        .groupby("user_id")["business_id"]
        .apply(list)
        .to_dict()
    )

    hits = []

    for user_id in sorted(eval_users):
        recommended = set(grouped_recs.get(user_id, []))
        relevant = relevant_items.get(user_id, set())

        hit = 1.0 if recommended.intersection(relevant) else 0.0
        hits.append(hit)

    return float(np.mean(hits)) if hits else 0.0


def calculate_ndcg_at_k(
    recommendations: pd.DataFrame,
    relevant_items: Dict[str, Set[str]],
    eval_users: Set[str],
    k: int = 10,
) -> float:
    """
    NDCG@K with binary relevance.
    """

    grouped_recs = (
        recommendations[recommendations["rank"] <= k]
        .sort_values(["user_id", "rank", "business_id"])
        .groupby("user_id")["business_id"]
        .apply(list)
        .to_dict()
    )

    ndcg_values = []

    for user_id in sorted(eval_users):
        ranked_items = grouped_recs.get(user_id, [])
        relevant = relevant_items.get(user_id, set())

        dcg = 0.0

        for rank_index, business_id in enumerate(ranked_items[:k], start=1):
            rel = 1.0 if business_id in relevant else 0.0
            dcg += rel / math.log2(rank_index + 1)

        ideal_hits = min(k, len(relevant))

        idcg = sum(
            1.0 / math.log2(rank_index + 1)
            for rank_index in range(1, ideal_hits + 1)
        )

        ndcg = dcg / idcg if idcg > 0.0 else 0.0
        ndcg_values.append(ndcg)

    return float(np.mean(ndcg_values)) if ndcg_values else 0.0


def compute_rmse_from_predictions(
    predictions_path: str,
    eval_users: Set[str],
    training_catalogue: Set[str],
) -> float:
    """
    Compute RMSE using prediction CSV.

    Expected prediction CSV columns:
        user_id, business_id, stars, predicted_stars

    Catalogue constraint:
        only rows whose business_id is in train catalogue are evaluated.
    """

    predictions = pd.read_csv(
        predictions_path,
        dtype={"user_id": str, "business_id": str},
    )

    required = {"user_id", "business_id", "stars", "predicted_stars"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(
            f"{predictions_path} is missing columns: {missing}. "
            f"Available columns: {list(predictions.columns)}"
        )

    predictions["user_id"] = predictions["user_id"].astype(str)
    predictions["business_id"] = predictions["business_id"].astype(str)
    predictions["stars"] = predictions["stars"].astype(float)
    predictions["predicted_stars"] = predictions["predicted_stars"].astype(float)

    eval_df = predictions[
        predictions["user_id"].isin(eval_users)
        & predictions["business_id"].isin(training_catalogue)
    ].copy()

    return rmse(
        eval_df["stars"].to_numpy(dtype=np.float64),
        eval_df["predicted_stars"].to_numpy(dtype=np.float64),
    )


def evaluate_single_model(
    model_name: str,
    info: Dict[str, Any],
    eval_users: Set[str],
    training_catalogue: Set[str],
    relevant_items: Dict[str, Set[str]],
    k: int = 10,
) -> Dict[str, float]:
    """
    Evaluate one model/baseline using its prediction and recommendation files.
    """

    if "dev_predictions_path" not in info:
        raise ValueError(f"{model_name} missing dev_predictions_path in manifest.")

    if "dev_recommendations_path" not in info:
        raise ValueError(f"{model_name} missing dev_recommendations_path in manifest.")

    predictions_path = info["dev_predictions_path"]
    recommendations_path = info["dev_recommendations_path"]

    model_rmse = compute_rmse_from_predictions(
        predictions_path=predictions_path,
        eval_users=eval_users,
        training_catalogue=training_catalogue,
    )

    recommendations = load_recommendations(
        recommendations_path=recommendations_path,
        eval_users=eval_users,
        training_catalogue=training_catalogue,
        k=k,
    )

    hr = calculate_hr_at_k(
        recommendations=recommendations,
        relevant_items=relevant_items,
        eval_users=eval_users,
        k=k,
    )

    ndcg = calculate_ndcg_at_k(
        recommendations=recommendations,
        relevant_items=relevant_items,
        eval_users=eval_users,
        k=k,
    )

    return {
        "rmse": float(model_rmse),
        f"hr@{k}": float(hr),
        f"ndcg@{k}": float(ndcg),
    }


def load_baseline_manifest(manifest_path: str) -> Dict[str, Any]:
    """
    Load baseline_manifest.json generated by hybrid_model.py.
    """

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    required_models = {
        "Popularity-Based",
        "Ratings-Only SVD",
        "Aspect Content-Only",
        "VADER-Hybrid",
    }

    missing_models = required_models - set(manifest.keys())
    if missing_models:
        raise ValueError(
            f"baseline_manifest.json is missing required models: {missing_models}. "
            f"Available models: {list(manifest.keys())}"
        )

    return manifest


def evaluate_all_models(
    train_reviews_path: str,
    dev_users_path: str,
    dev_qrels_path: str,
    baseline_manifest_path: str,
    output_path: str,
    k: int = 10,
) -> Dict[str, Dict[str, float]]:
    """
    Build final Assignment-required dev_metrics.json.
    """

    eval_users = load_eval_users(dev_users_path)
    training_catalogue = load_training_catalogue(train_reviews_path)

    relevant_items = load_qrels(
        qrels_path=dev_qrels_path,
        eval_users=eval_users,
        training_catalogue=training_catalogue,
    )

    manifest = load_baseline_manifest(baseline_manifest_path)

    # Keep required output order readable.
    model_order = [
        "Popularity-Based",
        "Ratings-Only SVD",
        "Aspect Content-Only",
        "VADER-Hybrid",
    ]

    # Include any bonus models too, after required ones.
    extra_models = [
        model_name
        for model_name in manifest.keys()
        if model_name not in model_order
    ]

    final_order = model_order + sorted(extra_models)

    metrics: Dict[str, Dict[str, float]] = {}

    for model_name in final_order:
        info = manifest[model_name]

        print(f"Evaluating {model_name}...")

        metrics[model_name] = evaluate_single_model(
            model_name=model_name,
            info=info,
            eval_users=eval_users,
            training_catalogue=training_catalogue,
            relevant_items=relevant_items,
            k=k,
        )

        print(
            f"{model_name}: "
            f"RMSE={metrics[model_name]['rmse']:.6f}, "
            f"HR@{k}={metrics[model_name][f'hr@{k}']:.6f}, "
            f"NDCG@{k}={metrics[model_name][f'ndcg@{k}']:.6f}"
        )

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with output_file.open("w", encoding="utf-8") as f:
        json.dump(
            metrics,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"Saved dev metrics to: {output_path}")

    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Assignment 3 dev_metrics.json from generated model outputs."
    )

    parser.add_argument(
        "--train-reviews",
        required=True,
        help="Path to HW3_data/train_reviews.csv",
    )

    parser.add_argument(
        "--dev-users",
        required=True,
        help="Path to HW3_data/dev_users.csv",
    )

    parser.add_argument(
        "--dev-qrels",
        required=True,
        help="Path to HW3_data/dev_qrels.tsv",
    )

    parser.add_argument(
        "--baseline-manifest",
        required=True,
        help="Path to results/baseline_manifest.json generated by hybrid_model.py",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Where to save results/dev_metrics.json",
    )

    parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="K for HR@K and NDCG@K",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    metrics = evaluate_all_models(
        train_reviews_path=args.train_reviews,
        dev_users_path=args.dev_users,
        dev_qrels_path=args.dev_qrels,
        baseline_manifest_path=args.baseline_manifest,
        output_path=args.output,
        k=args.k,
    )

    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
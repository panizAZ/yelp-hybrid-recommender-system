import argparse
import json
import platform
import random
import time
from dataclasses import dataclass, asdict
from typing import Dict, Tuple, Optional
from typing import Any, List

import numpy as np
import pandas as pd
import os


CANONICAL_ASPECTS = ["food", "service", "ambiance", "price", "location"]
# python -P hybrid_model.py `
#   --seed 42 `
#   --alpha 0.7 `
#   --latent_dim 32 `
#   --epochs 80 `
#   --learning_rate 0.005 `
#   --regularization 0.02 `
#   --patience 5 `
#   --min_train_reviews 5 `
#   --tau 0.30 `
#   --preprocessing_summary_path HW3_data/preprocessing_summary.json `
#   --aspect_summary_path HW3_data/aspect_summary.json


def set_seed(seed: int) -> None:
    """
    Fix all pseudo-random sources.
    """
    random.seed(seed)
    np.random.seed(seed)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute Root Mean Squared Error.
    """
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


@dataclass
class MFConfig:
    implementation: str = "Biased Matrix Factorization with SGD"
    latent_dim: int = 32
    learning_rate: float = 0.005
    regularization: float = 0.02
    epochs: int = 30
    seed: int = 42
    alpha: float = 0.7
    patience: int = 8

    # 3.5 parameters
    min_train_reviews: int = 5
    tau: float = 0.30


class BiasedMatrixFactorization:
    """
    Classical Biased Matrix Factorization trained with SGD.

    ScoreCF(u, i) = mu + b_u + b_i + p_u^T q_i
    """

    def __init__(
        self,
        latent_dim: int = 32,
        learning_rate: float = 0.005,
        regularization: float = 0.02,
        epochs: int = 30,
        seed: int = 42,
        verbose: bool = True,
    ):
        self.latent_dim = latent_dim
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.epochs = epochs
        self.seed = seed
        self.verbose = verbose

        self.global_mean: float = 0.0

        self.user_to_idx: Dict[str, int] = {}
        self.item_to_idx: Dict[str, int] = {}
        self.idx_to_user: Dict[int, str] = {}
        self.idx_to_item: Dict[int, str] = {}

        self.user_bias: Optional[np.ndarray] = None
        self.item_bias: Optional[np.ndarray] = None
        self.user_factors: Optional[np.ndarray] = None
        self.item_factors: Optional[np.ndarray] = None

    def fit(
            self,
            ratings_df: pd.DataFrame,
            dev_df: Optional[pd.DataFrame] = None,
            U: Optional[pd.DataFrame] = None,
            V: Optional[pd.DataFrame] = None,
            alpha: float = 0.7,
            patience: Optional[int] = None,
    ) -> None:
        """
        Train Biased Matrix Factorization using SGD.

        If dev_df, U, V, and patience are provided, early stopping is applied
        based on dev hybrid RMSE.
        """

        set_seed(self.seed)

        required_cols = {"user_id", "business_id", "stars"}
        missing_cols = required_cols - set(ratings_df.columns)
        if missing_cols:
            raise ValueError(f"Missing columns in train dataframe: {missing_cols}")

        # Aggregate duplicate (user, item) pairs using arithmetic mean.
        data = (
            ratings_df[["user_id", "business_id", "stars"]]
            .copy()
            .groupby(["user_id", "business_id"], as_index=False)["stars"]
            .mean()
        )

        data["user_id"] = data["user_id"].astype(str)
        data["business_id"] = data["business_id"].astype(str)
        data["stars"] = data["stars"].astype(float)

        users = sorted(data["user_id"].unique())
        items = sorted(data["business_id"].unique())

        self.user_to_idx = {u: idx for idx, u in enumerate(users)}
        self.item_to_idx = {i: idx for idx, i in enumerate(items)}
        self.idx_to_user = {idx: u for u, idx in self.user_to_idx.items()}
        self.idx_to_item = {idx: i for i, idx in self.item_to_idx.items()}

        n_users = len(users)
        n_items = len(items)

        self.global_mean = float(data["stars"].mean())

        rng = np.random.default_rng(self.seed)

        self.user_bias = np.zeros(n_users, dtype=np.float64)
        self.item_bias = np.zeros(n_items, dtype=np.float64)

        self.user_factors = rng.normal(
            loc=0.0,
            scale=0.01,
            size=(n_users, self.latent_dim),
        ).astype(np.float64)

        self.item_factors = rng.normal(
            loc=0.0,
            scale=0.01,
            size=(n_items, self.latent_dim),
        ).astype(np.float64)

        user_indices = data["user_id"].map(self.user_to_idx).to_numpy()
        item_indices = data["business_id"].map(self.item_to_idx).to_numpy()
        ratings = data["stars"].to_numpy(dtype=np.float64)

        n_obs = len(ratings)

        use_early_stopping = (
                dev_df is not None
                and U is not None
                and V is not None
                and patience is not None
                and patience > 0
        )

        self.training_history = []
        self.best_epoch = None
        self.best_dev_rmse = None

        best_dev_rmse = float("inf")
        best_state = None
        epochs_without_improvement = 0

        if self.verbose:
            print(f"Training Biased MF")
            print(f"Users: {n_users}")
            print(f"Items: {n_items}")
            print(f"Observed user-item ratings after aggregation: {n_obs}")
            print(f"Global mean rating: {self.global_mean:.4f}")

            if use_early_stopping:
                print(f"Early stopping: enabled")
                print(f"Patience: {patience}")
                print(f"Early stopping metric: dev hybrid RMSE")
            else:
                print(f"Early stopping: disabled")

        for epoch in range(1, self.epochs + 1):
            order = rng.permutation(n_obs)

            total_squared_error = 0.0

            for idx in order:
                u_idx = user_indices[idx]
                i_idx = item_indices[idx]
                r_ui = ratings[idx]

                pred = (
                        self.global_mean
                        + self.user_bias[u_idx]
                        + self.item_bias[i_idx]
                        + np.dot(self.user_factors[u_idx], self.item_factors[i_idx])
                )

                err = r_ui - pred
                total_squared_error += err ** 2

                # Store old vectors before update.
                p_u_old = self.user_factors[u_idx].copy()
                q_i_old = self.item_factors[i_idx].copy()

                lr = self.learning_rate
                reg = self.regularization

                # Bias updates.
                self.user_bias[u_idx] += lr * (err - reg * self.user_bias[u_idx])
                self.item_bias[i_idx] += lr * (err - reg * self.item_bias[i_idx])

                # Latent factor updates.
                self.user_factors[u_idx] += lr * (
                        err * q_i_old - reg * p_u_old
                )
                self.item_factors[i_idx] += lr * (
                        err * p_u_old - reg * q_i_old
                )

            train_rmse = float(np.sqrt(total_squared_error / n_obs))

            dev_rmse = None

            if use_early_stopping:
                dev_true = []
                dev_pred = []

                for row in dev_df.itertuples(index=False):
                    user_id = getattr(row, "user_id")
                    business_id = getattr(row, "business_id")
                    true_rating = float(getattr(row, "stars"))

                    (
                        _cf_raw,
                        _cf_norm,
                        _asp_raw,
                        _asp_norm,
                        hybrid_norm,
                    ) = hybrid_predict_norm(
                        user_id=user_id,
                        business_id=business_id,
                        mf_model=self,
                        U=U,
                        V=V,
                        alpha=alpha,
                    )

                    predicted_rating = norm_to_stars(hybrid_norm)

                    dev_true.append(true_rating)
                    dev_pred.append(predicted_rating)

                dev_rmse = rmse(
                    np.array(dev_true, dtype=np.float64),
                    np.array(dev_pred, dtype=np.float64),
                )

                improved = dev_rmse < best_dev_rmse

                if improved:
                    best_dev_rmse = dev_rmse
                    self.best_dev_rmse = float(best_dev_rmse)
                    self.best_epoch = int(epoch)

                    best_state = {
                        "user_bias": self.user_bias.copy(),
                        "item_bias": self.item_bias.copy(),
                        "user_factors": self.user_factors.copy(),
                        "item_factors": self.item_factors.copy(),
                    }

                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1

                self.training_history.append(
                    {
                        "epoch": int(epoch),
                        "train_rmse": float(train_rmse),
                        "dev_rmse": float(dev_rmse),
                        "best_dev_rmse": float(best_dev_rmse),
                        "epochs_without_improvement": int(epochs_without_improvement),
                    }
                )

                if self.verbose:
                    marker = " *best*" if improved else ""
                    print(
                        f"Epoch {epoch:03d}/{self.epochs} - "
                        f"train RMSE: {train_rmse:.4f} - "
                        f"dev hybrid RMSE: {dev_rmse:.4f}"
                        f"{marker}"
                    )

                if epochs_without_improvement >= patience:
                    if self.verbose:
                        print(
                            f"Early stopping at epoch {epoch}. "
                            f"Best dev hybrid RMSE: {best_dev_rmse:.4f} "
                            f"at epoch {self.best_epoch}."
                        )
                    break

            else:
                self.training_history.append(
                    {
                        "epoch": int(epoch),
                        "train_rmse": float(train_rmse),
                    }
                )

                if self.verbose:
                    print(
                        f"Epoch {epoch:03d}/{self.epochs} - "
                        f"train RMSE: {train_rmse:.4f}"
                    )

        # Restore best model if early stopping was used.
        if use_early_stopping and best_state is not None:
            self.user_bias = best_state["user_bias"]
            self.item_bias = best_state["item_bias"]
            self.user_factors = best_state["user_factors"]
            self.item_factors = best_state["item_factors"]

            if self.verbose:
                print(
                    f"Restored best model from epoch {self.best_epoch} "
                    f"with dev hybrid RMSE: {self.best_dev_rmse:.4f}"
                )

    def predict_raw(self, user_id: str, business_id: str) -> float:
        """
        Predict ScoreCF(u, i) on the original 1-5 rating scale.

        For unknown users/items:
            - unknown user and known item: mu + b_i
            - known user and unknown item: mu + b_u
            - both unknown: mu
        """

        user_id = str(user_id)
        business_id = str(business_id)

        pred = self.global_mean

        if user_id in self.user_to_idx:
            u_idx = self.user_to_idx[user_id]
            pred += self.user_bias[u_idx]
        else:
            u_idx = None

        if business_id in self.item_to_idx:
            i_idx = self.item_to_idx[business_id]
            pred += self.item_bias[i_idx]
        else:
            i_idx = None

        if u_idx is not None and i_idx is not None:
            pred += float(np.dot(self.user_factors[u_idx], self.item_factors[i_idx]))

        return float(pred)

    def predict_norm(self, user_id: str, business_id: str) -> float:
        """
        ScoreCF_norm(u, i) = (clip(ScoreCF(u, i), 1.0, 5.0) - 1.0) / 4.0
        """
        raw = self.predict_raw(user_id, business_id)
        clipped = np.clip(raw, 1.0, 5.0)
        return float((clipped - 1.0) / 4.0)


def load_aspect_matrices(
    u_path: str,
    v_path: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load U and V matrices.

    U:
        index = user_id
        columns = food, service, ambiance, price, location

    V:
        index = business_id
        columns = food, service, ambiance, price, location
    """

    U = pd.read_csv(u_path, index_col=0)
    V = pd.read_csv(v_path, index_col=0)

    # Ensure index is string to match user_id/business_id reliably.
    U.index = U.index.astype(str)
    V.index = V.index.astype(str)

    # Ensure all canonical aspect columns exist.
    for aspect in CANONICAL_ASPECTS:
        if aspect not in U.columns:
            raise ValueError(f"Missing aspect column in U: {aspect}")
        if aspect not in V.columns:
            raise ValueError(f"Missing aspect column in V: {aspect}")

    U = U[CANONICAL_ASPECTS].astype(float)
    V = V[CANONICAL_ASPECTS].astype(float)

    if not np.isfinite(U.to_numpy()).all():
        raise ValueError("U contains NaN or infinite values.")

    if not np.isfinite(V.to_numpy()).all():
        raise ValueError("V contains NaN or infinite values.")

    if (U.to_numpy() < 0).any():
        raise ValueError(
            "User attention matrix U must be non-negative according to Assignment 3."
        )

    return U, V


def score_aspect_raw(
    user_id: str,
    business_id: str,
    U: pd.DataFrame,
    V: pd.DataFrame,
) -> float:
    """
    ScoreAspect(u, i) = sum_a U[u,a] * V[i,a]

    If user has no mapped training aspects, define ScoreAspect = 0.
    If user or item is unknown, also use ScoreAspect = 0.
    """

    user_id = str(user_id)
    business_id = str(business_id)

    if user_id not in U.index:
        return 0.0

    if business_id not in V.index:
        return 0.0

    u_vec = U.loc[user_id, CANONICAL_ASPECTS].to_numpy(dtype=np.float64)
    v_vec = V.loc[business_id, CANONICAL_ASPECTS].to_numpy(dtype=np.float64)

    # If user has no mapped training aspects, U row is all zero.
    if np.isclose(u_vec.sum(), 0.0):
        return 0.0

    score = float(np.dot(u_vec, v_vec))

    # Mathematically should be in [-1, 1], but clip for numerical safety.
    score = float(np.clip(score, -1.0, 1.0))

    return score


def score_aspect_norm(
    user_id: str,
    business_id: str,
    U: pd.DataFrame,
    V: pd.DataFrame,
) -> float:
    """
    ScoreAspect_norm(u, i) = (ScoreAspect(u, i) + 1.0) / 2.0
    """
    raw = score_aspect_raw(user_id, business_id, U, V)
    return float((raw + 1.0) / 2.0)


def hybrid_predict_norm(
    user_id: str,
    business_id: str,
    mf_model: BiasedMatrixFactorization,
    U: pd.DataFrame,
    V: pd.DataFrame,
    alpha: float,
) -> Tuple[float, float, float, float, float]:
    """
    Return:
        score_cf_raw
        score_cf_norm
        score_aspect_raw
        score_aspect_norm
        hybrid_norm
    """

    score_cf_raw = mf_model.predict_raw(user_id, business_id)
    score_cf_norm = mf_model.predict_norm(user_id, business_id)

    score_aspect_raw_value = score_aspect_raw(user_id, business_id, U, V)
    score_aspect_norm_value = (score_aspect_raw_value + 1.0) / 2.0

    hybrid_norm = (
        alpha * score_cf_norm
        + (1.0 - alpha) * score_aspect_norm_value
    )

    hybrid_norm = float(np.clip(hybrid_norm, 0.0, 1.0))

    return (
        float(score_cf_raw),
        float(score_cf_norm),
        float(score_aspect_raw_value),
        float(score_aspect_norm_value),
        float(hybrid_norm),
    )


def norm_to_stars(score_norm: float) -> float:
    """
    R_hat_stars = 1.0 + 4.0 * R_hat_norm
    """
    return float(1.0 + 4.0 * score_norm)


def run_hybrid_scoring(
    train_path: str,
    dev_path: str,
    u_path: str,
    v_path: str,
    output_predictions_path: str,
    summary_path: str,
    config: MFConfig,
) -> Tuple[
    BiasedMatrixFactorization,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    float,
]:
    """
    Train MF on train reviews, combine with aspect scores, evaluate on dev.
    """

    set_seed(config.seed)

    print("Loading train and dev data...")
    train_df = pd.read_csv(train_path)
    dev_df = pd.read_csv(dev_path)

    required_cols = {"user_id", "business_id", "stars"}
    missing_train = required_cols - set(train_df.columns)
    missing_dev = required_cols - set(dev_df.columns)

    if missing_train:
        raise ValueError(f"Missing columns in train file: {missing_train}")
    if missing_dev:
        raise ValueError(f"Missing columns in dev file: {missing_dev}")

    train_df["user_id"] = train_df["user_id"].astype(str)
    train_df["business_id"] = train_df["business_id"].astype(str)
    train_df["stars"] = train_df["stars"].astype(float)

    dev_df["user_id"] = dev_df["user_id"].astype(str)
    dev_df["business_id"] = dev_df["business_id"].astype(str)
    dev_df["stars"] = dev_df["stars"].astype(float)

    print("Loading U and V aspect matrices...")
    U, V = load_aspect_matrices(u_path, v_path)

    print("Training collaborative filtering model...")
    mf_model = BiasedMatrixFactorization(
        latent_dim=config.latent_dim,
        learning_rate=config.learning_rate,
        regularization=config.regularization,
        epochs=config.epochs,
        seed=config.seed,
        verbose=True,
    )
    mf_model.fit(
        ratings_df=train_df,
        dev_df=dev_df,
        U=U,
        V=V,
        alpha=config.alpha,
        patience=config.patience,
    )

    print("Predicting dev reviews...")

    score_cf_raw_list = []
    score_cf_norm_list = []
    score_aspect_raw_list = []
    score_aspect_norm_list = []
    hybrid_norm_list = []
    hybrid_stars_list = []

    for row in dev_df.itertuples(index=False):
        user_id = getattr(row, "user_id")
        business_id = getattr(row, "business_id")

        (
            cf_raw,
            cf_norm,
            asp_raw,
            asp_norm,
            hybrid_norm,
        ) = hybrid_predict_norm(
            user_id=user_id,
            business_id=business_id,
            mf_model=mf_model,
            U=U,
            V=V,
            alpha=config.alpha,
        )

        hybrid_stars = norm_to_stars(hybrid_norm)

        score_cf_raw_list.append(cf_raw)
        score_cf_norm_list.append(cf_norm)
        score_aspect_raw_list.append(asp_raw)
        score_aspect_norm_list.append(asp_norm)
        hybrid_norm_list.append(hybrid_norm)
        hybrid_stars_list.append(hybrid_stars)

    predictions_df = dev_df.copy()
    predictions_df["score_cf_raw"] = score_cf_raw_list
    predictions_df["score_cf_norm"] = score_cf_norm_list
    predictions_df["score_aspect_raw"] = score_aspect_raw_list
    predictions_df["score_aspect_norm"] = score_aspect_norm_list
    predictions_df["hybrid_score_norm"] = hybrid_norm_list
    predictions_df["predicted_stars"] = hybrid_stars_list

    y_true = predictions_df["stars"].to_numpy(dtype=np.float64)
    y_pred = predictions_df["predicted_stars"].to_numpy(dtype=np.float64)

    dev_rmse = rmse(y_true, y_pred)

    print(f"Dev RMSE: {dev_rmse:.4f}")

    print(f"Saving predictions to: {output_predictions_path}")
    predictions_df.to_csv(output_predictions_path, index=False)

    summary = {
        "section": "3.4 Hybrid Scoring & Matrix Factorization",
        "implementation": config.implementation,
        "latent_dimension": config.latent_dim,
        "optimizer": "Stochastic Gradient Descent",
        "learning_rate": config.learning_rate,
        "regularization": config.regularization,
        "epochs": config.epochs,
        "patience": config.patience,
        "seed": config.seed,
        "alpha": config.alpha,
        "best_epoch": mf_model.best_epoch,
        "best_dev_rmse_during_training": mf_model.best_dev_rmse,
        "training_history": mf_model.training_history,
        "global_mean_rating": mf_model.global_mean,
        "normalization": {
            "ScoreCF_norm": "(clip(ScoreCF, 1.0, 5.0) - 1.0) / 4.0",
            "ScoreAspect_norm": "(ScoreAspect + 1.0) / 2.0",
            "Hybrid_norm": "alpha * ScoreCF_norm + (1-alpha) * ScoreAspect_norm",
            "Rating_rescaling": "1.0 + 4.0 * Hybrid_norm",
        },
        "train": {
            "num_train_review_rows": int(len(train_df)),
            "num_unique_train_users": int(train_df["user_id"].nunique()),
            "num_unique_train_businesses": int(train_df["business_id"].nunique()),
        },
        "dev": {
            "num_dev_review_rows": int(len(dev_df)),
            "rmse": dev_rmse,
        },
        "aspect_files": {
            "U": u_path,
            "V": v_path,
        },
        "output_predictions": output_predictions_path,
    }

    print(f"Saving summary to: {summary_path}")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)

    print("Done.")

    return mf_model, train_df, dev_df, U, V, dev_rmse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Part 3.4 Hybrid Scoring & Matrix Factorization"
    )

    parser.add_argument(
        "--train_path",
        type=str,
        default="HW3_data/train_reviews.csv",
        help="Path to train_reviews.csv",
    )

    parser.add_argument(
        "--dev_path",
        type=str,
        default="HW3_data/dev_reviews.csv",
        help="Path to dev_reviews.csv",
    )

    parser.add_argument(
        "--u_path",
        type=str,
        default="HW3_data/user_attention_matrix_U.csv",
        help="Path to user_attention_matrix_U.csv",
    )

    parser.add_argument(
        "--v_path",
        type=str,
        default="HW3_data/item_quality_matrix_V.csv",
        help="Path to item_quality_matrix_V.csv",
    )

    parser.add_argument(
        "--output_predictions_path",
        type=str,
        default="HW3_data/dev_predictions_hybrid.csv",
        help="Where to save dev predictions",
    )

    parser.add_argument(
        "--summary_path",
        type=str,
        default="results/summary.json",
        help="Where to save summary.json",
    )

    parser.add_argument(
        "--latent_dim",
        type=int,
        default=32,
        help="Latent factor dimension k",
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.005,
        help="SGD learning rate",
    )

    parser.add_argument(
        "--regularization",
        type=float,
        default=0.02,
        help="L2 regularization strength",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Number of SGD epochs",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=0.7,
        help="Hybrid blending weight for CF. alpha=0.7 means 70% CF and 30% Aspect.",
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=8,
        help="Number of epochs to wait without dev RMSE improvement before early stopping",
    )

    parser.add_argument(
        "--min_train_reviews",
        type=int,
        default=5,
        help="Minimum number of training-review rows before trusting the model",
    )

    parser.add_argument(
        "--tau",
        type=float,
        default=0.30,
        help="Abstention threshold on normalized hybrid score",
    )

    parser.add_argument(
        "--dev_users_path",
        type=str,
        default="HW3_data/dev_users.csv",
        help="Path to dev_users.csv for Top-K recommendation",
    )

    parser.add_argument(
        "--test_users_path",
        type=str,
        default="HW3_data/test_users.csv",
        help="Path to test_users.csv for Top-K recommendation",
    )

    parser.add_argument(
        "--output_dev_path",
        type=str,
        default="results/dev_recommendations.csv",
        help="Where to save dev_recommendations.csv",
    )

    parser.add_argument(
        "--output_test_path",
        type=str,
        default="results/test_recommendations.csv",
        help="Where to save test_recommendations.csv",
    )

    parser.add_argument(
        "--output_trace_path",
        type=str,
        default="results/trace.jsonl",
        help="Where to save trace.jsonl",
    )

    parser.add_argument(
        "--outdir",
        type=str,
        default="results",
        help="Directory for all generated output files",
    )

    parser.add_argument(
        "--preprocessing_summary_path",
        type=str,
        default="HW3_data/preprocessing_summary.json",
        help="Path to preprocessing_summary.json generated by text_processor.py",
    )

    parser.add_argument(
        "--aspect_summary_path",
        type=str,
        default="results/aspect_summary.json",
        help="Optional path to aspect_summary.json generated by sentiment/aspect extraction",
    )

    parser.add_argument(
        "--business_path",
        type=str,
        default="HW3_data/businesses.jsonl",
        help="Path to business metadata JSONL file containing business_id and name.",
    )

    return parser.parse_args()


# Recommendation
def load_user_ids(users_path: str) -> List[str]:
    """
    Load target users from dev_users.csv or test_users.csv.

    The file must contain a user_id column.
    """
    users_df = pd.read_csv(users_path)

    if "user_id" not in users_df.columns:
        raise ValueError(
            f"{users_path} must contain a 'user_id' column. "
            f"Available columns: {list(users_df.columns)}"
        )

    return (
        users_df["user_id"]
        .astype(str)
        .drop_duplicates()
        .tolist()
    )


def build_training_statistics(
    train_df: pd.DataFrame,
) -> Tuple[List[str], Dict[str, set], Dict[str, int], int]:
    """
    Build:

    catalogue:
        Businesses with at least one training review.

    user_seen_items:
        Businesses reviewed by each user in training.

    item_popularity:
        Number of training-review rows for each business.

    max_catalogue_popularity:
        Maximum business popularity in the catalogue.
    """
    train = train_df.copy()

    train["user_id"] = train["user_id"].astype(str)
    train["business_id"] = train["business_id"].astype(str)

    # Catalogue: every business with at least one training review.
    catalogue = sorted(train["business_id"].unique().tolist())

    # Items already reviewed by each user.
    user_seen_items = (
        train.groupby("user_id")["business_id"]
        .apply(set)
        .to_dict()
    )

    # IMPORTANT:
    # Assignment says training-review rows, so do not aggregate duplicates here.
    item_popularity = (
        train["business_id"]
        .value_counts()
        .astype(int)
        .to_dict()
    )

    max_catalogue_popularity = max(
        item_popularity[business_id]
        for business_id in catalogue
    )

    return (
        catalogue,
        user_seen_items,
        item_popularity,
        max_catalogue_popularity,
    )


def extract_user_aspect_preferences(
    user_id: str,
    U: pd.DataFrame,
) -> Dict[str, float]:
    """
    Return the user's U-vector as a JSON-serializable dictionary.

    If the user is absent from U, all aspects are set to zero.
    """
    user_id = str(user_id)

    if user_id not in U.index:
        return {
            aspect: 0.0
            for aspect in CANONICAL_ASPECTS
        }

    vector = U.loc[
        user_id,
        CANONICAL_ASPECTS,
    ].astype(float)

    return {
        aspect: float(vector[aspect])
        for aspect in CANONICAL_ASPECTS
    }


def load_business_name_map(business_path: str) -> Dict[str, Optional[str]]:
    """
    Load business_id -> business name mapping from a JSONL business metadata file.

    Expected JSONL format per line:
        {"business_id": "...", "name": "...", ...}
    """
    business_name_map: Dict[str, Optional[str]] = {}

    if business_path is None:
        return business_name_map

    if not os.path.exists(business_path):
        print(
            f"Warning: business metadata file not found at {business_path}. "
            "Trace business_name values will be null."
        )
        return business_name_map

    with open(business_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()

            if not line:
                continue

            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue

            business_id = str(row.get("business_id", "")).strip()
            business_name = row.get("name")

            if business_id:
                business_name_map[business_id] = business_name

    print(
        f"Loaded {len(business_name_map):,} business names from {business_path}"
    )

    return business_name_map


def calculate_recommendations(
    target_user_ids: List[str],
    train_df: pd.DataFrame,
    mf_model: BiasedMatrixFactorization,
    U: pd.DataFrame,
    V: pd.DataFrame,
    business_name_map: Optional[Dict[str, Optional[str]]] = None,
    alpha: float = 0.7,
    min_train_reviews: int = 5,
    tau: float = 0.30,
    top_k: int = 10,
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """
    Generate Top-K recommendations and trace records efficiently.

    This implementation is vectorized across all catalogue items for each user.
    It preserves the exact Section 3.5 logic:

    - Catalogue: businesses with at least one training review.
    - Candidates: catalogue businesses not reviewed by the user in training.
    - Normal ranking: hybrid score descending, business_id ascending for ties.
    - Fallback ranking: popularity descending, business_id ascending for ties.
    - Fallback trigger:
        user_train_review_count < min_train_reviews
        OR max hybrid candidate score < tau.
    """

    if business_name_map is None:
        business_name_map = {}

    (
        catalogue,
        user_seen_items,
        item_popularity,
        max_catalogue_popularity,
    ) = build_training_statistics(train_df)

    if len(catalogue) == 0:
        raise ValueError("The training catalogue is empty.")

    # ------------------------------------------------------------------
    # Precompute catalogue-level arrays once.
    # ------------------------------------------------------------------
    catalogue_ids = np.asarray(catalogue, dtype=str)
    n_catalogue = len(catalogue_ids)

    catalogue_id_to_pos = {
        business_id: position
        for position, business_id in enumerate(catalogue_ids)
    }

    popularity_array = np.asarray(
        [item_popularity[business_id] for business_id in catalogue_ids],
        dtype=np.float64,
    )

    mf_item_indices = np.asarray(
        [
            mf_model.item_to_idx.get(business_id, -1)
            for business_id in catalogue_ids
        ],
        dtype=np.int64,
    )

    known_mf_items = mf_item_indices >= 0

    item_bias_array = np.zeros(n_catalogue, dtype=np.float64)
    item_bias_array[known_mf_items] = mf_model.item_bias[
        mf_item_indices[known_mf_items]
    ]

    item_factors_array = np.zeros(
        (n_catalogue, mf_model.latent_dim),
        dtype=np.float64,
    )
    item_factors_array[known_mf_items] = mf_model.item_factors[
        mf_item_indices[known_mf_items]
    ]

    item_aspect_matrix = np.zeros(
        (n_catalogue, len(CANONICAL_ASPECTS)),
        dtype=np.float64,
    )

    for position, business_id in enumerate(catalogue_ids):
        if business_id in V.index:
            item_aspect_matrix[position] = V.loc[
                business_id,
                CANONICAL_ASPECTS,
            ].to_numpy(dtype=np.float64)

    fallback_global_order = np.lexsort(
        (
            catalogue_ids,
            -popularity_array,
        )
    )

    user_train_review_counts = (
        train_df["user_id"]
        .astype(str)
        .value_counts()
        .astype(int)
        .to_dict()
    )

    recommendation_rows: List[Dict[str, Any]] = []
    trace_records: List[Dict[str, Any]] = []

    print(
        f"Vectorized recommendation scoring for {len(target_user_ids)} users "
        f"over {n_catalogue} catalogue businesses..."
    )

    for user_number, user_id in enumerate(target_user_ids, start=1):
        user_id = str(user_id)

        if user_number % 1000 == 0:
            print(
                f"Processed {user_number:,}/{len(target_user_ids):,} target users..."
            )

        seen_items = user_seen_items.get(user_id, set())
        user_train_review_count = int(
            user_train_review_counts.get(user_id, 0)
        )

        candidate_mask = np.ones(n_catalogue, dtype=bool)

        for seen_business_id in seen_items:
            position = catalogue_id_to_pos.get(seen_business_id)
            if position is not None:
                candidate_mask[position] = False

        candidate_positions = np.flatnonzero(candidate_mask)

        # --------------------------------------------------------------
        # Explicit trace entry for a user with no candidates.
        # Important: do NOT reference top_position, cf_norm_scores,
        # hybrid_scores, use_fallback, etc. here, because they do not exist yet.
        # --------------------------------------------------------------
        if len(candidate_positions) == 0:
            user_aspect_preferences = extract_user_aspect_preferences(user_id, U)

            user_aspect_preference_list = [
                {
                    "aspect": aspect,
                    "score": float(user_aspect_preferences.get(aspect, 0.0)),
                }
                for aspect in CANONICAL_ASPECTS
            ]

            user_aspect_preference_list.sort(
                key=lambda x: (
                    -x["score"],
                    CANONICAL_ASPECTS.index(x["aspect"]),
                )
            )

            trace_records.append(
                {
                    "user_id": user_id,
                    "user_train_review_count": user_train_review_count,
                    "extracted_user_aspect_preferences": user_aspect_preference_list,
                    "top_recommended_item": None,
                    "abstain_fallback_triggered": True,
                    "decision_reason": (
                        "No candidate items available because the user has already "
                        "reviewed every catalogue business in the training data."
                    ),
                }
            )
            continue

        # --------------------------------------------------------------
        # Vectorized Collaborative Filtering score for all catalogue items
        # ScoreCF(u, i) = mu + b_u + b_i + p_u^T q_i
        # --------------------------------------------------------------
        cf_raw_scores = (
            np.full(
                n_catalogue,
                mf_model.global_mean,
                dtype=np.float64,
            )
            + item_bias_array
        )

        if user_id in mf_model.user_to_idx:
            user_idx = mf_model.user_to_idx[user_id]

            cf_raw_scores += mf_model.user_bias[user_idx]
            cf_raw_scores += item_factors_array @ mf_model.user_factors[user_idx]

        cf_norm_scores = (
            np.clip(cf_raw_scores, 1.0, 5.0) - 1.0
        ) / 4.0

        # --------------------------------------------------------------
        # Vectorized Aspect score for all catalogue items
        # ScoreAspect(u, i) = U_u dot V_i
        # --------------------------------------------------------------
        if user_id in U.index:
            user_aspect_vector = U.loc[
                user_id,
                CANONICAL_ASPECTS,
            ].to_numpy(dtype=np.float64)

            if np.isclose(user_aspect_vector.sum(), 0.0):
                aspect_raw_scores = np.zeros(
                    n_catalogue,
                    dtype=np.float64,
                )
            else:
                aspect_raw_scores = item_aspect_matrix @ user_aspect_vector
                aspect_raw_scores = np.clip(
                    aspect_raw_scores,
                    -1.0,
                    1.0,
                )
        else:
            user_aspect_vector = np.zeros(
                len(CANONICAL_ASPECTS),
                dtype=np.float64,
            )
            aspect_raw_scores = np.zeros(
                n_catalogue,
                dtype=np.float64,
            )

        aspect_norm_scores = (aspect_raw_scores + 1.0) / 2.0

        hybrid_scores = (
            alpha * cf_norm_scores
            + (1.0 - alpha) * aspect_norm_scores
        )
        hybrid_scores = np.clip(hybrid_scores, 0.0, 1.0)

        max_candidate_score = float(
            np.max(hybrid_scores[candidate_positions])
        )

        low_review_count = user_train_review_count < min_train_reviews
        low_model_score = max_candidate_score < tau
        use_fallback = low_review_count or low_model_score

        # --------------------------------------------------------------
        # Rank candidate businesses.
        # --------------------------------------------------------------
        if use_fallback:
            ranked_positions = fallback_global_order[
                candidate_mask[fallback_global_order]
            ][:top_k]
        else:
            candidate_scores = hybrid_scores[candidate_positions]

            ranking_order = np.lexsort(
                (
                    catalogue_ids[candidate_positions],
                    -candidate_scores,
                )
            )

            ranked_positions = candidate_positions[
                ranking_order[:top_k]
            ]

        # --------------------------------------------------------------
        # Save recommendation rows.
        # --------------------------------------------------------------
        for rank, position in enumerate(ranked_positions, start=1):
            business_id = str(catalogue_ids[position])

            if use_fallback:
                output_score = float(
                    popularity_array[position] / max_catalogue_popularity
                )
            else:
                output_score = float(hybrid_scores[position])

            recommendation_rows.append(
                {
                    "user_id": user_id,
                    "business_id": business_id,
                    "rank": int(rank),
                    "score": output_score,
                    "is_fallback": bool(use_fallback),
                }
            )

        # --------------------------------------------------------------
        # Top item for trace.
        # --------------------------------------------------------------
        top_position = int(ranked_positions[0])
        top_business_id = str(catalogue_ids[top_position])

        # Convert user aspect preferences from vector/dict-like structure
        # into the sample-style list of objects.
        user_aspect_preference_list = [
            {
                "aspect": aspect,
                "score": float(user_aspect_vector[aspect_index]),
            }
            for aspect_index, aspect in enumerate(CANONICAL_ASPECTS)
        ]

        user_aspect_preference_list.sort(
            key=lambda x: (
                -x["score"],
                CANONICAL_ASPECTS.index(x["aspect"]),
            )
        )

        # Convert item alignment from a plain dictionary to a sample-style
        # list of objects with item_sentiment and textual contribution.
        item_aspect_alignment = []

        for aspect_index, aspect in enumerate(CANONICAL_ASPECTS):
            user_preference = float(user_aspect_vector[aspect_index])
            item_sentiment = float(
                item_aspect_matrix[top_position, aspect_index]
            )
            weighted_contribution = float(user_preference * item_sentiment)

            if abs(weighted_contribution) <= 1e-9:
                contribution = "neutral"
            elif weighted_contribution > 0.0:
                contribution = "positive"
            else:
                contribution = "negative"

            item_aspect_alignment.append(
                {
                    "aspect": aspect,
                    "item_sentiment": item_sentiment,
                    "weighted_contribution": weighted_contribution,
                    "contribution": contribution,
                }
            )

        item_aspect_alignment.sort(
            key=lambda x: (
                -abs(x["weighted_contribution"]),
                CANONICAL_ASPECTS.index(x["aspect"]),
            )
        )

        # Human-readable decision reason.
        if use_fallback:
            fallback_reasons = []

            if low_review_count:
                fallback_reasons.append(
                    f"user has {user_train_review_count} training reviews, "
                    f"below the minimum of {min_train_reviews}"
                )

            if low_model_score:
                fallback_reasons.append(
                    f"best hybrid candidate score {max_candidate_score:.4f} "
                    f"is below threshold {tau:.4f}"
                )

            if len(fallback_reasons) == 0:
                decision_reason = (
                    "Popularity fallback selected because the hybrid model "
                    "could not produce a confident recommendation."
                )
            else:
                decision_reason = (
                    "Popularity fallback selected because "
                    + " and ".join(fallback_reasons)
                    + "."
                )
        else:
            strongest_aspect = item_aspect_alignment[0]["aspect"]
            decision_reason = (
                f"Hybrid score {hybrid_scores[top_position]:.2f} exceeds threshold "
                f"{tau:.4f}; high aspect alignment on '{strongest_aspect}'."
            )

        trace_records.append(
            {
                "user_id": user_id,
                "user_train_review_count": user_train_review_count,
                "extracted_user_aspect_preferences": user_aspect_preference_list,
                "top_recommended_item": {
                    "business_id": top_business_id,
                    "business_name": business_name_map.get(top_business_id),
                    "svd_score_norm": float(cf_norm_scores[top_position]),
                    "aspect_score_norm": float(
                        aspect_norm_scores[top_position]
                    ),
                    "final_hybrid_score": float(
                        hybrid_scores[top_position]
                    ),
                    "item_aspect_alignment": item_aspect_alignment,
                },
                "abstain_fallback_triggered": bool(use_fallback),
                "decision_reason": decision_reason,
            }
        )

    recommendations_df = pd.DataFrame(
        recommendation_rows,
        columns=[
            "user_id",
            "business_id",
            "rank",
            "score",
            "is_fallback",
        ],
    )

    return recommendations_df, trace_records


def save_recommendations(
    recommendations_df: pd.DataFrame,
    output_path: str,
) -> None:
    """
    Save recommendation CSV with exact required schema.
    """

    output_df = recommendations_df.copy()

    output_df["user_id"] = output_df["user_id"].astype(str)
    output_df["business_id"] = output_df["business_id"].astype(str)
    output_df["rank"] = output_df["rank"].astype(int)

    output_df["is_fallback"] = output_df["is_fallback"].map(
        lambda value: "true" if bool(value) else "false"
    )

    # Assignment requires user_id ascending, then rank ascending.
    output_df = output_df.sort_values(
        by=["user_id", "rank"],
        ascending=[True, True],
        kind="mergesort",
    ).reset_index(drop=True)

    output_df = output_df[
        [
            "user_id",
            "business_id",
            "rank",
            "score",
            "is_fallback",
        ]
    ]

    output_df.to_csv(
        output_path,
        index=False,
        float_format="%.6f",
    )


def save_trace_jsonl(
    trace_records: List[Dict[str, Any]],
    output_path: str,
) -> None:
    """
    Save one valid JSON object per line.
    """
    with open(output_path, "w", encoding="utf-8") as file:
        for record in trace_records:
            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )

def run_top_k_recommendation(
    train_df: pd.DataFrame,
    mf_model: BiasedMatrixFactorization,
    U: pd.DataFrame,
    V: pd.DataFrame,
    business_name_map: Optional[Dict[str, Optional[str]]],
    dev_users_path: str,
    test_users_path: str,
    output_dev_path: str,
    output_test_path: str,
    output_trace_path: str,
    config: MFConfig,
) -> None:
    """
    Generate:

    1. dev_recommendations.csv
    2. test_recommendations.csv
    3. trace.jsonl for the first 20 lexicographically sorted test users

    Important:
    This function does NOT train the model.
    It reuses the already trained mf_model from run_hybrid_scoring().
    """

    print("Generating Top-K recommendations using the already trained model...")

    train_df = train_df.copy()
    train_df["user_id"] = train_df["user_id"].astype(str)
    train_df["business_id"] = train_df["business_id"].astype(str)
    train_df["stars"] = train_df["stars"].astype(float)

    dev_user_ids = load_user_ids(dev_users_path)
    test_user_ids = load_user_ids(test_users_path)

    dev_recommendations, _ = calculate_recommendations(
        target_user_ids=dev_user_ids,
        train_df=train_df,
        mf_model=mf_model,
        U=U,
        V=V,
        business_name_map=business_name_map,
        alpha=config.alpha,
        min_train_reviews=config.min_train_reviews,
        tau=config.tau,
        top_k=10,
    )

    test_recommendations, trace_records = calculate_recommendations(
        target_user_ids=test_user_ids,
        train_df=train_df,
        mf_model=mf_model,
        U=U,
        V=V,
        business_name_map=business_name_map,
        alpha=config.alpha,
        min_train_reviews=config.min_train_reviews,
        tau=config.tau,
        top_k=10,
    )

    save_recommendations(
        recommendations_df=dev_recommendations,
        output_path=output_dev_path,
    )

    save_recommendations(
        recommendations_df=test_recommendations,
        output_path=output_test_path,
    )

    # Trace is required for first 20 test users in lexicographic order.
    test_user_ids_for_trace = sorted(test_user_ids)[:20]

    trace_by_user = {
        record["user_id"]: record
        for record in trace_records
    }

    selected_trace_records = [
        trace_by_user[user_id]
        for user_id in test_user_ids_for_trace
        if user_id in trace_by_user
    ]

    save_trace_jsonl(
        trace_records=selected_trace_records,
        output_path=output_trace_path,
    )

    print(f"Saved dev recommendations to: {output_dev_path}")
    print(f"Saved test recommendations to: {output_test_path}")
    print(f"Saved trace to: {output_trace_path}")


# Baselines
# ============================================================
# Baselines and model variants
# ============================================================

def ensure_parent_dir(path: str) -> None:
    """
    Create parent directory for an output file if needed.
    """
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def make_variant_config(
    base_config: MFConfig,
    alpha: float,
) -> MFConfig:
    """
    Create a copy of MFConfig with a different alpha.
    """
    return MFConfig(
        implementation=base_config.implementation,
        latent_dim=base_config.latent_dim,
        learning_rate=base_config.learning_rate,
        regularization=base_config.regularization,
        epochs=base_config.epochs,
        seed=base_config.seed,
        alpha=alpha,
        patience=base_config.patience,
        min_train_reviews=base_config.min_train_reviews,
        tau=base_config.tau,
    )


def build_dev_predictions_for_alpha(
    dev_df: pd.DataFrame,
    mf_model: BiasedMatrixFactorization,
    U: pd.DataFrame,
    V: pd.DataFrame,
    alpha: float,
) -> Tuple[pd.DataFrame, float]:
    """
    Build dev prediction dataframe for any alpha value.

    alpha = 1.0 -> Ratings-Only SVD
    alpha = 0.0 -> Aspect Content-Only
    alpha = 0.7 -> Hybrid
    """

    dev = dev_df.copy()
    dev["user_id"] = dev["user_id"].astype(str)
    dev["business_id"] = dev["business_id"].astype(str)
    dev["stars"] = dev["stars"].astype(float)

    score_cf_raw_list = []
    score_cf_norm_list = []
    score_aspect_raw_list = []
    score_aspect_norm_list = []
    hybrid_norm_list = []
    predicted_stars_list = []

    for row in dev.itertuples(index=False):
        user_id = getattr(row, "user_id")
        business_id = getattr(row, "business_id")

        (
            cf_raw,
            cf_norm,
            asp_raw,
            asp_norm,
            hybrid_norm,
        ) = hybrid_predict_norm(
            user_id=user_id,
            business_id=business_id,
            mf_model=mf_model,
            U=U,
            V=V,
            alpha=alpha,
        )

        predicted_stars = norm_to_stars(hybrid_norm)

        score_cf_raw_list.append(cf_raw)
        score_cf_norm_list.append(cf_norm)
        score_aspect_raw_list.append(asp_raw)
        score_aspect_norm_list.append(asp_norm)
        hybrid_norm_list.append(hybrid_norm)
        predicted_stars_list.append(predicted_stars)

    predictions_df = dev.copy()
    predictions_df["score_cf_raw"] = score_cf_raw_list
    predictions_df["score_cf_norm"] = score_cf_norm_list
    predictions_df["score_aspect_raw"] = score_aspect_raw_list
    predictions_df["score_aspect_norm"] = score_aspect_norm_list
    predictions_df["hybrid_score_norm"] = hybrid_norm_list
    predictions_df["predicted_stars"] = predicted_stars_list

    dev_rmse = rmse(
        predictions_df["stars"].to_numpy(dtype=np.float64),
        predictions_df["predicted_stars"].to_numpy(dtype=np.float64),
    )

    return predictions_df, dev_rmse


def run_alpha_variant_outputs(
    variant_name: str,
    alpha: float,
    train_df: pd.DataFrame,
    dev_df: pd.DataFrame,
    mf_model: BiasedMatrixFactorization,
    U: pd.DataFrame,
    V: pd.DataFrame,
    dev_users_path: str,
    test_users_path: str,
    outdir: str,
    base_config: MFConfig,
    business_name_map: Optional[Dict[str, Optional[str]]],
) -> Dict[str, Any]:
    """
    Generate predictions and Top-K recommendations for a model variant.

    This function reuses the already-trained MF model and changes only alpha.

    Example:
        variant_name = "ratings_only_svd", alpha = 1.0
        variant_name = "aspect_content_only", alpha = 0.0
    """

    print("=" * 80)
    print(f"Running variant: {variant_name}")
    print(f"alpha = {alpha}")
    print("=" * 80)

    variant_config = make_variant_config(
        base_config=base_config,
        alpha=alpha,
    )

    predictions_path = os.path.join(
        outdir,
        f"dev_predictions_{variant_name}.csv",
    )

    dev_recs_path = os.path.join(
        outdir,
        f"dev_recommendations_{variant_name}.csv",
    )

    test_recs_path = os.path.join(
        outdir,
        f"test_recommendations_{variant_name}.csv",
    )

    trace_path = os.path.join(
        outdir,
        f"trace_{variant_name}.jsonl",
    )

    ensure_parent_dir(predictions_path)
    ensure_parent_dir(dev_recs_path)
    ensure_parent_dir(test_recs_path)
    ensure_parent_dir(trace_path)

    predictions_df, variant_rmse = build_dev_predictions_for_alpha(
        dev_df=dev_df,
        mf_model=mf_model,
        U=U,
        V=V,
        alpha=alpha,
    )

    predictions_df.to_csv(
        predictions_path,
        index=False,
        float_format="%.6f",
    )

    print(f"{variant_name} dev RMSE: {variant_rmse:.4f}")
    print(f"Saved predictions to: {predictions_path}")

    run_top_k_recommendation(
        train_df=train_df,
        mf_model=mf_model,
        U=U,
        V=V,
        business_name_map=business_name_map,
        dev_users_path=dev_users_path,
        test_users_path=test_users_path,
        output_dev_path=dev_recs_path,
        output_test_path=test_recs_path,
        output_trace_path=trace_path,
        config=variant_config,
    )

    return {
        "variant_name": variant_name,
        "alpha": alpha,
        "dev_rmse": float(variant_rmse),
        "dev_predictions_path": predictions_path,
        "dev_recommendations_path": dev_recs_path,
        "test_recommendations_path": test_recs_path,
        "trace_path": trace_path,
    }


# ============================================================
# Popularity-Based baseline
# ============================================================

def build_popularity_statistics(
    train_df: pd.DataFrame,
) -> Tuple[List[str], Dict[str, int], Dict[str, float], float]:
    """
    Build popularity statistics from training data.

    catalogue:
        all businesses with at least one train review

    business_counts:
        number of training-review rows per business

    business_mean_stars:
        mean train stars per business

    global_mean:
        mean stars over all train reviews
    """

    train = train_df.copy()
    train["user_id"] = train["user_id"].astype(str)
    train["business_id"] = train["business_id"].astype(str)
    train["stars"] = train["stars"].astype(float)

    catalogue = sorted(train["business_id"].unique().tolist())

    business_counts = (
        train["business_id"]
        .value_counts()
        .astype(int)
        .to_dict()
    )

    business_mean_stars = (
        train.groupby("business_id")["stars"]
        .mean()
        .astype(float)
        .to_dict()
    )

    global_mean = float(train["stars"].mean())

    return catalogue, business_counts, business_mean_stars, global_mean


def build_popularity_dev_predictions(
    train_df: pd.DataFrame,
    dev_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, float]:
    """
    Build dev predictions for Popularity-Based baseline.

    Rating prediction:
        predicted_stars = business mean stars in train

    If business is unknown:
        predicted_stars = global train mean
    """

    (
        catalogue,
        business_counts,
        business_mean_stars,
        global_mean,
    ) = build_popularity_statistics(train_df)

    catalogue_set = set(catalogue)

    dev = dev_df.copy()
    dev["user_id"] = dev["user_id"].astype(str)
    dev["business_id"] = dev["business_id"].astype(str)
    dev["stars"] = dev["stars"].astype(float)

    # Catalogue constraint:
    # Evaluate only businesses present in training catalogue.
    dev_eval = dev[dev["business_id"].isin(catalogue_set)].copy()

    dev_eval["predicted_stars"] = (
        dev_eval["business_id"]
        .map(business_mean_stars)
        .fillna(global_mean)
        .astype(float)
    )

    # Optional normalized popularity score for inspection.
    max_count = max(business_counts.values()) if business_counts else 1

    dev_eval["popularity_count"] = (
        dev_eval["business_id"]
        .map(business_counts)
        .fillna(0)
        .astype(int)
    )

    dev_eval["popularity_score_norm"] = (
        dev_eval["popularity_count"].astype(float) / float(max_count)
    )

    popularity_rmse = rmse(
        dev_eval["stars"].to_numpy(dtype=np.float64),
        dev_eval["predicted_stars"].to_numpy(dtype=np.float64),
    )

    return dev_eval, popularity_rmse


def calculate_popularity_recommendations(
    target_user_ids: List[str],
    train_df: pd.DataFrame,
    top_k: int = 10,
) -> pd.DataFrame:
    """
    Generate Top-K recommendations using only item popularity.

    Candidates:
        catalogue businesses not already reviewed by user in train

    Ranking:
        popularity descending, business_id ascending for ties
    """

    (
        catalogue,
        business_counts,
        business_mean_stars,
        global_mean,
    ) = build_popularity_statistics(train_df)

    if len(catalogue) == 0:
        raise ValueError("The training catalogue is empty.")

    train = train_df.copy()
    train["user_id"] = train["user_id"].astype(str)
    train["business_id"] = train["business_id"].astype(str)

    user_seen_items = (
        train.groupby("user_id")["business_id"]
        .apply(set)
        .to_dict()
    )

    max_count = max(business_counts.values()) if business_counts else 1

    # Pre-sort all catalogue businesses once.
    popularity_ordered_items = sorted(
        catalogue,
        key=lambda b: (-business_counts.get(b, 0), b),
    )

    rows: List[Dict[str, Any]] = []

    for user_number, user_id in enumerate(target_user_ids, start=1):
        user_id = str(user_id)

        if user_number % 1000 == 0:
            print(
                f"Popularity baseline processed "
                f"{user_number:,}/{len(target_user_ids):,} users..."
            )

        seen = user_seen_items.get(user_id, set())

        candidates = [
            business_id
            for business_id in popularity_ordered_items
            if business_id not in seen
        ]

        top_items = candidates[:top_k]

        for rank, business_id in enumerate(top_items, start=1):
            score = float(business_counts.get(business_id, 0) / max_count)

            rows.append(
                {
                    "user_id": user_id,
                    "business_id": business_id,
                    "rank": int(rank),
                    "score": score,
                    # This is the actual popularity model, not abstention fallback.
                    "is_fallback": False,
                }
            )

    recommendations_df = pd.DataFrame(
        rows,
        columns=[
            "user_id",
            "business_id",
            "rank",
            "score",
            "is_fallback",
        ],
    )

    return recommendations_df


def run_popularity_baseline_outputs(
    train_df: pd.DataFrame,
    dev_df: pd.DataFrame,
    dev_users_path: str,
    test_users_path: str,
    outdir: str,
) -> Dict[str, Any]:
    """
    Generate all Popularity-Based baseline files.
    """

    print("=" * 80)
    print("Running baseline: popularity")
    print("=" * 80)

    predictions_path = os.path.join(
        outdir,
        "dev_predictions_popularity.csv",
    )

    dev_recs_path = os.path.join(
        outdir,
        "dev_recommendations_popularity.csv",
    )

    test_recs_path = os.path.join(
        outdir,
        "test_recommendations_popularity.csv",
    )

    ensure_parent_dir(predictions_path)
    ensure_parent_dir(dev_recs_path)
    ensure_parent_dir(test_recs_path)

    predictions_df, popularity_rmse = build_popularity_dev_predictions(
        train_df=train_df,
        dev_df=dev_df,
    )

    predictions_df.to_csv(
        predictions_path,
        index=False,
        float_format="%.6f",
    )

    print(f"Popularity dev RMSE: {popularity_rmse:.4f}")
    print(f"Saved popularity dev predictions to: {predictions_path}")

    dev_user_ids = load_user_ids(dev_users_path)
    test_user_ids = load_user_ids(test_users_path)

    dev_recommendations = calculate_popularity_recommendations(
        target_user_ids=dev_user_ids,
        train_df=train_df,
        top_k=10,
    )

    test_recommendations = calculate_popularity_recommendations(
        target_user_ids=test_user_ids,
        train_df=train_df,
        top_k=10,
    )

    save_recommendations(
        recommendations_df=dev_recommendations,
        output_path=dev_recs_path,
    )

    save_recommendations(
        recommendations_df=test_recommendations,
        output_path=test_recs_path,
    )

    print(f"Saved popularity dev recommendations to: {dev_recs_path}")
    print(f"Saved popularity test recommendations to: {test_recs_path}")

    return {
        "variant_name": "popularity",
        "alpha": None,
        "dev_rmse": float(popularity_rmse),
        "dev_predictions_path": predictions_path,
        "dev_recommendations_path": dev_recs_path,
        "test_recommendations_path": test_recs_path,
    }


def save_baseline_manifest(
    manifest: Dict[str, Any],
    output_path: str,
) -> None:
    """
    Save a small file listing the generated model/baseline outputs.

    This is not the final dev_metrics.json.
    evaluator.py can read these paths and compute HR@10/NDCG@10.
    """

    ensure_parent_dir(output_path)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            manifest,
            f,
            indent=4,
            ensure_ascii=False,
        )

    print(f"Saved baseline manifest to: {output_path}")


# for upgrading summery
def safe_load_json(path: str) -> Dict[str, Any]:
    if path is None:
        return {}

    if not os.path.exists(path):
        return {}

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_package_versions() -> Dict[str, str]:
    versions = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }

    try:
        import sklearn
        versions["scikit_learn"] = sklearn.__version__
    except Exception:
        versions["scikit_learn"] = "not_installed"

    try:
        import nltk
        versions["nltk"] = nltk.__version__
    except Exception:
        versions["nltk"] = "not_installed"

    try:
        import spacy
        versions["spacy"] = spacy.__version__
    except Exception:
        versions["spacy"] = "not_installed"

    return versions


def update_final_summary_json(
    summary_path: str,
    preprocessing_summary_path: str,
    aspect_summary_path: str,
    train_df: pd.DataFrame,
    dev_df: pd.DataFrame,
    config: MFConfig,
    wall_clock_runtime_seconds: float,
) -> None:
    """
    Merge preprocessing stats, aspect stats, runtime, versions,
    and final hyperparameters into results/summary.json.
    """

    summary = safe_load_json(summary_path)
    preprocessing_summary = safe_load_json(preprocessing_summary_path)
    aspect_summary = safe_load_json(aspect_summary_path)

    raw_record_discard_counts = preprocessing_summary.get(
        "raw_record_discard_counts",
        {
            "business_total_records": None,
            "business_retained_records": None,
            "business_discarded_records": None,
            "review_total_records": None,
            "review_retained_records": None,
            "review_discarded_records": None,
            "note": "preprocessing_summary.json was not found or did not contain raw_record_discard_counts.",
        },
    )

    preprocessing_row_counts = preprocessing_summary.get("row_counts", {})

    total_tokens_processed = preprocessing_summary.get(
        "total_tokens_processed",
        None,
    )

    # Prefer real aspect frequencies from aspect_summary.json.
    # Expected format:
    # {
    #   "aspect_frequencies": {
    #       "food": 123,
    #       "service": 456,
    #       ...
    #   }
    # }
    aspect_frequencies = aspect_summary.get(
        "aspect_frequencies",
        {
            aspect: None
            for aspect in CANONICAL_ASPECTS
        },
    )

    summary["raw_record_discard_counts"] = raw_record_discard_counts

    summary["row_counts"] = {
        "train_reviews": int(len(train_df)),
        "dev_reviews": int(len(dev_df)),
        "unique_train_users": int(train_df["user_id"].astype(str).nunique()),
        "unique_train_businesses": int(train_df["business_id"].astype(str).nunique()),
        "unique_dev_users": int(dev_df["user_id"].astype(str).nunique()),
        "unique_dev_businesses": int(dev_df["business_id"].astype(str).nunique()),

        # Keep Phase 0 counts too, if present.
        "phase0": preprocessing_row_counts,
    }

    summary["total_tokens_processed"] = total_tokens_processed

    summary["aspect_frequencies"] = {
        aspect: aspect_frequencies.get(aspect)
        for aspect in CANONICAL_ASPECTS
    }

    summary["wall_clock_runtime_seconds"] = float(wall_clock_runtime_seconds)

    summary["package_model_versions"] = get_package_versions()

    summary["hyperparameters"] = {
        "implementation": config.implementation,
        "latent_dim": int(config.latent_dim),
        "learning_rate": float(config.learning_rate),
        "regularization": float(config.regularization),
        "epochs": int(config.epochs),
        "seed": int(config.seed),
        "alpha": float(config.alpha),
        "patience": int(config.patience),
        "min_train_reviews": int(config.min_train_reviews),
        "tau": float(config.tau),
        "optimizer": "Stochastic Gradient Descent",
        "top_k": 10,
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)

    print(f"Final assignment summary updated at: {summary_path}")

if __name__ == "__main__":
    start_time = time.time()
    args = parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    args.output_predictions_path = os.path.join(
        args.outdir,
        "dev_predictions_hybrid.csv",
    )

    args.summary_path = os.path.join(
        args.outdir,
        "summary.json",
    )

    args.output_dev_path = os.path.join(
        args.outdir,
        "dev_recommendations.csv",
    )

    args.output_test_path = os.path.join(
        args.outdir,
        "test_recommendations.csv",
    )

    args.output_trace_path = os.path.join(
        args.outdir,
        "trace.jsonl",
    )

    if not 0.0 <= args.alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1].")

    if args.min_train_reviews < 0:
        raise ValueError("min_train_reviews must be non-negative.")

    if not 0.0 <= args.tau <= 1.0:
        raise ValueError(
            "tau must be in [0, 1] because it is compared to normalized hybrid score."
        )

    config = MFConfig(
        latent_dim=args.latent_dim,
        learning_rate=args.learning_rate,
        regularization=args.regularization,
        epochs=args.epochs,
        seed=args.seed,
        alpha=args.alpha,
        patience=args.patience,
        min_train_reviews=args.min_train_reviews,
        tau=args.tau,
    )

    # Part 3.4: Train ONCE, evaluate dev RMSE, save dev predictions and summary.
    (
        mf_model,
        train_df,
        dev_df,
        U,
        V,
        dev_rmse,
    ) = run_hybrid_scoring(
        train_path=args.train_path,
        dev_path=args.dev_path,
        u_path=args.u_path,
        v_path=args.v_path,
        output_predictions_path=args.output_predictions_path,
        summary_path=args.summary_path,
        config=config,
    )

    business_name_map = load_business_name_map(args.business_path)

    # Part 3.5: Reuse the same trained model. No second training.
    run_top_k_recommendation(
        train_df=train_df,
        mf_model=mf_model,
        U=U,
        V=V,
        business_name_map=business_name_map,
        dev_users_path=args.dev_users_path,
        test_users_path=args.test_users_path,
        output_dev_path=args.output_dev_path,
        output_test_path=args.output_test_path,
        output_trace_path=args.output_trace_path,
        config=config,
    )

    # ============================================================
    # Baseline/model-variant outputs
    # These are generated here, while evaluator.py will only compute metrics.
    # ============================================================

    baseline_manifest = {
        "VADER-Hybrid": {
            "variant_name": "hybrid",
            "alpha": float(config.alpha),
            "dev_rmse": float(dev_rmse),
            "dev_predictions_path": args.output_predictions_path,
            "dev_recommendations_path": args.output_dev_path,
            "test_recommendations_path": args.output_test_path,
            "trace_path": args.output_trace_path,
        }
    }

    # Ratings-Only SVD baseline:
    # alpha = 1.0 means final score uses only CF.
    ratings_only_info = run_alpha_variant_outputs(
        variant_name="ratings_only_svd",
        alpha=1.0,
        train_df=train_df,
        dev_df=dev_df,
        mf_model=mf_model,
        U=U,
        V=V,
        dev_users_path=args.dev_users_path,
        test_users_path=args.test_users_path,
        outdir=args.outdir,
        base_config=config,
        business_name_map=business_name_map
    )

    baseline_manifest["Ratings-Only SVD"] = ratings_only_info

    # Aspect Content-Only baseline:
    # alpha = 0.0 means final score uses only aspect content.
    aspect_only_info = run_alpha_variant_outputs(
        variant_name="aspect_content_only",
        alpha=0.0,
        train_df=train_df,
        dev_df=dev_df,
        mf_model=mf_model,
        U=U,
        V=V,
        dev_users_path=args.dev_users_path,
        test_users_path=args.test_users_path,
        outdir=args.outdir,
        base_config=config,
        business_name_map=business_name_map
    )

    baseline_manifest["Aspect Content-Only"] = aspect_only_info

    # Popularity-Based baseline:
    # Ranking uses review counts. RMSE uses business mean train stars.
    popularity_info = run_popularity_baseline_outputs(
        train_df=train_df,
        dev_df=dev_df,
        dev_users_path=args.dev_users_path,
        test_users_path=args.test_users_path,
        outdir=args.outdir,
    )

    baseline_manifest["Popularity-Based"] = popularity_info

    save_baseline_manifest(
        manifest=baseline_manifest,
        output_path=os.path.join(args.outdir, "baseline_manifest.json"),
    )

    wall_clock_runtime_seconds = time.time() - start_time

    update_final_summary_json(
        summary_path=args.summary_path,
        preprocessing_summary_path=args.preprocessing_summary_path,
        aspect_summary_path=args.aspect_summary_path,
        train_df=train_df,
        dev_df=dev_df,
        config=config,
        wall_clock_runtime_seconds=wall_clock_runtime_seconds,
    )

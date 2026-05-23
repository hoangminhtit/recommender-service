"""
Step 8 — Evaluation Metrics.

Computes retrieval and ranking quality metrics for the two pipeline stages:

Retrieval (BPR):
    - Recall@50
    - HitRate@50

Reranking (BGE):
    - NDCG@5
    - MAP@5
    - MRR

Evaluation uses a temporal split:
    - Train: Jan – Oct
    - Validation: Nov  (hyperparameter tuning)
    - Test: Dec        (final evaluation)

Ground truth: books purchased OR reviewed with score >= 4 in the eval period.
Cart interactions are excluded from ground truth.
"""

from __future__ import annotations

import math
import pandas as pd
from tqdm import tqdm

from src.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Ground truth builder
# ---------------------------------------------------------------------------

def build_ground_truth(
    sales_df: pd.DataFrame,
    reviews_df: pd.DataFrame,
    eval_year: int,
    eval_month: int,
    review_threshold: float = 4.0,
) -> dict[int, set[int]]:
    """
    Build ground truth relevant book sets for the evaluation period.

    Args:
        sales_df:         fact_sales with [buyer_id, book_id, order_date, item_status]
        reviews_df:       fact_reviews with [buyer_id, book_id, score, snapshot_date]
        eval_year:        Year of evaluation period.
        eval_month:       Month of evaluation period (1-indexed).
        review_threshold: Minimum review score to count as relevant.

    Returns:
        Dict user_id → set of relevant book_ids.
    """
    logger.info(
        "Building ground truth for %d-%02d (purchases + reviews >= %.1f) …",
        eval_year,
        eval_month,
        review_threshold,
    )

    # Filter to evaluation period
    purchases = sales_df[
        (pd.to_datetime(sales_df["order_date"]).dt.year == eval_year)
        & (pd.to_datetime(sales_df["order_date"]).dt.month == eval_month)
        & (sales_df["item_status"] == "completed")
    ]

    reviews = reviews_df[
        (pd.to_datetime(reviews_df["snapshot_date"]).dt.year == eval_year)
        & (pd.to_datetime(reviews_df["snapshot_date"]).dt.month == eval_month)
        & (reviews_df["score"] >= review_threshold)
    ]

    ground_truth: dict[int, set[int]] = {}

    for _, row in purchases.iterrows():
        uid, bid = int(row["buyer_id"]), int(row["book_id"])
        ground_truth.setdefault(uid, set()).add(bid)

    for _, row in reviews.iterrows():
        uid, bid = int(row["buyer_id"]), int(row["book_id"])
        ground_truth.setdefault(uid, set()).add(bid)

    logger.info(
        "Ground truth: %d users | avg relevant books/user: %.2f",
        len(ground_truth),
        sum(len(v) for v in ground_truth.values()) / max(len(ground_truth), 1),
    )
    return ground_truth


# ---------------------------------------------------------------------------
# Retrieval metrics (BPR)
# ---------------------------------------------------------------------------

def recall_at_k(
    candidates: dict[int, list[int]],
    ground_truth: dict[int, set[int]],
    k: int = 50,
) -> float:
    """
    Recall@K: fraction of ground-truth books found in top-K candidates.

    candidates: user_id → list of candidate book_ids (ordered)
    """
    recalls = []
    for uid in tqdm(ground_truth, desc=f"Computing Recall@{k}", unit="user", leave=False):
        relevant = ground_truth[uid]
        retrieved = set(candidates.get(uid, [])[:k])
        if relevant:
            recalls.append(len(relevant & retrieved) / len(relevant))

    score = float(sum(recalls) / len(recalls)) if recalls else 0.0
    logger.info("Recall@%d = %.4f (over %d users)", k, score, len(recalls))
    return score


def hit_rate_at_k(
    candidates: dict[int, list[int]],
    ground_truth: dict[int, set[int]],
    k: int = 50,
) -> float:
    """
    HitRate@K: fraction of users with at least one relevant book in top-K.
    """
    hits = 0
    total = 0
    for uid in tqdm(ground_truth, desc=f"Computing HitRate@{k}", unit="user", leave=False):
        relevant = ground_truth[uid]
        retrieved = set(candidates.get(uid, [])[:k])
        if relevant:
            total += 1
            if relevant & retrieved:
                hits += 1

    score = hits / total if total else 0.0
    logger.info("HitRate@%d = %.4f (%d/%d users hit)", k, score, hits, total)
    return score


# ---------------------------------------------------------------------------
# Reranking metrics (BGE)
# ---------------------------------------------------------------------------

def ndcg_at_k(
    recommendations: dict[int, list[dict]],
    ground_truth: dict[int, set[int]],
    k: int = 5,
) -> float:
    """
    NDCG@K: Normalized Discounted Cumulative Gain at rank K.

    recommendations: user_id → list of {"book_id": int, "rank": int, ...}
    """
    scores = []
    for uid in tqdm(ground_truth, desc=f"Computing NDCG@{k}", unit="user", leave=False):
        relevant = ground_truth.get(uid, set())
        recs = [r["book_id"] for r in recommendations.get(uid, [])][:k]

        dcg = sum(
            1.0 / math.log2(rank + 2)
            for rank, bid in enumerate(recs)
            if bid in relevant
        )
        ideal_hits = min(len(relevant), k)
        idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))

        scores.append(dcg / idcg if idcg > 0 else 0.0)

    score = sum(scores) / len(scores) if scores else 0.0
    logger.info("NDCG@%d = %.4f (over %d users)", k, score, len(scores))
    return score


def map_at_k(
    recommendations: dict[int, list[dict]],
    ground_truth: dict[int, set[int]],
    k: int = 5,
) -> float:
    """
    MAP@K: Mean Average Precision at rank K.
    """
    aps = []
    for uid in tqdm(ground_truth, desc=f"Computing MAP@{k}", unit="user", leave=False):
        relevant = ground_truth.get(uid, set())
        recs = [r["book_id"] for r in recommendations.get(uid, [])][:k]

        hits = 0
        precision_sum = 0.0
        for rank, bid in enumerate(recs):
            if bid in relevant:
                hits += 1
                precision_sum += hits / (rank + 1)

        aps.append(precision_sum / min(len(relevant), k) if relevant else 0.0)

    score = sum(aps) / len(aps) if aps else 0.0
    logger.info("MAP@%d = %.4f (over %d users)", k, score, len(aps))
    return score


def mean_reciprocal_rank(
    recommendations: dict[int, list[dict]],
    ground_truth: dict[int, set[int]],
) -> float:
    """
    MRR: Mean Reciprocal Rank (position of first relevant item).
    """
    rr_scores = []
    for uid in tqdm(ground_truth, desc="Computing MRR", unit="user", leave=False):
        relevant = ground_truth.get(uid, set())
        recs = [r["book_id"] for r in recommendations.get(uid, [])]

        rr = 0.0
        for rank, bid in enumerate(recs):
            if bid in relevant:
                rr = 1.0 / (rank + 1)
                break
        rr_scores.append(rr)

    score = sum(rr_scores) / len(rr_scores) if rr_scores else 0.0
    logger.info("MRR = %.4f (over %d users)", score, len(rr_scores))
    return score


# ---------------------------------------------------------------------------
# Full evaluation report
# ---------------------------------------------------------------------------

def evaluate_pipeline(
    candidates: dict[int, list[int]],
    recommendations: dict[int, list[dict]],
    ground_truth: dict[int, set[int]],
) -> dict[str, float]:
    """
    Run all metrics and return a summary report dict.

    Args:
        candidates:      BPR Top-50 output: user_id → [book_id, ...]
        recommendations: BGE Top-5 output:  user_id → [{book_id, rank, ...}]
        ground_truth:    Eval period relevant books: user_id → {book_id}

    Returns:
        Dict of metric_name → score.
    """
    logger.info("=" * 60)
    logger.info("Running full pipeline evaluation …")
    logger.info("=" * 60)

    cfg_ev = settings.evaluation

    report = {
        f"recall@{cfg_ev.recall_k}":  recall_at_k(candidates, ground_truth, k=cfg_ev.recall_k),
        f"hitrate@{cfg_ev.recall_k}": hit_rate_at_k(candidates, ground_truth, k=cfg_ev.recall_k),
        f"ndcg@{cfg_ev.ndcg_k}":      ndcg_at_k(recommendations, ground_truth, k=cfg_ev.ndcg_k),
        f"map@{cfg_ev.map_k}":        map_at_k(recommendations, ground_truth, k=cfg_ev.map_k),
        "mrr":                        mean_reciprocal_rank(recommendations, ground_truth),
    }

    logger.info("-" * 40)
    logger.info("Evaluation Summary:")
    for metric, value in report.items():
        logger.info("  %-15s %.4f", metric, value)
    logger.info("-" * 40)

    # Target comparison
    targets = {
        "recall@50":  0.75,
        "hitrate@50": 0.85,
        "ndcg@5":     0.35,
        "map@5":      0.25,
        "mrr":        0.40,
    }
    for metric, target in targets.items():
        status = "✓ PASS" if report[metric] >= target else "✗ FAIL"
        logger.info(
            "  %s %-15s %.4f (target >= %.2f)",
            status,
            metric,
            report[metric],
            target,
        )

    return report

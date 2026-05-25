"""
Step 7 — Cold Start Strategy.

Handles two cold-start scenarios:
    1. New users (< 5 interactions) → popularity-based fallback or hybrid
    2. New books (< 10 interactions) → excluded from BPR, served via content-based similarity

New User Logic:
    - 0 interactions  → pure popularity fallback
    - 1–4 interactions → hybrid (BPR low weight + popularity high weight)
    - 5+  interactions → full BPR (handled by main pipeline)

New Book Logic:
    - books with < 10 interactions are excluded from BPR training
    - served via BGE content-based similarity against user profile
"""

from __future__ import annotations

import random
import pandas as pd
from tqdm import tqdm

from src.config import settings
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.db.databricks_client import DatabricksClient
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# User interaction counts
# ---------------------------------------------------------------------------

def get_user_interaction_counts(interaction_df: pd.DataFrame) -> dict[int, int]:
    """
    Count total interactions per user from the interaction table.

    Returns:
        Dict mapping user_id → interaction_count
    """
    counts = (
        interaction_df.groupby("user_id")
        .size()
        .to_dict()
    )
    logger.debug(
        "Interaction count distribution: min=%d, max=%d, median=%.1f",
        min(counts.values()) if counts else 0,
        max(counts.values()) if counts else 0,
        pd.Series(list(counts.values())).median() if counts else 0,
    )
    return counts


def classify_users(
    all_user_ids: list[int],
    interaction_counts: dict[int, int],
) -> dict[str, list[int]]:
    """
    Classify users into three groups for routing.

    Returns:
        {
            "cold":   users with 0 interactions,
            "hybrid": users with 1–4 interactions,
            "warm":   users with 5+ interactions,
        }
    """
    cfg = settings.cold_start
    groups: dict[str, list[int]] = {"cold": [], "hybrid": [], "warm": []}

    for uid in all_user_ids:
        count = interaction_counts.get(uid, 0)
        if count == 0:
            groups["cold"].append(uid)
        elif count < cfg.min_interactions_for_full_bpr:
            groups["hybrid"].append(uid)
        else:
            groups["warm"].append(uid)

    logger.info(
        "User classification: cold=%d | hybrid=%d | warm=%d",
        len(groups["cold"]),
        len(groups["hybrid"]),
        len(groups["warm"]),
    )
    return groups


# ---------------------------------------------------------------------------
# Popularity fallback (new users)
# ---------------------------------------------------------------------------

_POPULARITY_SQL = """
SELECT
    book_id,
    title,
    author,
    category_name,
    rating_avg,
    purchase_count,
    (0.6 * PERCENT_RANK() OVER (ORDER BY purchase_count)
     + 0.4 * PERCENT_RANK() OVER (ORDER BY rating_avg)) AS popularity_score
FROM {catalog}.{schema}.dim_books
WHERE rating_avg >= 3.5
ORDER BY popularity_score DESC
LIMIT {pool_size}
"""


def get_popularity_recommendations(
    client,
    top_k: int = 5,
    user_id: int | None = None,
    random_state: int | None = None,
) -> list[dict]:
    """
    Fetch popularity-based recommendations for cold-start users.
    Applies random sampling within the Top-N pool for diversity.

    Returns:
        List of recommendation dicts: [{"book_id", "title", "popularity_score", "rank"}]
    """
    cfg_db = settings.databricks
    cfg_cs = settings.cold_start
    pool_size = cfg_cs.popularity_pool_size

    logger.info(
        "Cold start: fetching popularity pool (size=%d) → sample top_%d",
        pool_size,
        top_k,
    )

    sql = _POPULARITY_SQL.format(
        catalog=cfg_db.catalog,
        schema=cfg_db.schema,
        pool_size=pool_size,
    )

    with tqdm(total=1, desc="Popularity fallback query", unit="query") as pbar:
        df = client.query(sql)
        pbar.update(1)

    if random_state is None:
        if user_id is not None:
            random_state = int(user_id)
        else:
            random_state = settings.bpr.seed

    # Random sample for diversity
    sample_size = min(top_k, len(df))
    sampled = df.sample(n=sample_size, random_state=random_state)

    results = [
        {
            "book_id": int(row["book_id"]),
            "title": row["title"],
            "popularity_score": round(float(row["popularity_score"]), 6),
            "rank": rank + 1,
        }
        for rank, (_, row) in enumerate(sampled.iterrows())
    ]

    logger.info("Popularity recommendations ready: %d books", len(results))
    return results


# ---------------------------------------------------------------------------
# Hybrid fallback (users with < 5 interactions)
# ---------------------------------------------------------------------------

def build_hybrid_recommendations(
    bpr_results: list[dict],
    popularity_results: list[dict],
    bpr_weight: float = 0.3,
    popularity_weight: float = 0.7,
    top_k: int = 5,
) -> list[dict]:
    """
    Blend BPR candidate scores with popularity scores for hybrid users.

    BPR weight is low (0.3) because the user has very few interactions.
    Popularity weight is high (0.7) to compensate.

    Returns:
        Ranked list of top_k recommendation dicts.
    """
    logger.debug(
        "Hybrid blend: bpr_weight=%.1f | popularity_weight=%.1f",
        bpr_weight,
        popularity_weight,
    )

    scores: dict[int, float] = {}
    titles: dict[int, str] = {}

    for item in bpr_results:
        bid = item["book_id"]
        # BUG FIX: đọc "mf_score" (BPR matrix factorization score), không phải "semantic_score"
        scores[bid] = scores.get(bid, 0.0) + bpr_weight * item.get("mf_score", 0.0)

    for item in popularity_results:
        bid = item["book_id"]
        scores[bid] = scores.get(bid, 0.0) + popularity_weight * item.get("popularity_score", 0.0)
        titles[bid] = item.get("title", "")

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

    return [
        {"book_id": bid, "hybrid_score": round(score, 6), "rank": rank + 1}
        for rank, (bid, score) in enumerate(ranked)
    ]


# ---------------------------------------------------------------------------
# New book filter (exclude cold books from BPR training)
# ---------------------------------------------------------------------------

def filter_cold_books(
    interaction_df: pd.DataFrame,
    min_interactions: int | None = None,
) -> tuple[pd.DataFrame, set[int]]:
    """
    Exclude books with fewer than min_interactions from the BPR training set.

    Returns:
        (filtered_interaction_df, cold_book_ids)
    """
    if min_interactions is None:
        min_interactions = settings.cold_start.min_book_interactions

    book_counts = interaction_df.groupby("book_id").size()
    cold_books = set(book_counts[book_counts < min_interactions].index.tolist())
    warm_df = interaction_df[~interaction_df["book_id"].isin(cold_books)].copy()

    logger.info(
        "Book cold-start filter: %d books excluded (< %d interactions) | %d warm books retained",
        len(cold_books),
        min_interactions,
        warm_df["book_id"].nunique(),
    )

    return warm_df, cold_books

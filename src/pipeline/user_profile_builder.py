"""
Step 4 — Build Weighted User Profile.

Constructs a natural-language semantic profile for each user based on:
- Top-3 preferred categories (weighted by time-decayed interaction scores)
- Top-3 preferred authors
- Recent high-rated books (rating >= 4.0)
- Recently purchased book titles

The resulting profile text is the query-side input to the BGE reranker.
"""

from __future__ import annotations

import pandas as pd
from tqdm import tqdm

from src.config import settings
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.db.databricks_client import DatabricksClient
from src.utils.logger import get_logger

logger = get_logger(__name__)

_TOP_CATEGORIES_SQL = """
SELECT
    i.user_id,
    b.category_name,
    SUM(i.interaction_score) AS category_weight
FROM {catalog}.{schema}.interaction_scores i
JOIN {catalog}.{schema}.dim_books b ON i.book_id = b.book_id
GROUP BY i.user_id, b.category_name
ORDER BY i.user_id, category_weight DESC
"""

_TOP_AUTHORS_SQL = """
SELECT
    i.user_id,
    b.author,
    SUM(i.interaction_score) AS author_weight
FROM {catalog}.{schema}.interaction_scores i
JOIN {catalog}.{schema}.dim_books b ON i.book_id = b.book_id
GROUP BY i.user_id, b.author
ORDER BY i.user_id, author_weight DESC
"""

_RECENT_PURCHASES_SQL = """
SELECT
    fs.buyer_id AS user_id,
    b.title,
    fs.order_date
FROM {catalog}.{schema}.fact_sales fs
JOIN {catalog}.{schema}.dim_books b ON fs.book_id = b.book_id
WHERE fs.item_status = 'completed'
ORDER BY fs.buyer_id, fs.order_date DESC
"""

_HIGH_RATED_SQL = """
SELECT
    fr.buyer_id AS user_id,
    b.title,
    fr.score
FROM {catalog}.{schema}.fact_reviews fr
JOIN {catalog}.{schema}.dim_books b ON fr.book_id = b.book_id
WHERE fr.score >= {review_threshold}
ORDER BY fr.buyer_id, fr.score DESC
"""


def build_user_profiles(
    client,
    interaction_df: pd.DataFrame,
) -> dict[int, str]:
    """
    Build natural-language user profiles for all users in interaction_df.

    Args:
        client:         Active DatabricksClient.
        interaction_df: Time-aware interaction table from Step 1.

    Returns:
        Dict mapping user_id → profile string.
    """
    cfg_db = settings.databricks
    cfg_td = settings.time_decay
    user_ids = sorted(interaction_df["user_id"].unique().tolist())

    logger.info("Step 4 — Building user profiles for %d users …", len(user_ids))

    fmt = dict(catalog=cfg_db.catalog, schema=cfg_db.schema)

    with tqdm(total=4, desc="User profile queries", unit="query") as pbar:
        pbar.set_description("Top categories")
        categories_df = client.query(_TOP_CATEGORIES_SQL.format(**fmt))
        pbar.update(1)

        pbar.set_description("Top authors")
        authors_df = client.query(_TOP_AUTHORS_SQL.format(**fmt))
        pbar.update(1)

        pbar.set_description("Recent purchases")
        purchases_df = client.query(_RECENT_PURCHASES_SQL.format(**fmt))
        pbar.update(1)

        pbar.set_description("High-rated books")
        rated_df = client.query(
            _HIGH_RATED_SQL.format(
                review_threshold=cfg_td.positive_review_threshold, **fmt
            )
        )
        pbar.update(1)


    # Build per-user profile text
    profiles: dict[int, str] = {}

    for uid in tqdm(user_ids, desc="Assembling user profiles", unit="user"):
        profile_text = _assemble_profile(
            user_id=uid,
            categories_df=categories_df,
            authors_df=authors_df,
            purchases_df=purchases_df,
            rated_df=rated_df,
            top_n=3,
            recent_n=2,
        )
        profiles[uid] = profile_text

    logger.info("User profiles built for %d users.", len(profiles))
    return profiles


def _assemble_profile(
    user_id: int,
    categories_df: pd.DataFrame,
    authors_df: pd.DataFrame,
    purchases_df: pd.DataFrame,
    rated_df: pd.DataFrame,
    top_n: int = 3,
    recent_n: int = 2,
) -> str:
    """Assemble a profile text string for a single user."""

    # Top categories
    cats = (
        categories_df[categories_df["user_id"] == user_id]
        .head(top_n)["category_name"]
        .tolist()
    )

    # Top authors
    authors = (
        authors_df[authors_df["user_id"] == user_id]
        .head(top_n)["author"]
        .tolist()
    )

    # Recent purchases
    recent_buys = (
        purchases_df[purchases_df["user_id"] == user_id]
        .head(recent_n)["title"]
        .tolist()
    )

    # High-rated books
    high_rated = (
        rated_df[rated_df["user_id"] == user_id]
        .head(recent_n)["title"]
        .tolist()
    )

    lines = ["User strongly prefers:"]
    if cats:
        lines.append(f"- Categories: {', '.join(cats)}")
    if authors:
        lines.append(f"- Authors: {', '.join(authors)}")
    if high_rated:
        lines.append(f"- Highly rated books: {', '.join(high_rated)}")
    if recent_buys:
        lines.append(f"- Recently purchased: {', '.join(recent_buys)}")

    profile_text = "\n".join(lines)

    # Log a sample at DEBUG level
    logger.debug("Profile for user %d:\n%s", user_id, profile_text)

    return profile_text

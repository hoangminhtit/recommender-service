"""
Step 1 — Build Time-Aware Interaction Table.

Transforms user behavioral data (purchases, carts, reviews) into
implicit feedback scores using exponential time decay.

Formula:
    interaction_score(user, book) =
        SUM( base_score × exp(-λ × months_ago) )
        per (user_id, book_id)
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

# SQL template — parameterized by decay rate
_INTERACTION_SQL = """
WITH interaction_source AS (

    -- Purchases (base_score = 10)
    SELECT
        buyer_id,
        book_id,
        DATEDIFF(CURRENT_DATE, order_date) / 30.0  AS months_ago,
        {purchase_score}                            AS base_score
    FROM {catalog}.{schema}.fact_sales
    WHERE item_status = 'completed'

    UNION ALL

    -- Cart interactions (base_score = 3)
    SELECT
        buyer_id,
        book_id,
        DATEDIFF(CURRENT_DATE, added_at) / 30.0    AS months_ago,
        {cart_score}                                AS base_score
    FROM {catalog}.{schema}.fact_cart

    UNION ALL

    -- Positive reviews (score >= {review_threshold}, base_score = 5)
    SELECT
        buyer_id,
        book_id,
        DATEDIFF(CURRENT_DATE, snapshot_date) / 30.0 AS months_ago,
        CASE
            WHEN score >= {review_threshold} THEN {review_score}
            ELSE 0
        END AS base_score
    FROM {catalog}.{schema}.fact_reviews
    WHERE score >= {review_threshold}

),

decayed_interactions AS (
    SELECT
        buyer_id        AS user_id,
        book_id,
        base_score * EXP(-{lambda_rate} * months_ago) AS decayed_score
    FROM interaction_source
    WHERE base_score > 0
)

SELECT
    user_id,
    book_id,
    SUM(decayed_score) AS interaction_score
FROM decayed_interactions
GROUP BY user_id, book_id
"""


def build_interaction_table(client) -> pd.DataFrame:
    """
    Query Gold Layer and build the time-aware interaction table.

    Returns:
        DataFrame with columns: [user_id, book_id, interaction_score]
    """
    cfg_td = settings.time_decay
    cfg_db = settings.databricks

    logger.info("Step 1 — Building time-aware interaction table …")
    logger.info(
        "  Decay config: lambda=%.3f | purchase=%d | cart=%d | review=%d (threshold>=%.1f)",
        cfg_td.lambda_rate,
        cfg_td.purchase_score,
        cfg_td.cart_score,
        cfg_td.positive_review_score,
        cfg_td.positive_review_threshold,
    )

    sql = _INTERACTION_SQL.format(
        catalog=cfg_db.catalog,
        schema=cfg_db.schema,
        lambda_rate=cfg_td.lambda_rate,
        purchase_score=cfg_td.purchase_score,
        cart_score=cfg_td.cart_score,
        review_score=cfg_td.positive_review_score,
        review_threshold=cfg_td.positive_review_threshold,
    )

    # Single heavy query — wrap progress bar around the I/O wait
    with tqdm(total=1, desc="Fetching interaction table", unit="query") as pbar:
        df = client.query(sql)
        pbar.update(1)

    logger.info(
        "Interaction table built: %d rows | %d unique users | %d unique books",
        len(df),
        df["user_id"].nunique(),
        df["book_id"].nunique(),
    )

    df = _validate_interaction_table(df)
    return df


def _validate_interaction_table(df: pd.DataFrame) -> pd.DataFrame:
    """Basic sanity checks on the interaction table."""
    assert "user_id" in df.columns, "Missing column: user_id"
    assert "book_id" in df.columns, "Missing column: book_id"
    assert "interaction_score" in df.columns, "Missing column: interaction_score"

    null_count = df["interaction_score"].isna().sum()
    if null_count:
        logger.warning("interaction_score has %d null values — dropping them.", null_count)
        df.dropna(subset=["interaction_score"], inplace=True)

    neg_count = (df["interaction_score"] <= 0).sum()
    if neg_count:
        logger.warning("Dropping %d rows with interaction_score <= 0.", neg_count)
        df = df[df["interaction_score"] > 0]

    logger.debug("Interaction score stats:\n%s", df["interaction_score"].describe())
    return df

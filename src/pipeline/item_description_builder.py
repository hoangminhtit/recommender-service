"""
Step 5 — Build Item Description for BGE Reranker.

Constructs a natural-language description for each candidate book
retrieved from the BPR Top-50 output.

The description is the document-side input to the BGE cross-encoder.

Output:
    Dict mapping book_id → description string.
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

_ITEM_DESCRIPTION_SQL = """
SELECT
    book_id,
    title,
    author,
    category_name,
    ROUND(rating_avg, 1) AS rating_avg,
    purchase_count
FROM {catalog}.{schema}.dim_books
WHERE book_id IN ({placeholders})
"""


def build_item_descriptions(
    client,
    candidate_book_ids: list[int],
) -> dict[int, str]:
    """
    Build semantic descriptions for a set of candidate books.

    Args:
        client:              Active DatabricksClient.
        candidate_book_ids:  List of unique book IDs from BPR Top-50 output.

    Returns:
        Dict mapping book_id → description string.
    """
    if not candidate_book_ids:
        logger.warning("No candidate book IDs provided — returning empty dict.")
        return {}

    cfg_db = settings.databricks
    unique_ids = list(set(candidate_book_ids))

    logger.info(
        "Step 5 — Building item descriptions for %d unique candidate books …",
        len(unique_ids),
    )

    chunk_size = 1000
    chunks = [unique_ids[i : i + chunk_size] for i in range(0, len(unique_ids), chunk_size)]
    books_parts = []

    with tqdm(total=len(chunks), desc="Fetching book metadata", unit="query") as pbar:
        for chunk in chunks:
            placeholders = ", ".join(str(bid) for bid in chunk)
            sql = _ITEM_DESCRIPTION_SQL.format(
                catalog=cfg_db.catalog,
                schema=cfg_db.schema,
                placeholders=placeholders,
            )
            books_parts.append(client.query(sql))
            pbar.update(1)

    books_df = pd.concat(books_parts, ignore_index=True) if books_parts else pd.DataFrame()

    logger.info("Fetched metadata for %d books.", len(books_df))

    # Warn about any missing books
    fetched_ids = set(books_df["book_id"].tolist())
    missing = set(unique_ids) - fetched_ids
    if missing:
        logger.warning(
            "%d candidate books not found in dim_books: %s",
            len(missing),
            list(missing)[:10],  # log at most 10
        )

    descriptions: dict[int, str] = {}
    for _, row in tqdm(books_df.iterrows(), total=len(books_df), desc="Formatting descriptions", unit="book"):
        descriptions[row["book_id"]] = _format_description(row)

    logger.info("Item descriptions built for %d books.", len(descriptions))
    return descriptions


def _format_description(row: pd.Series) -> str:
    """Format a single book row into a description string for the BGE reranker."""
    return (
        f"Title: {row['title']}\n"
        f"Author: {row['author']}\n"
        f"Category: {row['category_name']}\n"
        f"Average Rating: {row['rating_avg']}\n"
        f"Popularity rank: {row['purchase_count']}"
    )

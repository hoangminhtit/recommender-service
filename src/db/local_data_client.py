from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class LocalDataClient:
    """Local CSV-backed client that mimics DatabricksClient.query()."""

    def __init__(self, data_dir: str | Path) -> None:
        self._data_dir = Path(data_dir)
        self._tables: dict[str, pd.DataFrame] = {}
        self._interaction_scores: pd.DataFrame | None = None

    def connect(self) -> None:
        logger.info("Using local CSV data from %s", self._data_dir)

    def close(self) -> None:
        return None

    def __enter__(self) -> "LocalDataClient":
        self.connect()
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def set_interaction_scores(self, df: pd.DataFrame) -> None:
        """Set precomputed interaction_scores for local SQL emulation."""
        self._interaction_scores = df.copy()

    def query(self, sql: str, params: dict | None = None) -> pd.DataFrame:
        """Execute a limited subset of SQL using local CSV data."""
        sql_lower = " ".join(sql.lower().split())

        if "interaction_source" in sql_lower and "interaction_score" in sql_lower:
            return self._build_interaction_scores()

        if "popularity_score" in sql_lower and "dim_books" in sql_lower:
            return self._popularity_pool(sql_lower)

        if "interaction_scores" in sql_lower and "category_weight" in sql_lower:
            return self._top_categories()

        if "interaction_scores" in sql_lower and "author_weight" in sql_lower:
            return self._top_authors()

        if "fact_sales" in sql_lower and "dim_books" in sql_lower and "order_date" in sql_lower:
            return self._recent_purchases()

        if "fact_sales" in sql_lower:
            return self._load_table("fact_sales")

        if "fact_cart" in sql_lower:
            return self._load_table("fact_cart")

        # high-rated books query: fact_reviews JOIN dim_books — must come before plain fact_reviews
        if "fact_reviews" in sql_lower and "dim_books" in sql_lower:
            return self._high_rated_books(sql_lower)

        if "fact_reviews" in sql_lower:
            return self._load_table("fact_reviews")

        if "dim_users" in sql_lower:
            return self._load_table("dim_users")

        if "dim_books" in sql_lower and "where book_id in" in sql_lower:
            return self._dim_books_in(sql)

        if "dim_books" in sql_lower:
            return self._load_table("dim_books")

        raise ValueError(f"Unsupported local SQL: {sql}")

    def _load_table(self, name: str) -> pd.DataFrame:
        if name in self._tables:
            return self._tables[name]
        path = self._data_dir / f"{name}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing local table: {path}")
        df = pd.read_csv(path)
        self._tables[name] = df
        return df

    def _build_interaction_scores(self) -> pd.DataFrame:
        cfg = settings.time_decay
        now = pd.Timestamp.utcnow().tz_localize(None).normalize()

        sales = self._load_table("fact_sales").copy()
        sales = sales[sales["item_status"] == "completed"]
        sales["months_ago"] = (
            (now - pd.to_datetime(sales["order_date"]))
            .dt.days
            .astype(float)
            / 30.0
        )
        sales["base_score"] = cfg.purchase_score
        sales = sales[["buyer_id", "book_id", "months_ago", "base_score"]]

        cart = self._load_table("fact_cart").copy()
        cart["months_ago"] = (
            (now - pd.to_datetime(cart["added_at"]))
            .dt.days
            .astype(float)
            / 30.0
        )
        cart["base_score"] = cfg.cart_score
        cart = cart[["buyer_id", "book_id", "months_ago", "base_score"]]

        reviews = self._load_table("fact_reviews").copy()
        reviews = reviews[reviews["score"] >= cfg.positive_review_threshold]
        reviews["months_ago"] = (
            (now - pd.to_datetime(reviews["snapshot_date"]))
            .dt.days
            .astype(float)
            / 30.0
        )
        reviews["base_score"] = cfg.positive_review_score
        reviews = reviews[["buyer_id", "book_id", "months_ago", "base_score"]]

        interactions = pd.concat([sales, cart, reviews], ignore_index=True)
        interactions = interactions[interactions["base_score"] > 0]

        interactions["decayed_score"] = interactions["base_score"] * np.exp(
            -cfg.lambda_rate * interactions["months_ago"]
        )

        grouped = (
            interactions
            .rename(columns={"buyer_id": "user_id"})
            .groupby(["user_id", "book_id"], as_index=False)["decayed_score"]
            .sum()
            .rename(columns={"decayed_score": "interaction_score"})
        )
        return grouped

    def _popularity_pool(self, sql_lower: str) -> pd.DataFrame:
        books = self._load_table("dim_books").copy()
        books = books[books["rating_avg"] >= 3.5]

        def percent_rank(series: pd.Series) -> pd.Series:
            n = len(series)
            if n <= 1:
                return pd.Series([0.0] * n, index=series.index)
            ranks = series.rank(method="min", ascending=True)
            return (ranks - 1) / (n - 1)

        books["purchase_pr"] = percent_rank(books["purchase_count"])
        books["rating_pr"] = percent_rank(books["rating_avg"])
        books["popularity_score"] = 0.6 * books["purchase_pr"] + 0.4 * books["rating_pr"]

        limit_match = re.search(r"limit\s+(\d+)", sql_lower)
        limit = int(limit_match.group(1)) if limit_match else len(books)

        cols = [
            "book_id",
            "title",
            "author",
            "category_name",
            "rating_avg",
            "purchase_count",
            "popularity_score",
        ]
        return books.sort_values("popularity_score", ascending=False)[cols].head(limit)

    def _top_categories(self) -> pd.DataFrame:
        if self._interaction_scores is None:
            raise RuntimeError("interaction_scores not set for local client")
        books = self._load_table("dim_books")
        merged = self._interaction_scores.merge(books, on="book_id", how="left")
        grouped = (
            merged
            .groupby(["user_id", "category_name"], as_index=False)["interaction_score"]
            .sum()
            .rename(columns={"interaction_score": "category_weight"})
        )
        return grouped.sort_values(["user_id", "category_weight"], ascending=[True, False])

    def _top_authors(self) -> pd.DataFrame:
        if self._interaction_scores is None:
            raise RuntimeError("interaction_scores not set for local client")
        books = self._load_table("dim_books")
        merged = self._interaction_scores.merge(books, on="book_id", how="left")
        grouped = (
            merged
            .groupby(["user_id", "author"], as_index=False)["interaction_score"]
            .sum()
            .rename(columns={"interaction_score": "author_weight"})
        )
        return grouped.sort_values(["user_id", "author_weight"], ascending=[True, False])

    def _recent_purchases(self) -> pd.DataFrame:
        sales = self._load_table("fact_sales").copy()
        sales = sales[sales["item_status"] == "completed"]
        books = self._load_table("dim_books")
        merged = sales.merge(books, left_on="book_id", right_on="book_id", how="left")
        merged = merged.rename(columns={"buyer_id": "user_id"})
        cols = ["user_id", "title", "order_date"]
        merged = merged[cols]
        return merged.sort_values(["user_id", "order_date"], ascending=[True, False])

    def _high_rated_books(self, sql_lower: str) -> pd.DataFrame:
        """Handle: SELECT buyer_id AS user_id, title, score FROM fact_reviews JOIN dim_books WHERE score >= threshold."""
        cfg = settings.time_decay
        reviews = self._load_table("fact_reviews").copy()
        reviews = reviews[reviews["score"] >= cfg.positive_review_threshold]
        books = self._load_table("dim_books")
        merged = reviews.merge(books, on="book_id", how="left")
        merged = merged.rename(columns={"buyer_id": "user_id"})
        cols = ["user_id", "title", "score"]
        merged = merged[cols]
        return merged.sort_values(["user_id", "score"], ascending=[True, False])

    def _dim_books_in(self, sql: str) -> pd.DataFrame:
        match = re.search(r"book_id\s+in\s*\(([^)]+)\)", sql, re.IGNORECASE)
        if not match:
            raise ValueError("Failed to parse book_id list from SQL")
        raw = match.group(1)
        ids = [int(x.strip()) for x in raw.split(",") if x.strip().isdigit()]
        books = self._load_table("dim_books")
        return books[books["book_id"].isin(ids)].copy()

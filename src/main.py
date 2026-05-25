"""
Recommendation Pipeline Orchestrator.

Entry point for the full batch recommendation pipeline:

    Step 1 — Build time-aware interaction table
    Step 2 — Train BPR model
    Step 3 — Generate Top-50 candidates
    Step 4 — Build weighted user profiles
    Step 5 — Build item descriptions
    Step 6 — BGE semantic reranking → Top-5
    Step 7 — Handle cold-start users (popularity / hybrid fallback)
    Step 8 — Cache final recommendations
    Step 9 — Evaluate (optional, triggered by --evaluate flag)

Usage:
    python -m src.main
    python -m src.main --evaluate
    python -m src.main --user-id 101          # single-user debug run
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.db.databricks_client import DatabricksClient

from tqdm import tqdm

from src.config import settings
from src.db import LocalDataClient
from src.evaluation import evaluate_pipeline, build_ground_truth
from src.fallback import (
    classify_users,
    filter_cold_books,
    get_popularity_recommendations,
    get_user_interaction_counts,
    build_hybrid_recommendations,
)
from src.pipeline import (
    build_interaction_table,
    train_bpr,
    save_bpr_model,
    load_bpr_model,
    generate_candidates,
    build_user_profiles,
    build_item_descriptions,
    rerank_candidates,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="BPR + BGE Recommendation Pipeline"
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Run evaluation after pipeline completes.",
    )
    parser.add_argument(
        "--user-id",
        type=int,
        default=None,
        metavar="USER_ID",
        help="Run pipeline for a single user (debug mode).",
    )
    parser.add_argument(
        "--skip-cache",
        action="store_true",
        help="Do not write results to Redis cache.",
    )
    parser.add_argument(
        "--local-data-dir",
        default=None,
        help="Use local CSV data instead of Databricks (path to dwh_mock).",
    )
    parser.add_argument(
        "--bpr-checkpoint-path",
        default=settings.bpr.checkpoint_path,
        help="Path to save/load BPR checkpoint.",
    )
    parser.add_argument(
        "--load-bpr",
        action="store_true",
        help="Load BPR checkpoint instead of training.",
    )
    parser.add_argument(
        "--no-save-bpr",
        action="store_true",
        help="Do not save BPR checkpoint after training.",
    )
    return parser.parse_args()


def _make_client(local_data_dir: str | None):
    # Use local CSV data for testing when provided; otherwise use Databricks.
    if local_data_dir:
        return LocalDataClient(local_data_dir)
    from src.db.databricks_client import DatabricksClient

    return DatabricksClient()


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------

def run_pipeline(
    evaluate: bool = False,
    target_user_id: int | None = None,
    skip_cache: bool = False,
    bpr_checkpoint_path: str | None = None,
    load_bpr: bool = False,
    save_bpr: bool = True,
    local_data_dir: str | None = None,
) -> dict[int, list[dict]]:
    """
    Execute the full recommendation pipeline.

    Args:
        evaluate:        Run evaluation metrics after pipeline.
        target_user_id:  If set, run only for this user (debug).
        skip_cache:      If True, skip writing to Redis.
        bpr_checkpoint_path: Path to save/load BPR checkpoint.
        load_bpr:         If True, load BPR checkpoint instead of training.
        save_bpr:         If True, save BPR checkpoint after training.
        local_data_dir:   If set, use local CSV data instead of Databricks.

    Returns:
        Dict user_id → Top-5 recommendation list.
    """
    pipeline_start = time.perf_counter()
    logger.info("=" * 60)
    logger.info("Recommendation Pipeline Starting — %s", datetime.now().isoformat())
    logger.info("=" * 60)

    with _make_client(local_data_dir) as client:

        # ----------------------------------------------------------------
        # Step 1 — Build interaction table
        # ----------------------------------------------------------------
        t0 = time.perf_counter()
        interaction_df = build_interaction_table(client)
        logger.info("Step 1 done in %.2fs", time.perf_counter() - t0)
        # Local client uses interaction_scores to emulate SQL tables.
        if hasattr(client, "set_interaction_scores"):
            client.set_interaction_scores(interaction_df)

        # Narrow to single user if debug mode
        if target_user_id is not None:
            logger.info("Debug mode: filtering to user_id=%d", target_user_id)
            interaction_df = interaction_df[
                interaction_df["user_id"] == target_user_id
            ].copy()
            if interaction_df.empty:
                logger.warning(
                    "No interactions found for user %d. Returning popularity fallback.",
                    target_user_id,
                )
                return {
                    target_user_id: get_popularity_recommendations(
                        client,
                        top_k=5,
                        user_id=target_user_id,
                    )
                }

        # ----------------------------------------------------------------
        # Step 7a — Cold-start: filter cold books from training
        # ----------------------------------------------------------------
        interaction_df, cold_book_ids = filter_cold_books(interaction_df)

        # Classify users (include users with zero interactions)
        all_user_ids = interaction_df["user_id"].unique().tolist()
        try:
            users_df = client.query(
                f"SELECT user_id FROM {settings.databricks.catalog}."
                f"{settings.databricks.schema}.dim_users"
            )
            all_user_ids = sorted(
                set(all_user_ids) | set(users_df["user_id"].astype(int).tolist())
            )
        except Exception as exc:
            logger.warning("Failed to load dim_users for cold-start: %s", exc)
        interaction_counts = get_user_interaction_counts(interaction_df)
        user_groups = classify_users(all_user_ids, interaction_counts)

        # ----------------------------------------------------------------
        # Step 2 — Train BPR
        # ----------------------------------------------------------------
        t0 = time.perf_counter()
        if load_bpr:
            bpr_model = load_bpr_model(bpr_checkpoint_path or settings.bpr.checkpoint_path)
        else:
            bpr_model = train_bpr(interaction_df)
            if save_bpr:
                save_bpr_model(bpr_model, bpr_checkpoint_path or settings.bpr.checkpoint_path)
        logger.info("Step 2 done in %.2fs", time.perf_counter() - t0)

        # ----------------------------------------------------------------
        # Step 3 — Generate Top-50 candidates (warm users only)
        # ----------------------------------------------------------------
        t0 = time.perf_counter()
        candidates_df = generate_candidates(bpr_model, interaction_df)

        # Extract per-user candidate lists (for evaluation)
        candidates_dict: dict[int, list[int]] = (
            candidates_df
            .groupby("user_id")["candidate_book_id"]
            .apply(list)
            .to_dict()
        )
        logger.info("Step 3 done in %.2fs", time.perf_counter() - t0)

        # ----------------------------------------------------------------
        # Step 4 — Build user profiles
        # ----------------------------------------------------------------
        t0 = time.perf_counter()
        user_profiles = build_user_profiles(client, interaction_df)
        logger.info("Step 4 done in %.2fs", time.perf_counter() - t0)

        # ----------------------------------------------------------------
        # Step 5 — Build item descriptions
        # ----------------------------------------------------------------
        t0 = time.perf_counter()
        candidate_book_ids = candidates_df["candidate_book_id"].unique().tolist()
        item_descriptions = build_item_descriptions(client, candidate_book_ids)
        logger.info("Step 5 done in %.2fs", time.perf_counter() - t0)

        # ----------------------------------------------------------------
        # Step 6 — BGE Reranking (warm users)
        # ----------------------------------------------------------------
        t0 = time.perf_counter()
        warm_candidates = candidates_df[
            candidates_df["user_id"].isin(user_groups["warm"])
        ]
        recommendations: dict[int, list[dict]] = rerank_candidates(
            candidates_df=warm_candidates,
            user_profiles=user_profiles,
            item_descriptions=item_descriptions,
        )
        logger.info("Step 6 done in %.2fs", time.perf_counter() - t0)

        # ----------------------------------------------------------------
        # Step 7b — Cold & Hybrid user fallback
        # ----------------------------------------------------------------
        t0 = time.perf_counter()
        logger.info(
            "Step 7 — Processing %d cold + %d hybrid users …",
            len(user_groups["cold"]),
            len(user_groups["hybrid"]),
        )

        for uid in tqdm(user_groups["cold"], desc="Cold-start users", unit="user"):
            recommendations[uid] = get_popularity_recommendations(
                client,
                top_k=5,
                user_id=uid,
            )

        for uid in tqdm(user_groups["hybrid"], desc="Hybrid users", unit="user"):
            popularity_recs = get_popularity_recommendations(
                client,
                top_k=settings.reranker.top_k,
                user_id=uid,
            )
            bpr_recs = [
                # BUG FIX: dùng đúng key "mf_score" (không phải "semantic_score")
                {"book_id": r["candidate_book_id"], "mf_score": r.get("mf_score", 0.0)}
                for _, r in candidates_df[candidates_df["user_id"] == uid].iterrows()
            ]
            recommendations[uid] = build_hybrid_recommendations(
                bpr_results=bpr_recs,
                popularity_results=popularity_recs,
                bpr_weight=0.3,
                popularity_weight=0.7,
                top_k=settings.reranker.top_k,
            )

        logger.info("Step 7 done in %.2fs", time.perf_counter() - t0)

        # ----------------------------------------------------------------
        # Step 8 — Cache results (disabled)
        # ----------------------------------------------------------------
        if skip_cache:
            logger.info("--skip-cache flag set: caching disabled.")
        else:
            logger.info("Caching disabled: Redis not configured for this run.")

        # ----------------------------------------------------------------
        # Step 9 — Evaluation (optional)
        # ----------------------------------------------------------------
        if evaluate:
            _run_evaluation(client, candidates_dict, recommendations)

    total_elapsed = time.perf_counter() - pipeline_start
    logger.info("=" * 60)
    logger.info(
        "Pipeline complete in %.2fs — %d users processed.",
        total_elapsed,
        len(recommendations),
    )
    logger.info("=" * 60)

    return recommendations


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------

def _run_evaluation(
    client,  # DatabricksClient | LocalDataClient (TYPE_CHECKING guard ở trên)
    candidates_dict: dict[int, list[int]],
    recommendations: dict[int, list[dict]],
) -> None:
    """Fetch ground truth and run all evaluation metrics."""
    cfg_db = settings.databricks
    cfg_ev = settings.evaluation

    logger.info("Fetching ground-truth tables for evaluation …")

    sales_df = client.query(
        f"SELECT buyer_id, book_id, order_date, item_status "
        f"FROM {cfg_db.catalog}.{cfg_db.schema}.fact_sales"
    )
    reviews_df = client.query(
        f"SELECT buyer_id, book_id, score, snapshot_date "
        f"FROM {cfg_db.catalog}.{cfg_db.schema}.fact_reviews"
    )

    eval_year = datetime.now().year
    ground_truth = build_ground_truth(
        sales_df=sales_df,
        reviews_df=reviews_df,
        eval_year=eval_year,
        eval_month=cfg_ev.test_month,
        review_threshold=settings.time_decay.positive_review_threshold,
    )

    evaluate_pipeline(
        candidates=candidates_dict,
        recommendations=recommendations,
        ground_truth=ground_truth,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = _parse_args()
    run_pipeline(
        evaluate=args.evaluate,
        target_user_id=args.user_id,
        skip_cache=args.skip_cache,
        bpr_checkpoint_path=args.bpr_checkpoint_path,
        load_bpr=args.load_bpr,
        save_bpr=not args.no_save_bpr,
        local_data_dir=args.local_data_dir,
    )

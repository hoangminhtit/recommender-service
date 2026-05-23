"""
Step 3 — Generate Top-50 Candidate Books per User.

Uses the trained BPR model to retrieve high-recall candidate books.
Filters out books the user has already interacted with.

Output:
    DataFrame with columns: [user_id, candidate_book_id, mf_score]
"""

from __future__ import annotations

import pandas as pd
from tqdm import tqdm

from src.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


def generate_candidates(bpr_model, interaction_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate top-N BPR candidate books for every user.

    Args:
        bpr_model:       Trained BPR model.
        interaction_df:  Interaction table (used to exclude already-seen books).

    Returns:
        DataFrame with columns: [user_id, candidate_book_id, mf_score]
    """
    cfg = settings.bpr
    num_candidates = cfg.num_candidates
    logger.info("Step 3 — Generating Top-%d BPR candidates per user …", num_candidates)

    user_ids = interaction_df["user_id"].unique().tolist()
    seen_df = interaction_df[["user_id", "book_id"]].copy()
    seen_df = seen_df.rename(columns={"book_id": "candidate_book_id"})

    rows = []
    item_ids = bpr_model.item_ids

    for uid in tqdm(user_ids, desc="Scoring users", unit="user"):
        scores = bpr_model.score_items(uid, item_ids)
        scored = list(zip(item_ids, scores))
        scored.sort(key=lambda x: x[1], reverse=True)
        top = scored[: num_candidates * 2]  # buffer để bù bước filter seen-books
        for bid, score in top:
            rows.append({
                "user_id": uid,
                "candidate_book_id": int(bid),
                "mf_score": float(score),
            })

    candidates_df = pd.DataFrame(rows)

    if candidates_df.empty:
        logger.warning(
            "No candidates generated — interaction_df may be empty after cold-book filtering. "
            "Consider lowering settings.cold_start.min_book_interactions."
        )
        return candidates_df

    logger.info(
        "Candidates generated: %d rows | %d unique users",
        len(candidates_df),
        candidates_df["user_id"].nunique(),
    )

    # Sanity check: ensure no seen books leak through
    before = len(candidates_df)
    if not seen_df.empty and not candidates_df.empty:
        candidates_df = (
            candidates_df
            .merge(seen_df.drop_duplicates(), on=["user_id", "candidate_book_id"], how="left", indicator=True)
        )
        candidates_df = candidates_df[candidates_df["_merge"] == "left_only"].drop(columns=["_merge"])
    removed = before - len(candidates_df)
    if removed:
        logger.warning(
            "Removed %d already-interacted (user, book) pairs from candidates.", removed
        )

    # BUG FIX: cắt lại về đúng num_candidates sau khi filter seen-books
    candidates_df = (
        candidates_df
        .sort_values(["user_id", "mf_score"], ascending=[True, False])
        .groupby("user_id")
        .head(num_candidates)
        .reset_index(drop=True)
    )
    logger.info(
        "Candidates trimmed to Top-%d: %d rows | %d users",
        num_candidates,
        len(candidates_df),
        candidates_df["user_id"].nunique(),
    )

    return candidates_df

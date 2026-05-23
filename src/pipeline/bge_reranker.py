"""
Step 6 — BGE Semantic Reranker.

Reranks BPR Top-50 candidate books using a cross-encoder model
(BAAI/bge-reranker-base) that scores (user_profile, item_description) pairs.

Strategy:
- Batch inference across all (user, candidate) pairs for throughput.
- Return Top-5 per user ranked by semantic relevance score.

Output:
    Dict mapping user_id → list of top-k dicts:
        [{"book_id": int, "mf_score": float, "semantic_score": float, "rank": int}]
"""

from __future__ import annotations

import os
import pandas as pd
from tqdm import tqdm

from src.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


def rerank_candidates(
    candidates_df: pd.DataFrame,
    user_profiles: dict[int, str],
    item_descriptions: dict[int, str],
) -> dict[int, list[dict]]:
    """
    Rerank BPR candidates using the BGE cross-encoder.

    Args:
        candidates_df:    DataFrame [user_id, candidate_book_id, mf_score]
        user_profiles:    Dict user_id → profile string (query side)
        item_descriptions: Dict book_id → description string (document side)

    Returns:
        Dict user_id → ranked list of top-k recommendation dicts.
    """
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    import torch

    cfg = settings.reranker
    device = torch.device(cfg.device)

    logger.info("Step 6 — BGE Semantic Reranking …")
    logger.info(
        "  Model: %s | top_k: %d | batch_size: %d | device: %s",
        cfg.model_name,
        cfg.top_k,
        cfg.batch_size,
        cfg.device,
    )

    # Load model and tokenizer
    logger.info("Loading BGE reranker model …")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(cfg.model_name)
    model.eval()
    model.to(device)
    logger.info("Model loaded on %s.", device)

    # Save checkpoint locally for reuse (best-effort)
    if cfg.checkpoint_path:
        try:
            os.makedirs(cfg.checkpoint_path, exist_ok=True)
            tokenizer.save_pretrained(cfg.checkpoint_path)
            model.save_pretrained(cfg.checkpoint_path)
            logger.info("Saved BGE checkpoint to %s", cfg.checkpoint_path)
        except Exception as exc:
            logger.warning("Failed to save BGE checkpoint: %s", exc)

    # Group candidates by user
    user_ids = candidates_df["user_id"].unique().tolist()
    results: dict[int, list[dict]] = {}

    for uid in tqdm(user_ids, desc="Reranking users", unit="user"):
        user_candidates = candidates_df[candidates_df["user_id"] == uid]

        profile = user_profiles.get(uid)
        if not profile:
            logger.warning("No profile found for user %d — skipping reranking.", uid)
            continue

        pairs: list[tuple[str, str]] = []
        book_ids: list[int] = []
        mf_scores: list[float] = []

        for _, row in user_candidates.iterrows():
            bid = row["candidate_book_id"]
            desc = item_descriptions.get(bid)
            if desc is None:
                logger.debug(
                    "No description for book %d (user %d) — skipping pair.", bid, uid
                )
                continue
            pairs.append((profile, desc))
            book_ids.append(bid)
            mf_scores.append(float(row["mf_score"]))

        if not pairs:
            logger.warning("No valid (profile, description) pairs for user %d.", uid)
            continue

        semantic_scores = _batch_score(
            model=model,
            tokenizer=tokenizer,
            pairs=pairs,
            batch_size=cfg.batch_size,
            device=device,
        )

        # Sort by semantic score descending, return top-k
        ranked = sorted(
            zip(book_ids, mf_scores, semantic_scores),
            key=lambda x: x[2],
            reverse=True,
        )[: cfg.top_k]

        results[uid] = [
            {
                "book_id": book_id,
                "mf_score": round(mf_score, 6),
                "semantic_score": round(sem_score, 6),
                "rank": rank + 1,
            }
            for rank, (book_id, mf_score, sem_score) in enumerate(ranked)
        ]

        logger.debug(
            "User %d → Top-%d books: %s",
            uid,
            cfg.top_k,
            [r["book_id"] for r in results[uid]],
        )

    logger.info(
        "Reranking complete: %d users processed | %d users with results.",
        len(user_ids),
        len(results),
    )
    return results


def _batch_score(
    model,
    tokenizer,
    pairs: list[tuple[str, str]],
    batch_size: int,
    device,
) -> list[float]:
    """
    Score (query, document) pairs in batches.

    Returns:
        List of raw logit scores (higher = more relevant).
    """
    import torch

    all_scores: list[float] = []
    num_batches = (len(pairs) + batch_size - 1) // batch_size

    for i in tqdm(
        range(0, len(pairs), batch_size),
        total=num_batches,
        desc="  BGE batch inference",
        unit="batch",
        leave=False,
    ):
        batch = pairs[i : i + batch_size]
        queries = [p[0] for p in batch]
        docs = [p[1] for p in batch]

        encoded = tokenizer(
            queries,
            docs,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}

        with torch.no_grad():
            logits = model(**encoded).logits.squeeze(-1)
            # BUG FIX: sigmoid để normalize về [0,1], nhất quán với popularity_score
            scores_batch = torch.sigmoid(logits)

        all_scores.extend(scores_batch.cpu().tolist())

    return all_scores

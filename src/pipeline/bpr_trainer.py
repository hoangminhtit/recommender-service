"""
Step 2 — Train BPR Model (Bayesian Personalized Ranking).

Optimizes pairwise ranking for implicit feedback with negative sampling.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import random

import numpy as np
import pandas as pd

from src.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class BPRModel:
    user_factors: np.ndarray
    item_factors: np.ndarray
    user_ids: list[int]
    item_ids: list[int]
    user_index: dict[int, int] = field(init=False)
    item_index: dict[int, int] = field(init=False)

    def __post_init__(self) -> None:
        self.user_index = {uid: idx for idx, uid in enumerate(self.user_ids)}
        self.item_index = {bid: idx for idx, bid in enumerate(self.item_ids)}

    def score_items(self, user_id: int, item_ids: list[int]) -> np.ndarray:
        if user_id not in self.user_index:
            return np.zeros(len(item_ids), dtype=np.float32)
        u_idx = self.user_index[user_id]
        valid_idx = [self.item_index.get(bid) for bid in item_ids]
        scores = []
        for idx in valid_idx:
            if idx is None:
                scores.append(0.0)
            else:
                scores.append(float(self.user_factors[u_idx].dot(self.item_factors[idx])))
        return np.array(scores, dtype=np.float32)


def train_bpr(interaction_df: pd.DataFrame) -> BPRModel:
    """Train a BPR model using implicit feedback interactions."""
    cfg = settings.bpr
    rng = random.Random(cfg.seed)

    user_ids = sorted(interaction_df["user_id"].unique().tolist())
    item_ids = sorted(interaction_df["book_id"].unique().tolist())
    user_index = {uid: idx for idx, uid in enumerate(user_ids)}
    item_index = {bid: idx for idx, bid in enumerate(item_ids)}

    logger.info(
        "Step 2 — Training BPR model … factors=%d | epochs=%d | lr=%.4f | reg=%.4f",
        cfg.factors,
        cfg.epochs,
        cfg.learning_rate,
        cfg.reg,
    )
    logger.info("  Input: %d (user, book) pairs", len(interaction_df))

    user_factors = 0.1 * np.random.randn(len(user_ids), cfg.factors).astype(np.float32)
    item_factors = 0.1 * np.random.randn(len(item_ids), cfg.factors).astype(np.float32)

    user_pos: dict[int, set[int]] = {}
    for uid, group in interaction_df.groupby("user_id"):
        user_pos[uid] = set(group["book_id"].tolist())

    interactions = list(zip(interaction_df["user_id"], interaction_df["book_id"]))

    for epoch in range(cfg.epochs):
        rng.shuffle(interactions)
        for uid, pos_bid in interactions:
            u_idx = user_index[uid]
            i_idx = item_index[pos_bid]

            for _ in range(cfg.num_negatives):
                neg_bid = _sample_negative(uid, item_ids, user_pos, rng)
                j_idx = item_index[neg_bid]

                u_vec = user_factors[u_idx]
                i_vec = item_factors[i_idx]
                j_vec = item_factors[j_idx]

                x_uij = float(np.dot(u_vec, i_vec - j_vec))
                sigmoid = 1.0 / (1.0 + np.exp(-x_uij))
                grad = 1.0 - sigmoid

                user_factors[u_idx] = u_vec + cfg.learning_rate * (
                    grad * (i_vec - j_vec) - cfg.reg * u_vec
                )
                item_factors[i_idx] = i_vec + cfg.learning_rate * (
                    grad * u_vec - cfg.reg * i_vec
                )
                item_factors[j_idx] = j_vec + cfg.learning_rate * (
                    -grad * u_vec - cfg.reg * j_vec
                )

        logger.info("  Epoch %d/%d complete", epoch + 1, cfg.epochs)

    return BPRModel(
        user_factors=user_factors,
        item_factors=item_factors,
        user_ids=user_ids,
        item_ids=item_ids,
    )


def _sample_negative(
    user_id: int,
    all_items: list[int],
    user_pos: dict[int, set[int]],
    rng: random.Random,
) -> int:
    pos_set = user_pos.get(user_id, set())
    if len(pos_set) >= len(all_items):
        return rng.choice(all_items)
    while True:
        candidate = rng.choice(all_items)
        if candidate not in pos_set:
            return candidate


def save_bpr_model(model: BPRModel, path: str) -> None:
    """Save BPR model checkpoint to a .npz file."""
    if not path:
        raise ValueError("BPR checkpoint path is empty.")
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Saving BPR model checkpoint to %s", path)
    np.savez(
        path,
        user_factors=model.user_factors,
        item_factors=model.item_factors,
        user_ids=np.array(model.user_ids, dtype=np.int64),
        item_ids=np.array(model.item_ids, dtype=np.int64),
    )


def load_bpr_model(path: str) -> BPRModel:
    """Load BPR model checkpoint from a .npz file."""
    if not path:
        raise ValueError("BPR checkpoint path is empty.")
    logger.info("Loading BPR model checkpoint from %s", path)
    data = np.load(path, allow_pickle=False)
    return BPRModel(
        user_factors=data["user_factors"],
        item_factors=data["item_factors"],
        user_ids=data["user_ids"].astype(int).tolist(),
        item_ids=data["item_ids"].astype(int).tolist(),
    )
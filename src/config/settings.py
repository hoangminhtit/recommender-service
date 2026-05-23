"""
Central configuration for the recommender service.
All hyperparameters and environment variables are defined here.
Load via environment variables or .env file.
"""

import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


# ---------------------------------------------------------------------------
# Databricks connection (optional for local runs)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DatabricksConfig:
    server_hostname: str = field(
        default_factory=lambda: os.getenv("DATABRICKS_SERVER_HOSTNAME", "")
    )
    http_path: str = field(
        default_factory=lambda: os.getenv("DATABRICKS_HTTP_PATH", "")
    )
    access_token: str = field(
        default_factory=lambda: os.getenv("DATABRICKS_ACCESS_TOKEN", "")
    )
    catalog: str = field(
        default_factory=lambda: os.getenv("DATABRICKS_CATALOG", "main")
    )
    schema: str = field(
        default_factory=lambda: os.getenv("DATABRICKS_SCHEMA", "gold")
    )


# ---------------------------------------------------------------------------
# BPR hyperparameters
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BPRConfig:
    factors: int = 64               # Latent factors
    epochs: int = 30                # Training epochs
    learning_rate: float = 0.05     # SGD learning rate
    reg: float = 0.0025             # L2 regularization
    num_negatives: int = 1          # Negatives per positive
    num_candidates: int = 50        # Top-N candidates to retrieve per user
    seed: int = 42
    checkpoint_path: str = field(
        default_factory=lambda: os.getenv("BPR_CHECKPOINT_PATH", "models/bpr/model.npz")
    )


# ---------------------------------------------------------------------------
# ALS hyperparameters (alternative MF, not used in main pipeline)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ALSConfig:
    rank: int = 64
    max_iter: int = 20
    reg_param: float = 0.01
    alpha: float = 40.0
    checkpoint_path: str = field(
        default_factory=lambda: os.getenv("ALS_CHECKPOINT_PATH", "models/als")
    )


# ---------------------------------------------------------------------------
# Time decay
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TimeDecayConfig:
    lambda_rate: float = 0.05      # Decay rate (months as time unit)
    # Interaction base scores
    purchase_score: int = 10
    cart_score: int = 3
    positive_review_score: int = 5
    positive_review_threshold: float = 4.0


# ---------------------------------------------------------------------------
# BGE Reranker
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RerankerConfig:
    model_name: str = "BAAI/bge-reranker-base"
    top_k: int = 5                 # Final recommendations returned
    batch_size: int = 32           # Pairs per inference batch
    device: str = field(
        default_factory=lambda: os.getenv("RERANKER_DEVICE", "cpu")
    )
    checkpoint_path: str = field(
        default_factory=lambda: os.getenv("BGE_CHECKPOINT_PATH", "models/bge-reranker")
    )


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CacheConfig:
    ttl_seconds: int = 86_400      # 24 hours max TTL
    redis_url: str = field(
        default_factory=lambda: os.getenv("REDIS_URL", "redis://localhost:6379")
    )


# ---------------------------------------------------------------------------
# Cold-start thresholds
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ColdStartConfig:
    # min_interactions_for_bpr đã xóa (trùng với min_interactions_for_full_bpr)
    min_interactions_for_full_bpr: int = 5  # At or above → full BPR; below → hybrid
    min_book_interactions: int = 3          # BUG FIX: hạ từ 10→3 để phù hợp mock data nhỏ
    popularity_pool_size: int = 10          # Size of popularity fallback pool


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EvaluationConfig:
    recall_k: int = 50
    ndcg_k: int = 5
    map_k: int = 5
    mrr_k: int = 5
    # Temporal split months (1-indexed)
    train_end_month: int = 10
    validation_month: int = 11
    test_month: int = 12


# ---------------------------------------------------------------------------
# Aggregated app config (single import point)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AppConfig:
    databricks: DatabricksConfig = field(default_factory=DatabricksConfig)
    bpr: BPRConfig = field(default_factory=BPRConfig)
    als: ALSConfig = field(default_factory=ALSConfig)          # alternative MF
    time_decay: TimeDecayConfig = field(default_factory=TimeDecayConfig)
    reranker: RerankerConfig = field(default_factory=RerankerConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    cold_start: ColdStartConfig = field(default_factory=ColdStartConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)


# Singleton — import this everywhere
settings = AppConfig()

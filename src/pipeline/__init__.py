"""
Pipeline package — exposes all step modules.
Import order reflects the pipeline execution order.
"""

from .interaction_builder import build_interaction_table
from .bpr_trainer import train_bpr, save_bpr_model, load_bpr_model
from .candidate_generator import generate_candidates
from .user_profile_builder import build_user_profiles
from .item_description_builder import build_item_descriptions
from .bge_reranker import rerank_candidates

__all__ = [
    "build_interaction_table",
    "train_bpr",
    "save_bpr_model",
    "load_bpr_model",
    "generate_candidates",
    "build_user_profiles",
    "build_item_descriptions",
    "rerank_candidates",
]

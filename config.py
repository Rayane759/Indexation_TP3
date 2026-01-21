"""
Search Engine Configuration
Default ranking weights and parameters
"""

from typing import Dict


class RankingWeights:
    """Default ranking weights for multi-signal ranking."""
    
    DEFAULT = {
        'title_presence': 2.0,
        'title_bm25': 2.5,
        'title_position': 0.3,
        'description_bm25': 1.0,
        'description_position': 0.2,
        'exact_match': 1.5,
        'frequency': 0.5,
        'reviews': 0.8,
        'review_recency': 0.3,
    }


class BM25Parameters:
    """BM25 algorithm parameters."""
    k1: float = 1.5  # Term frequency saturation
    b: float = 0.75  # Length normalization

# recommender.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

@dataclass
class PenaltyConfig:
    reserve_groups: Dict[str, List[str]]
    penalties: Dict[str, float]

DEFAULT_PENALTY = PenaltyConfig(
    reserve_groups={
        "reserve": ["colistine", "IPM"]
    },
    penalties={
        "reserve": 0.15
    }
)

def recommend(
        model,
        long_df: pd.DataFrame,
        bacteria: str,
        top_k: int = 5,
        penalty_config: PenaltyConfig = DEFAULT_PENALTY
) -> pd.DataFrame:
    
    df = long_df[long_df["bacteria"] == bacteria].copy()

    if df.empty:
        return pd.DataFrame({"message": ["No data for selected bacteria"]})
    
    antibiotics = df["antibiotic"].unique()
    results = []

    reserve_set = set(penalty_config.reserve_groups.get("reserve", []))

    for ab in antibiotics:
        subset = df[df["antibiotic"] == ab].copy()

        X = subset.drop(columns=["susceptible"], errors="ignore")

        probs = model.predict_proba(X)[:, 1]
        score = float(probs.mean())

        if ab in reserve_set:
            score -= float(penalty_config.penalties.get("reserve", 0.0))
        results.append((ab, score))

    result_df = pd.DataFrame(results, columns=["antibiotic", "score"])
    result_df = result_df.sort_values("score", ascending=False).reset_index(drop=True)
    
    result_df["rank"] = range(1, len(result_df) + 1)

    return result_df.head(top_k)
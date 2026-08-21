import numpy as np
import pandas as pd


def race_softmax(scores: np.ndarray) -> np.ndarray:
    """Apply softmax to raw model scores within a race, guaranteeing probabilities sum to 1."""
    shifted = scores - np.max(scores)
    exp_scores = np.exp(shifted)
    return exp_scores / exp_scores.sum()


def apply_race_softmax(val_df: pd.DataFrame, raw_scores: np.ndarray, prob_col: str = "prob") -> pd.DataFrame:
    """
    For each race in val_df, apply softmax to the raw_scores for drivers in that race.
    Returns val_df with prob_col added and asserts sums are ~1.0.
    """
    val_df = val_df.reset_index(drop=True)
    val_df[prob_col] = 0.0

    for race_id, group in val_df.groupby("race_id"):
        indices = group.index.tolist()
        race_scores = raw_scores[indices]
        probs = race_softmax(race_scores)
        val_df.loc[indices, prob_col] = probs

    # Assertion check
    for race_id, group in val_df.groupby("race_id"):
        total = group[prob_col].sum()
        assert abs(total - 1.0) < 1e-6, f"Race {race_id} probabilities sum to {total}, not 1.0"

    return val_df

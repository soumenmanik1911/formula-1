import os
import sys
from typing import Dict

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sqlalchemy.orm import Session
from xgboost import XGBClassifier

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from app.database import SessionLocal
from app.models import Driver
from app.ml.prepare_data import prepare_data
from app.ml.utils import apply_race_softmax


def _load_driver_names() -> Dict[int, str]:
    db: Session = SessionLocal()
    try:
        drivers = db.query(Driver).all()
        return {d.id: d.full_name for d in drivers}
    finally:
        db.close()


def train_winner_model() -> Dict:
    data = prepare_data()

    X_train = data["X_train"]
    y_train = (data["y_train"] == 1).astype(int)
    X_val = data["X_val"]
    y_val = (data["y_val"] == 1).astype(int)
    val_df = data["val_df"].copy()

    driver_names = _load_driver_names()

    n_pos = int(y_train.sum())
    n_neg = int(len(y_train) - n_pos)
    scale_pos_weight = n_neg / max(n_pos, 1)

    model = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        n_jobs=4,
        random_state=42,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    # Get raw margins and apply per-race softmax for calibrated probabilities
    raw_scores = model.predict(X_val, output_margin=True)
    val_df = apply_race_softmax(val_df, raw_scores, prob_col="win_prob")

    # Print per-race sum confirmation
    print("=== Per-Race Win Probability Sums (softmax calibrated) ===")
    race_sums = val_df.groupby("race_id")["win_prob"].sum()
    print(race_sums.describe().to_string())
    print(f"All races within 1e-6 of 1.0: {all(abs(s - 1.0) < 1e-6 for s in race_sums)}")

    # Per-race evaluation
    val_df = val_df.reset_index(drop=True)

    per_race = []
    for race_id, group in val_df.groupby("race_id"):
        actual_winner = group[group["finish_position"] == 1]
        actual_winner_id = int(actual_winner["driver_id"].values[0]) if not actual_winner.empty else None
        actual_winner_name = driver_names.get(actual_winner_id, "Unknown") if actual_winner_id is not None else "Unknown"

        top_driver = group.loc[group["win_prob"].idxmax()]
        top_driver_id = int(top_driver["driver_id"])
        top_driver_name = driver_names.get(top_driver_id, "Unknown")

        correct = int(actual_winner_id == top_driver_id) if actual_winner_id is not None else 0

        if actual_winner_id is not None:
            winner_prob = float(group[group["driver_id"] == actual_winner_id]["win_prob"].values[0])
            winner_prob = max(min(winner_prob, 1 - 1e-15), 1e-15)
            winner_log_loss = float(-np.log(winner_prob))
        else:
            winner_log_loss = None

        season = int(group["season"].values[0])
        round_num = int(group["round"].values[0])

        per_race.append({
            "season": season,
            "round": round_num,
            "race_id": int(race_id),
            "actual_winner_id": actual_winner_id,
            "actual_winner": actual_winner_name,
            "predicted_winner_id": top_driver_id,
            "predicted_winner": top_driver_name,
            "correct": correct,
            "winner_log_loss": round(winner_log_loss, 4) if winner_log_loss is not None else None,
        })

    per_race_df = pd.DataFrame(per_race)

    overall_log_loss = log_loss(y_val, val_df["win_prob"].values)
    top1_accuracy = float(per_race_df["correct"].mean())

    print(f"\n=== 2025 Validation Metrics (softmax calibrated) ===")
    print(f"Log Loss: {overall_log_loss:.4f}")
    print(f"Top-1 Accuracy (per race): {top1_accuracy:.4f}")
    print(f"Correct races: {int(per_race_df['correct'].sum())} / {len(per_race_df)}")
    print()
    print(per_race_df.to_string(index=False))

    # Save model
    artifacts_dir = os.path.join(os.path.dirname(__file__), "artifacts")
    os.makedirs(artifacts_dir, exist_ok=True)
    model_path = os.path.join(artifacts_dir, "winner_model.joblib")
    joblib.dump(model, model_path)
    print(f"\nModel saved to {model_path}")

    # Feature importances
    importances = pd.DataFrame({
        "feature": data["feature_cols"],
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)
    print("\n=== Feature Importances ===")
    print(importances.to_string(index=False))

    return {
        "model": model,
        "per_race": per_race_df,
        "overall_log_loss": overall_log_loss,
        "top1_accuracy": top1_accuracy,
        "scale_pos_weight": scale_pos_weight,
    }


if __name__ == "__main__":
    train_winner_model()

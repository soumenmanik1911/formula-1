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
from app.ml.prepare_data import prepare_data, FEATURE_COLS
from app.ml.utils import apply_race_softmax, race_softmax


def _load_driver_names() -> Dict[int, str]:
    db: Session = SessionLocal()
    try:
        drivers = db.query(Driver).all()
        return {d.id: d.full_name for d in drivers}
    finally:
        db.close()


def train_podium_model() -> Dict:
    data = prepare_data()

    X_train = data["X_train"]
    y_train = (data["y_train"] <= 3).astype(int)
    X_val = data["X_val"]
    y_val = (data["y_val"] <= 3).astype(int)
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

    # Use raw margins + softmax for calibrated podium probabilities
    raw_scores = model.predict(X_val, output_margin=True)
    val_df = val_df.reset_index(drop=True)
    val_df = apply_race_softmax(val_df, raw_scores, prob_col="podium_prob")
    val_df["podium_prob"] = val_df["podium_prob"] * 3.0

    # Print per-race sum confirmation
    print("=== Per-Race Podium Probability Sums (softmax calibrated, scaled x3) ===")
    race_sums = val_df.groupby("race_id")["podium_prob"].sum()
    print(race_sums.describe().to_string())
    print(f"All races within 0.1 of 3.0: {all(abs(s - 3.0) < 1e-6 for s in race_sums)}")

    per_race = []
    for race_id, group in val_df.groupby("race_id"):
        actual_podium = group[group["finish_position"] <= 3]
        actual_podium_ids = set(actual_podium["driver_id"].values)

        top_driver = group.loc[group["podium_prob"].idxmax()]
        top_driver_id = int(top_driver["driver_id"])
        top_driver_name = driver_names.get(top_driver_id, "Unknown")
        top1_correct = int(top_driver_id in actual_podium_ids)

        actual_winner = group[group["finish_position"] == 1]
        actual_winner_id = int(actual_winner["driver_id"].values[0]) if not actual_winner.empty else None
        actual_winner_name = driver_names.get(actual_winner_id, "Unknown") if actual_winner_id is not None else "Unknown"

        if actual_winner_id is not None:
            winner_prob = float(group[group["driver_id"] == actual_winner_id]["podium_prob"].values[0])
            winner_prob = max(min(winner_prob, 1 - 1e-15), 1e-15)
            winner_log_loss = float(-np.log(winner_prob))
        else:
            winner_log_loss = None

        top3 = group.nlargest(3, "podium_prob")
        top3_ids = set(top3["driver_id"].values)
        podium_precision = len(top3_ids & actual_podium_ids) / 3.0

        season = int(group["season"].values[0])
        round_num = int(group["round"].values[0])

        per_race.append({
            "season": season,
            "round": round_num,
            "race_id": int(race_id),
            "actual_podium_ids": sorted(actual_podium_ids),
            "top1_predicted_id": top_driver_id,
            "top1_predicted": top_driver_name,
            "top1_on_podium": top1_correct,
            "winner_log_loss": round(winner_log_loss, 4) if winner_log_loss is not None else None,
            "podium_precision_top3": round(podium_precision, 4),
        })

    per_race_df = pd.DataFrame(per_race)

    overall_log_loss = log_loss(y_val, model.predict_proba(X_val)[:, 1])
    top1_accuracy = float(per_race_df["top1_on_podium"].mean())
    avg_podium_precision = float(per_race_df["podium_precision_top3"].mean())

    print(f"\n=== 2025 Podium Validation Metrics (softmax calibrated) ===")
    print(f"Log Loss: {overall_log_loss:.4f}")
    print(f"Top-1 Podium Accuracy: {top1_accuracy:.4f}")
    print(f"Avg Top-3 Podium Precision: {avg_podium_precision:.4f}")
    print(f"Correct races (top1 on podium): {int(per_race_df['top1_on_podium'].sum())} / {len(per_race_df)}")
    print()
    print_cols = ["season", "round", "actual_podium_ids", "top1_predicted", "top1_on_podium", "winner_log_loss", "podium_precision_top3"]
    print(per_race_df[print_cols].to_string(index=False))

    artifacts_dir = os.path.join(os.path.dirname(__file__), "artifacts")
    os.makedirs(artifacts_dir, exist_ok=True)
    model_path = os.path.join(artifacts_dir, "podium_model.joblib")
    joblib.dump(model, model_path)
    print(f"\nModel saved to {model_path}")

    importances = pd.DataFrame({
        "feature": FEATURE_COLS,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)
    print("\n=== Feature Importances ===")
    print(importances.to_string(index=False))

    return {
        "model": model,
        "per_race": per_race_df,
        "overall_log_loss": overall_log_loss,
        "top1_accuracy": top1_accuracy,
        "avg_podium_precision": avg_podium_precision,
        "scale_pos_weight": scale_pos_weight,
    }


if __name__ == "__main__":
    train_podium_model()

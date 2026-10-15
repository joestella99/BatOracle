"""Chronological backtest comparing an analog-augmented model with baselines."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from .data import fetch_statcast, normalize_plate_appearances
from .matchup import chronological_features, FEATS


def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    p = np.clip(np.asarray(p, dtype=float), 1e-5, 1 - 1e-5)
    results = {"count": len(y), "hit_rate": round(float(np.mean(y)), 4), "brier": round(float(brier_score_loss(y, p)), 5), "log_loss": round(float(log_loss(y, p, labels=[0, 1])), 5)}
    results["auc"] = round(float(roc_auc_score(y, p)), 5) if len(np.unique(y)) == 2 else None
    return results


def backtest(events: pd.DataFrame, train_start: str, test_start: str, test_end: str, per_day: int = 30, k: int = 3) -> tuple[dict, pd.DataFrame]:
    """Train on pre-test dates; test on later days, updating analog history daily."""
    e = events.sort_values("game_date")
    train_start, test_start, test_end = map(pd.Timestamp, (train_start, test_start, test_end))
    if not train_start < test_start <= test_end:
        raise ValueError("Require train_start < test_start <= test_end")
    if e.empty:
        raise ValueError("No event data")
    days = e.loc[(e.game_date >= train_start) & (e.game_date <= test_end), "game_date"].drop_duplicates().tolist()
    feats = chronological_features(e, dates=days, max_rows_per_day=per_day, k=k)
    training = feats[(feats.date >= train_start) & (feats.date < test_start)]
    testing = feats[(feats.date >= test_start) & (feats.date <= test_end)]
    if len(training) < 100 or len(testing) < 10 or training.hit.nunique() < 2:
        raise ValueError(f"Insufficient PA features for realistic evaluation: train={len(training)}, test={len(testing)}. Fetch more history.")
    x_train = training[list(FEATS)]
    x_test = testing[list(FEATS)]
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    model.fit(x_train, training.hit)
    pred = model.predict_proba(x_test)[:, 1]
    # Baselines are also strictly historical, no outcomes from the scored PA/day.
    base_league = testing.league_hit_rate.to_numpy(dtype=float)
    base_analog = testing.analog_hit_rate.to_numpy(dtype=float)
    y = testing.hit.to_numpy(dtype=int)
    summary = {"training_events": len(training), "test_events": len(testing), "train_start": str(train_start.date()), "test_start": str(test_start.date()), "test_end": str(test_end.date()), "metrics": {
        "league_to_date": metrics(y, base_league),
        "similar_matchups_only": metrics(y, base_analog),
        "analogs_plus_logistic_regression": metrics(y, pred),
    }}
    predictions = testing[["date", "batter", "pitcher", "hit"]].copy()
    predictions["league_probability"] = base_league
    predictions["analog_probability"] = base_analog
    predictions["model_probability"] = pred
    return summary, predictions


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Bat Oracle 2.0: time-safe historical matchup backtest")
    parser.add_argument("--start", default="2023-03-30", help="Start of data collection, ideally a season before training")
    parser.add_argument("--end", default="2024-05-31")
    parser.add_argument("--train-start", default="2024-04-01")
    parser.add_argument("--test-start", default="2024-05-01")
    parser.add_argument("--test-end", default="2024-05-31")
    parser.add_argument("--per-day", type=int, default=30)
    parser.add_argument("--nearest", type=int, default=3)
    parser.add_argument("--input-csv", help="Existing complete Statcast pitch data, instead of download")
    parser.add_argument("--output-dir", default="data/backtests")
    args = parser.parse_args()
    raw = pd.read_csv(args.input_csv) if args.input_csv else fetch_statcast(args.start, args.end)
    events = normalize_plate_appearances(raw)
    summary, predictions = backtest(events, args.train_start, args.test_start, args.test_end, per_day=args.per_day, k=args.nearest)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(json.dumps(summary, indent=2))
    predictions.to_csv(output / "predictions.csv", index=False)
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()

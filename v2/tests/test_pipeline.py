from datetime import date, timedelta
import numpy as np
import pandas as pd
import pytest
from bat_oracle.data import normalize_plate_appearances
from bat_oracle.matchup import matchup_features, chronological_features
from bat_oracle.simulate import backtest


def pitches_fixture():
    data = [
        {"game_date":"2024-04-01", "game_pk":1, "at_bat_number":1, "pitch_number":1, "batter":11, "pitcher":91, "events":None, "stand":"L", "p_throws":"R", "game_type":"R"},
        {"game_date":"2024-04-01", "game_pk":1, "at_bat_number":1, "pitch_number":4, "batter":11, "pitcher":91, "events":"single", "stand":"L", "p_throws":"R", "game_type":"R"},
        {"game_date":"2024-04-01", "game_pk":1, "at_bat_number":2, "pitch_number":2, "batter":12, "pitcher":91, "events":"strikeout", "stand":"R", "p_throws":"R", "game_type":"R"},
        {"game_date":"2024-04-01", "game_pk":1, "at_bat_number":3, "pitch_number":1, "batter":11, "pitcher":91, "events":"home_run", "stand":"L", "p_throws":"R", "game_type":"S"},
        {"game_date":"2024-04-01", "game_pk":1, "at_bat_number":4, "pitch_number":2, "batter":12, "pitcher":92, "events":"walk", "stand":"R", "p_throws":"L", "game_type":"R"},
    ]
    return pd.DataFrame(data)


def test_one_pa_per_hit_and_regular_season():
    result = normalize_plate_appearances(pitches_fixture())
    assert len(result) == 3
    assert list(result.hit) == [1, 0, 0]
    assert list(result.walk) == [0, 0, 1]
    assert result[["game_pk","at_bat_number"]].duplicated().sum() == 0


def synthetic_events():
    """Synthetic fixture tests mechanics; it is NOT evidence of baseball accuracy."""
    rng = np.random.default_rng(18)
    base = date(2024, 1, 1)
    events = []
    for day in range(80):
        for n in range(25):
            batter = 100 + (n + day) % 8
            pitcher = 200 + (n * 3 + day) % 6
            hand = "L" if batter % 2 else "R"
            p_hand = "L" if pitcher % 2 else "R"
            p_hit = .20 + .02 * (batter % 4) + (.035 if hand != p_hand else 0)
            outcome = int(rng.random() < p_hit)
            events.append({"game_date": pd.Timestamp(base + timedelta(days=day)), "game_pk":day + 1, "at_bat_number":n + 1, "batter":batter, "pitcher":pitcher, "stand":hand, "p_throws":p_hand, "hit":outcome, "strikeout":int(rng.random() < .22), "walk":int(rng.random() < .08), "home_run":int(outcome and rng.random() < .12), "events":"single" if outcome else "field_out"})
    return pd.DataFrame(events)


def test_analogs_use_past_only():
    rows = synthetic_events()
    history = rows[rows.game_date < "2024-02-01"]
    r = rows[rows.game_date == "2024-02-01"].iloc[0]
    feat1, evidence = matchup_features(history, int(r.batter), int(r.pitcher), r.stand, r.p_throws)
    # Change the current day's real outcomes: it must not affect the same-day predictions.
    poisoned = rows.copy()
    poisoned.loc[poisoned.game_date == "2024-02-01", "hit"] = 1
    feat2 = chronological_features(poisoned, dates=["2024-02-01"], max_rows_per_day=1)
    assert feat1 == {k: feat2.iloc[0][k] for k in feat1}
    assert len(evidence["batter_comparisons"]) <= 4
    assert len(evidence["pitcher_comparisons"]) <= 4
    assert all(history[history.batter == player].stand.iloc[0] == r.stand for player, _ in evidence["batter_comparisons"])
    assert all(history[history.pitcher == player].p_throws.iloc[0] == r.p_throws for player, _ in evidence["pitcher_comparisons"])


def test_chronological_backtest_mechanics():
    events = synthetic_events()
    result, preds = backtest(events, "2024-01-08", "2024-03-01", "2024-03-15", per_day=5)
    assert result["training_events"] > 100
    assert result["test_events"] > 10
    assert preds.date.min() >= pd.Timestamp("2024-03-01")
    assert preds.date.max() <= pd.Timestamp("2024-03-15")
    assert set(result["metrics"]) == {"league_to_date", "similar_matchups_only", "analogs_plus_logistic_regression"}
    assert preds.model_probability.between(0, 1).all()


def test_bad_dates():
    with pytest.raises(ValueError):
        backtest(synthetic_events(), "2024-05-01", "2024-04-01", "2024-03-01")

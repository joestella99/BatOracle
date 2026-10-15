"""Time-safe player comparisons: batter A/B/C versus pitcher A/B/C."""
from __future__ import annotations

import numpy as np
import pandas as pd

RATE_NAMES = ("hit", "strikeout", "walk", "home_run")
FEATS = (
    "b_hit", "b_k", "b_bb", "b_hr", "b_pa", "p_hit", "p_k", "p_bb", "p_hr", "p_pa",
    "analog_hit_rate", "analog_n_eff", "analog_raw_n", "hand_same", "league_hit_rate",
)


def _profiles(history: pd.DataFrame, role: str) -> pd.DataFrame:
    hand_col = "stand" if role == "batter" else "p_throws"
    profiles = history.groupby(role)[list(RATE_NAMES)].agg(["sum", "count"])
    # Use all plate appearances to estimate player skills, not season-end totals.
    profile = pd.DataFrame(index=profiles.index)
    profile["pa"] = profiles[("hit", "count")]
    for stat in RATE_NAMES:
        league_rate = float(history[stat].mean())
        # Partial pooling avoids pretending 1-for-1 batters have true 1.000 hit talent.
        profile[stat] = (profiles[(stat, "sum")] + 80 * league_rate) / (profile["pa"] + 80)
    profile["hand"] = history.groupby(role)[hand_col].agg(lambda s: s.mode().iat[0])
    return profile


def _nearest(profiles: pd.DataFrame, player_id: int, hand: str, count: int = 3) -> list[tuple[int, float]]:
    """Return target plus closest same-handed neighbors, excluding low-sample candidates."""
    if player_id not in profiles.index:
        return []
    vector = profiles.loc[player_id, list(RATE_NAMES)].to_numpy(dtype=float)
    peers = profiles[(profiles["hand"] == hand) & (profiles["pa"] >= 25)].copy()
    if player_id not in peers.index:
        peers = pd.concat([peers, profiles.loc[[player_id]]])
    if peers.empty:
        return [(int(player_id), 1.0)]
    values = peers[list(RATE_NAMES)].to_numpy(dtype=float)
    # Scale statistical rates so K% does not trivially dominate HR%.
    scales = np.maximum(profiles[list(RATE_NAMES)].std().to_numpy(dtype=float), np.array([.025, .04, .03, .01]))
    distance = np.sqrt(np.sum(((values - vector) / scales) ** 2, axis=1))
    peers["distance"] = distance
    peers = peers.sort_values(["distance", "pa"], ascending=[True, False])
    # Keep the exact target even if a numerical tie sorts a comparison first.
    selected = [int(player_id)] + [int(i) for i in peers.index if int(i) != int(player_id)][:count]
    dist = dict(zip(peers.index.astype(int), peers["distance"]))
    return [(i, float(np.exp(-0.5 * dist.get(i, 0)))) for i in selected]


def matchup_features(history: pd.DataFrame, batter: int, pitcher: int, stand: str, p_throws: str, k: int = 3, profiles: tuple | None = None) -> tuple[dict, dict]:
    """Only `history` is consulted; caller guarantees it precedes prediction date."""
    if history.empty:
        raise ValueError("Need preceding plate appearances for a prediction")
    bat, pit, global_rates = profiles if profiles is not None else (
        _profiles(history, "batter"), _profiles(history, "pitcher"), history[list(RATE_NAMES)].mean()
    )
    b = bat.loc[batter] if batter in bat.index else None
    p = pit.loc[pitcher] if pitcher in pit.index else None
    features = {}
    for prefix, row in (("b", b), ("p", p)):
        for stat, suffix in (("hit", "hit"), ("strikeout", "k"), ("walk", "bb"), ("home_run", "hr")):
            features[f"{prefix}_{suffix}"] = float(row[stat] if row is not None else global_rates[stat])
        features[f"{prefix}_pa"] = float(row["pa"] if row is not None else 0)
    bpeers = _nearest(bat, batter, stand, k)
    ppeers = _nearest(pit, pitcher, p_throws, k)
    selected_b = dict(bpeers)
    selected_p = dict(ppeers)
    sub = history[history["batter"].isin(selected_b) & history["pitcher"].isin(selected_p)]
    prior = float(history["hit"].mean())
    if len(sub):
        weights = sub["batter"].map(selected_b).to_numpy() * sub["pitcher"].map(selected_p).to_numpy()
        effective_n = float(weights.sum())
        # 25 pseudo-observations strongly regularize sparse analogous head-to-heads.
        rate = float(((weights * sub["hit"].to_numpy()).sum() + 25 * prior) / (effective_n + 25))
    else:
        effective_n, rate = 0.0, prior
    features.update(analog_hit_rate=rate, analog_n_eff=effective_n, analog_raw_n=float(len(sub)), hand_same=float(stand == p_throws), league_hit_rate=prior)
    details = {"batter_comparisons": bpeers, "pitcher_comparisons": ppeers, "matchup_observations": int(len(sub))}
    return features, details


def chronological_features(events: pd.DataFrame, dates: list[str] | None = None, max_rows_per_day: int | None = None, k: int = 3) -> pd.DataFrame:
    """Build out-of-time features; same-day games are invisible to one another."""
    e = events.sort_values(["game_date", "game_pk", "at_bat_number"], kind="stable").reset_index(drop=True)
    days = list(pd.to_datetime(dates)) if dates is not None else list(e["game_date"].unique())
    parts = []
    for day in days:
        day = pd.Timestamp(day)
        history = e[e["game_date"] < day]
        current = e[e["game_date"] == day]
        if history.empty or current.empty:
            continue
        if max_rows_per_day is not None:
            current = current.head(max_rows_per_day)
        # Each pair has the same pre-day information; memoize expensive comparisons.
        memo = {}
        profiles = (_profiles(history, "batter"), _profiles(history, "pitcher"), history[list(RATE_NAMES)].mean())
        rows = []
        for r in current.itertuples(index=False):
            key = (int(r.batter), int(r.pitcher), r.stand, r.p_throws)
            if key not in memo:
                memo[key] = matchup_features(history, *key, k=k, profiles=profiles)[0]
            rows.append({**memo[key], "date": day, "hit": int(r.hit), "batter": int(r.batter), "pitcher": int(r.pitcher)})
        parts.append(pd.DataFrame(rows))
    if not parts:
        return pd.DataFrame(columns=[*FEATS, "date", "hit", "batter", "pitcher"])
    return pd.concat(parts, ignore_index=True)

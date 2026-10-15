"""Resumable Statcast ingestion and one-row-per-plate-appearance normalization."""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
import pandas as pd

HITS = {"single", "double", "triple", "home_run"}
STRIKEOUTS = {"strikeout", "strikeout_double_play"}
WALKS = {"walk", "intent_walk"}
REQUIRED = {"game_date", "game_pk", "at_bat_number", "pitch_number", "batter", "pitcher", "events", "stand", "p_throws"}


def fetch_statcast(start: str, end: str, cache_dir: str = "data/raw", refresh: bool = False) -> pd.DataFrame:
    """Download in <=3-day chunks and cache before processing. Requires pybaseball."""
    from pybaseball import statcast
    first, last = date.fromisoformat(start), date.fromisoformat(end)
    if last < first:
        raise ValueError("end must be >= start")
    base = Path(cache_dir)
    base.mkdir(parents=True, exist_ok=True)
    pieces = []
    cursor = first
    while cursor <= last:
        chunk_end = min(cursor + timedelta(days=2), last)
        path = base / f"statcast_{cursor}_{chunk_end}.pkl"
        if refresh or not path.exists():
            rows = statcast(start_dt=str(cursor), end_dt=str(chunk_end), verbose=False, parallel=False)
            if rows is None:
                rows = pd.DataFrame()  # Off-season and off days are valid empty chunks.
            rows.to_pickle(path)
        pieces.append(pd.read_pickle(path))
        cursor = chunk_end + timedelta(days=1)
    nonempty = [piece for piece in pieces if not piece.empty]
    if not nonempty:
        raise ValueError(f"No Statcast records were returned for {start}..{end}")
    return pd.concat(nonempty, ignore_index=True)


def normalize_plate_appearances(raw: pd.DataFrame) -> pd.DataFrame:
    """Keep only terminal plate-appearance events, never individual pitches as labels."""
    missing = REQUIRED - set(raw.columns)
    if missing:
        raise ValueError(f"Missing Statcast columns: {sorted(missing)}")
    df = raw.copy()
    if "game_type" in df.columns:
        df = df[df["game_type"] == "R"]
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce").dt.normalize()
    for key in ["game_pk", "at_bat_number", "pitch_number", "batter", "pitcher"]:
        df[key] = pd.to_numeric(df[key], errors="coerce")
    df = df.dropna(subset=["game_date", "game_pk", "at_bat_number", "pitch_number", "batter", "pitcher", "events", "stand", "p_throws"])
    df = df[df["stand"].isin(["L", "R"]) & df["p_throws"].isin(["L", "R"])]
    # Statcast normally places 'events' on exactly the terminal pitch.
    # Guard against duplicate downloads and pick the last terminal pitch.
    keys = ["game_pk", "at_bat_number"]
    df = df.sort_values(["game_date", *keys, "pitch_number"]).drop_duplicates(keys, keep="last")
    df["batter"] = df["batter"].astype("int64")
    df["pitcher"] = df["pitcher"].astype("int64")
    df["hit"] = df["events"].isin(HITS).astype("int8")
    df["strikeout"] = df["events"].isin(STRIKEOUTS).astype("int8")
    df["walk"] = df["events"].isin(WALKS).astype("int8")
    df["home_run"] = df["events"].eq("home_run").astype("int8")
    fields = ["game_date", "game_pk", "at_bat_number", "batter", "pitcher", "stand", "p_throws", "hit", "strikeout", "walk", "home_run", "events"]
    return df[fields].sort_values(["game_date", "game_pk", "at_bat_number"]).reset_index(drop=True)

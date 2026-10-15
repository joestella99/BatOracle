# Bat Oracle 2.0 — Time-safe Matchup Analogues

This is a separate proof-of-concept, preserving the original Bat Oracle `src/` and `oldsrc/` unchanged.

## Thesis

Given batter A and pitcher A, find up to three **comparable** batters B/C/D (same batting hand, similar historical hit/K/BB/HR rates) and comparable pitchers B/C/D (same throwing hand, similar historical outcomes allowed). Then check **actual prior plate appearances** of the 4x4 comparable groups, weighting close matches more heavily. Blend the analog history with player rates and a learned classifier.

**Target:** a hit in one completed plate appearance, **not** the probability of >=1 hit in a full game. No LLM adjusts the numeric forecast.

## Quick start

```bash
python -m pip install -r requirements.txt
python -m bat_oracle.simulate --start 2023-03-30 --end 2024-05-31 --train-start 2024-04-01 --test-start 2024-05-01 --test-end 2024-05-31 --per-day 30
python -m pytest -q
```

Offline, if you already have complete Statcast **pitch-level** records:

```bash
python -m bat_oracle.simulate --input-csv path/to/statcast.csv --train-start 2024-04-01 --test-start 2024-05-01 --test-end 2024-05-31
```

Caching: Statcast is retrieved in 3-day chunks, saved in `data/raw/` as pickle files and reused. Downloads across full seasons take significant time and disk space. Don't commit caches or model output.

## Methodology and caveats

- A label is one PA outcome, selected from the pitch row with a non-null `events` field. No arbitrary pitch is labeled a hit.
- All profile and comparison data must come from **earlier calendar dates**. Same-day games are held out as a group, even when a game ends sooner.
- Similar hitters are matched to other **same-handed** hitters and pitchers to **same-handed** pitchers; comparators require 25 observed PAs. Their skill rates are smoothed by a league prior to avoid overconfident small samples.
- Analog matchups are weighted by hitter similarity * pitcher similarity; 25 league-prior pseudo-observations avoid extreme rates from tiny samples.
- The classifier is trained on rows from earlier dates and evaluated on later dates; no oversampling or random splitting. Test predictions may use outcomes of **previous** test dates, which mimics daily refreshes.
- Metrics include Brier, log loss, AUC, plus a running league hit rate and pure-comparable-matchup baseline. `per-day` deterministically samples the first N PAs each day **for development speed, not an unbiased production estimate**. For final results, increase `--per-day` substantially or disable the cap by code change and evaluate full test days.
- Profiles currently reflect outcome rates, not arsenal/velocity/swing plane. Pitch repertoire and weather are candidates for subsequent feature experiments, not claimed implemented features.
- `src/` 2018-24 summary files are NOT used; they're end-of-season figures and leak future information for historical forecasts.
- A substantial historic backtest needs downloaded Statcast data, not a small synthetic dataset. Tests use fixtures only to verify logic; they provide no model-accuracy claims.

## Future verification

- Compare a no-analogs model against one using analog rate and analog effective sample size (an ablation).
- Add per-pitch repertoire statistics using only prior pitches, and season-to-season validation.
- Consider a rolling 365-day sample rather than the entire past for changing player skill.
- The current analog-augmented model is a **baseline**; don't claim forecasting skill unless it beats simple time-safe baselines out of sample.

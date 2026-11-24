"""
EP Player Development Hub v2 — Bronze layer
Emerson Jimenez · Emerson Performance

Downloads pitch-level Statcast (regular season only) from Baseball Savant,
one day at a time, keeping only the columns the next-start model needs.
Each day is saved as an immutable Parquet file: data_raw/bronze/YYYY-MM-DD.parquet

Run:  python src/pull_statcast.py 2024 2025 2026   (2024 only feeds the prior-season features of 2025)
"""
import io
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

BRONZE = Path(__file__).resolve().parents[1] / "data_raw" / "bronze"
URL = ("https://baseballsavant.mlb.com/statcast_search/csv?all=true&type=details"
       "&hfGT=R%7C&player_type=pitcher&game_date_gt={d}&game_date_lt={d}")

COLS = [
    "game_pk", "game_date", "game_type", "home_team", "away_team", "inning", "inning_topbot",
    "at_bat_number", "pitch_number", "pitcher", "player_name", "p_throws", "age_pit",
    "pitch_type", "release_speed", "release_spin_rate", "release_extension", "pfx_x", "pfx_z",
    "zone", "description", "events", "estimated_woba_using_speedangle", "woba_value",
    "woba_denom", "n_thruorder_pitcher", "pitcher_days_since_prev_game",
]


def season_days(year):
    d, end = date(year, 3, 15), date(year, 10, 5)
    while d <= end:
        yield d
        d += timedelta(days=1)


def fetch(d):
    dest = BRONZE / f"{d.isoformat()}.parquet"
    if dest.exists():
        return d, "cached"
    for attempt in range(3):
        try:
            req = urllib.request.Request(URL.format(d=d.isoformat()),
                                         headers={"User-Agent": "EP-research"})
            with urllib.request.urlopen(req, timeout=120) as r:
                raw = r.read().decode("utf-8-sig")
            break
        except Exception as e:  # noqa: BLE001
            if attempt == 2:
                return d, f"error {e}"
            time.sleep(5 * (attempt + 1))
    if len(raw) < 200:  # no games that day
        pd.DataFrame(columns=COLS).to_parquet(dest, index=False)
        return d, "empty"
    df = pd.read_csv(io.StringIO(raw), low_memory=False)
    df = df[[c for c in COLS if c in df.columns]]
    df = df[df["game_type"] == "R"]
    df.to_parquet(dest, index=False)
    return d, len(df)


def main(years):
    BRONZE.mkdir(parents=True, exist_ok=True)
    days = [d for y in years for d in season_days(int(y))]
    with ThreadPoolExecutor(max_workers=4) as ex:
        for d, status in ex.map(fetch, days):
            print(d, status, flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or [2024, 2025, 2026])

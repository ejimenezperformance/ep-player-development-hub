"""
EP Player Development Hub — data build
Emerson Jimenez · Emerson Performance

Downloads Baseball Savant leaderboards (2024–2026), merges team/position,
and calibrates the alert thresholds from real year-to-year change (2024→2025).

Run:  python build_data.py            (downloads + builds)
      python build_data.py --offline  (rebuild from data_raw/ only)
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path(__file__).parent / "data_raw"
OUT = Path(__file__).parent / "data"
YEARS = [2024, 2025, 2026]
MIN_PA = 100  # hitters: PA, pitchers: batters faced

SAVANT = "https://baseballsavant.mlb.com/leaderboard"
BAT_COLS = ("player_age,pa,k_percent,bb_percent,woba,xwoba,exit_velocity_avg,launch_angle_avg,"
            "sweet_spot_percent,barrel_batted_rate,hard_hit_percent,avg_best_speed,avg_swing_speed,"
            "avg_swing_length,oz_swing_percent,whiff_percent,sprint_speed")
PIT_COLS = ("player_age,pa,p_formatted_ip,k_percent,bb_percent,woba,xwoba,barrel_batted_rate,"
            "hard_hit_percent,whiff_percent,oz_swing_percent,ff_avg_speed,ff_avg_spin,"
            "ff_avg_break_z_induced,fastball_avg_speed,fastball_avg_spin,arm_angle,p_game,p_starting_p")

URLS = {
    "bat": SAVANT + "/custom?year={y}&type=batter&filter=&min=1&selections=" + BAT_COLS + "&chart=false&csv=true",
    "pit": SAVANT + "/custom?year={y}&type=pitcher&filter=&min=1&selections=" + PIT_COLS + "&chart=false&csv=true",
    "spd": SAVANT + "/sprint_speed?year={y}&position=&team=&min=0&csv=true",
    "ars": SAVANT + "/pitch-arsenal-stats?type=pitcher&pitchType=&year={y}&team=&min=1&csv=true",
}

# Metrics shown in the app. direction: +1 higher is better, -1 lower is better, 0 descriptive only
HIT_METRICS = {
    "avg_swing_speed":    dict(dir=1,  dec=1, unit="mph"),
    "avg_best_speed":     dict(dir=1,  dec=1, unit="mph"),
    "exit_velocity_avg":  dict(dir=1,  dec=1, unit="mph"),
    "hard_hit_percent":   dict(dir=1,  dec=1, unit="%"),
    "barrel_batted_rate": dict(dir=1,  dec=1, unit="%"),
    "sweet_spot_percent": dict(dir=1,  dec=1, unit="%"),
    "oz_swing_percent":   dict(dir=-1, dec=1, unit="%"),
    "whiff_percent":      dict(dir=-1, dec=1, unit="%"),
    "k_percent":          dict(dir=-1, dec=1, unit="%"),
    "bb_percent":         dict(dir=1,  dec=1, unit="%"),
    "sprint_speed":       dict(dir=1,  dec=1, unit="ft/s"),
    "xwoba":              dict(dir=1,  dec=3, unit=""),
}
PIT_METRICS = {
    "fastball_avg_speed":     dict(dir=1,  dec=1, unit="mph"),
    "fastball_avg_spin":      dict(dir=1,  dec=0, unit="rpm"),
    "ff_avg_break_z_induced": dict(dir=1,  dec=1, unit="in"),
    "whiff_percent":          dict(dir=1,  dec=1, unit="%"),
    "oz_swing_percent":       dict(dir=1,  dec=1, unit="%"),
    "k_percent":              dict(dir=1,  dec=1, unit="%"),
    "bb_percent":             dict(dir=-1, dec=1, unit="%"),
    "hard_hit_percent":       dict(dir=-1, dec=1, unit="%"),
    "barrel_batted_rate":     dict(dir=-1, dec=1, unit="%"),
    "xwoba":                  dict(dir=-1, dec=3, unit=""),
    "arm_angle":              dict(dir=0,  dec=1, unit="°"),
}


def download():
    import urllib.request
    RAW.mkdir(exist_ok=True)
    for key, url in URLS.items():
        for y in YEARS:
            dest = RAW / f"{key}_{y}.csv"
            print("downloading", dest.name)
            with urllib.request.urlopen(url.format(y=y), timeout=90) as r:
                dest.write_bytes(r.read())
            time.sleep(1)


def read(key, y):
    d = pd.read_csv(RAW / f"{key}_{y}.csv", encoding="utf-8-sig")
    d = d.rename(columns={"last_name, first_name": "name_lf"})
    return d


def to_name(lf):
    if isinstance(lf, str) and ", " in lf:
        last, first = lf.split(", ", 1)
        return f"{first} {last}"
    return lf


def build_hitters():
    frames = []
    for y in YEARS:
        d = read("bat", y)
        d = d[d["pa"] >= MIN_PA].copy()
        s = read("spd", y)[["player_id", "team", "position"]]
        d = d.merge(s, on="player_id", how="left")
        d["year"] = y
        frames.append(d)
    h = pd.concat(frames, ignore_index=True)
    h["name"] = h["name_lf"].map(to_name)
    for c in HIT_METRICS:
        h[c] = pd.to_numeric(h[c], errors="coerce")
    h["team"] = h["team"].fillna("—")
    h["position"] = h["position"].fillna("—")
    return h


def build_pitchers():
    frames = []
    for y in YEARS:
        d = read("pit", y)
        d = d[d["pa"] >= MIN_PA].copy()
        a = read("ars", y)
        a["pitches"] = pd.to_numeric(a["pitches"], errors="coerce")
        team = (a.sort_values("pitches", ascending=False)
                 .groupby("player_id")["team_name_alt"].first().rename("team"))
        d = d.merge(team, on="player_id", how="left")
        d["year"] = y
        frames.append(d)
    p = pd.concat(frames, ignore_index=True)
    p["name"] = p["name_lf"].map(to_name)
    for c in PIT_METRICS:
        p[c] = pd.to_numeric(p[c], errors="coerce")
    p["role"] = np.where(p["p_starting_p"] / p["p_game"] >= 0.5, "SP", "RP")
    p["position"] = p["role"]
    p["team"] = p["team"].fillna("—")
    return p


def calibrate(df, metrics, group=None):
    """SD of the real year-to-year change (2024→2025) for each metric.
    Alerts use: |change| > 1 SD = red, > 0.5 SD = yellow (direction-aware)."""
    a = df[df.year == 2024].set_index("player_id")
    b = df[df.year == 2025].set_index("player_id")
    both = a.index.intersection(b.index)
    out = {}
    for m, spec in metrics.items():
        if spec["dir"] == 0:
            continue
        delta = (b.loc[both, m] - a.loc[both, m]).dropna()
        out[m] = dict(sd=float(delta.std()), mean=float(delta.mean()), n=int(len(delta)))
    return out


def main():
    if "--offline" not in sys.argv:
        download()
    OUT.mkdir(exist_ok=True)
    h = build_hitters()
    p = build_pitchers()
    h.to_csv(OUT / "hitters.csv", index=False)
    p.to_csv(OUT / "pitchers.csv", index=False)
    meta = dict(
        years=YEARS, min_pa=MIN_PA,
        built=time.strftime("%Y-%m-%d"),
        hit_metrics=HIT_METRICS, pit_metrics=PIT_METRICS,
        hit_thresholds=calibrate(h, HIT_METRICS),
        pit_thresholds=calibrate(p, PIT_METRICS),
    )
    (OUT / "meta.json").write_text(json.dumps(meta, indent=2))
    print("hitters:", h.groupby("year").size().to_dict())
    print("pitchers:", p.groupby("year").size().to_dict())
    for k in ("hit_thresholds", "pit_thresholds"):
        print(k)
        for m, v in meta[k].items():
            print(f"  {m:24s} sd={v['sd']:.3f} mean={v['mean']:+.3f} n={v['n']}")


if __name__ == "__main__":
    main()

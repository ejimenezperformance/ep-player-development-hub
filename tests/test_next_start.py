"""
Known-answer tests for the next-start evaluation, in BOTH directions.

  1. SIGNAL world: next-start CSW depends on the opponent, which no baseline sees.
     The pre-registered test must say CONFIRMED.
  2. NULL world:   next-start CSW = true talent + binomial noise, nothing else.
     A model cannot meaningfully beat the shrunk baseline -> NOT CONFIRMED.

Run:  python -m pytest tests/ -q      (or: python tests/test_next_start.py)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from next_start_model import evaluate, prepare  # noqa: E402


def simulate(opp_effect, seed=1, n_pitchers=180, n_starts=28):
    rng = np.random.default_rng(seed)
    rows = []
    for season in (2025, 2026):
        talent = rng.normal(0.28, 0.022, n_pitchers)
        prev = talent + rng.normal(0, 0.01, n_pitchers)
        velo = rng.normal(94, 2, n_pitchers)
        for i in range(n_pitchers):
            for s in range(n_starts):
                opp = rng.normal(0.28, 0.012)                  # opponent CSW-faced tendency
                p = np.clip(talent[i] + opp_effect * (opp - 0.28), 0.05, 0.6)
                n = int(rng.integers(70, 105))
                rows.append(dict(season=season, pitcher=season * 1000 + i, game_pk=s * 10_000 + i,
                                 game_date=pd.Timestamp(f"{season}-04-01") + pd.Timedelta(days=5 * s),
                                 pitches=n, csw_n=rng.binomial(n, p), opp_csw_std=opp,
                                 prev_csw=prev[i], prev_pitches=2500, fb_velo=velo[i] + rng.normal(0, 0.5),
                                 age=28, days_since_prev_game=5.0, absence_flag=0))
    df = pd.DataFrame(rows).sort_values(["season", "pitcher", "game_date"])
    g = df.groupby(["season", "pitcher"])
    df["csw_rate"] = df["csw_n"] / df["pitches"]
    cum_c = g["csw_n"].cumsum() - df["csw_n"]
    cum_n = g["pitches"].cumsum() - df["pitches"]
    df["std_csw"] = cum_c / cum_n.replace(0, np.nan)
    df["std_pitches"] = cum_n
    df["prior_starts"] = g.cumcount()
    df["last_csw"] = g["csw_rate"].shift(1)
    df["last_pitches"] = g["pitches"].shift(1)
    c3 = g["csw_n"].transform(lambda s: s.shift(1).rolling(3, min_periods=1).sum())
    n3 = g["pitches"].transform(lambda s: s.shift(1).rolling(3, min_periods=1).sum())
    df["last3_csw"] = c3 / n3
    df["last_fb_velo"] = g["fb_velo"].shift(1)
    df["std_fb_velo"] = g["fb_velo"].transform(lambda s: s.shift(1).expanding().mean())
    df["velo_trend"] = 0.0
    for c in ("std_whiff", "std_zone", "std_chase"):
        df[c] = df["std_csw"] + rng.normal(0, 0.01, len(df))
    df["league_csw_std"] = 0.28
    return df


def run(opp_effect):
    d = prepare(simulate(opp_effect))
    res, _, _, _ = evaluate(d[d.season == 2025], d[d.season == 2026])
    return res


def test_signal_world_is_confirmed():
    res = run(opp_effect=1.5)
    assert res["label"] == "CONFIRMED", res


def test_null_world_is_not_confirmed():
    res = run(opp_effect=0.0)
    assert res["label"] == "NOT CONFIRMED", res


if __name__ == "__main__":
    for eff in (1.5, 0.0):
        r = run(eff)
        print(f"opp_effect={eff}: {r['label']}  model={r['model']}  "
              f"MAE model={r['mae_test_model']:.4f}  best base ({r['best_baseline']})="
              f"{r['mae_test_baselines'][r['best_baseline']]:.4f}  gain={r['relative_gain']:.3f}  CI={r['mae_diff_ci95']}")

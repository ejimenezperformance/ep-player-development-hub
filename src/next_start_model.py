"""
EP Player Development Hub v2 — Next-start model
Emerson Jimenez · Emerson Performance

Decision question: what should a club expect from a starter's NEXT start
(CSW%), using only what is known before it — and does a model beat simple,
honest baselines?

Pre-registered rules (fixed before looking at 2026 results):
  * Universe: starts with 40+ pitches, 3+ previous starts that season,
    and no 14+ day absence before the start (return from IL/absence excluded).
  * Train = 2025 starts. Test = 2026 starts (one look, true time holdout).
  * Baselines: league average, last start, season-to-date, shrunk season-to-date
    (season-to-date blended with prior season and league; 2 weights tuned on 2025).
  * The best baseline AND the primary model are both chosen on 2025 only
    (model chosen with 5-fold TimeSeriesSplit inside 2025).
  * Metric: mean absolute error (MAE) of next-start CSW%, in percentage points.
  * CONFIRMED only if, on 2026: model MAE < best-baseline MAE, the 95% CI of the
    difference (cluster bootstrap by pitcher, 2,000 reps) is entirely below 0,
    AND the relative improvement is at least 2% (practical threshold).

Run:  python src/next_start_model.py
Out:  data/next_start_results.json, data/next_start_backtest.csv,
      data/next_start_projection.csv
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

MIN_PRIOR_STARTS = 3
PRACTICAL_GAIN = 0.02      # 2% relative MAE reduction
N_BOOT = 2000
SEED = 7

FEATURES = [
    "std_csw", "std_pitches", "last_csw", "last3_csw", "prev_csw", "prev_pitches",
    "std_whiff", "std_zone", "std_chase", "opp_csw_std", "league_csw_std",
    "velo_trend", "velo_dip_last", "days_since_prev_game", "last_pitches", "age", "prior_starts",
]
TARGET = "csw_rate"


# ───────────────────────── data ─────────────────────────
def prepare(df):
    d = df.copy()
    d["velo_dip_last"] = d["last_fb_velo"] - d["std_fb_velo"]
    d["prev_pitches"] = d["prev_pitches"].fillna(0)
    keep = (d["prior_starts"] >= MIN_PRIOR_STARTS) & (d["absence_flag"].fillna(0) == 0) \
        & d["days_since_prev_game"].notna()
    return d[keep].sort_values(["game_date", "game_pk"]).reset_index(drop=True)


# ───────────────────────── baselines ─────────────────────────
def shrunk(d, prior_w, k):
    """Season-to-date CSW blended with prior season (weight prior_w per pitch) and league (k pitches)."""
    prev = d["prev_csw"].fillna(d["league_csw_std"])
    prev_n = d["prev_pitches"].clip(upper=3000) * prior_w
    num = d["std_csw"] * d["std_pitches"] + prev * prev_n + d["league_csw_std"] * k
    return num / (d["std_pitches"] + prev_n + k)


def tune_shrunk(train):
    best = None
    for prior_w in [0, 0.1, 0.25, 0.5, 0.75, 1.0]:
        for k in [0, 100, 250, 500, 1000, 2000]:
            mae = np.mean(np.abs(train[TARGET] - shrunk(train, prior_w, k)))
            if best is None or mae < best[0]:
                best = (mae, prior_w, k)
    return {"prior_w": best[1], "k": best[2]}


def baseline_preds(d, params):
    return {
        "league_average": d["league_csw_std"],
        "last_start": d["last_csw"],
        "season_to_date": d["std_csw"],
        "shrunk_season_to_date": shrunk(d, params["prior_w"], params["k"]),
    }


# ───────────────────────── models ─────────────────────────
def make_models():
    return {
        "ridge": make_pipeline(SimpleImputer(strategy="median", add_indicator=True),
                               StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 20))),
        "gradient_boosting": HistGradientBoostingRegressor(
            max_depth=3, learning_rate=0.05, max_iter=300, min_samples_leaf=50,
            l2_regularization=1.0, random_state=SEED),
    }


def choose_model(train, features):
    X, y = train[features], train[TARGET]
    scores = {}
    for name, model in make_models().items():
        maes = []
        for tr, va in TimeSeriesSplit(n_splits=5).split(X):
            model.fit(X.iloc[tr], y.iloc[tr])
            maes.append(np.mean(np.abs(y.iloc[va] - model.predict(X.iloc[va]))))
        scores[name] = float(np.mean(maes))
    return min(scores, key=scores.get), scores


# ───────────────────────── stats ─────────────────────────
def cluster_bootstrap_diff(err_model, err_base, clusters, n_boot=N_BOOT, seed=SEED):
    """95% CI of mean(|err_model|) - mean(|err_base|), resampling pitchers."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"m": np.abs(err_model), "b": np.abs(err_base), "c": clusters})
    g = df.groupby("c").agg(m=("m", "sum"), b=("b", "sum"), n=("m", "size"))
    m, b, n = g["m"].to_numpy(), g["b"].to_numpy(), g["n"].to_numpy()
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, len(g), len(g))
        diffs[i] = m[idx].sum() / n[idx].sum() - b[idx].sum() / n[idx].sum()
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def evaluate(train, test, features=FEATURES):
    """Full pre-registered comparison. Returns a results dict and test predictions."""
    params = tune_shrunk(train)
    base_train = {k: np.mean(np.abs(train[TARGET] - v)) for k, v in baseline_preds(train, params).items()}
    best_base = min(base_train, key=base_train.get)

    model_name, cv_scores = choose_model(train, features)
    model = make_models()[model_name].fit(train[features], train[TARGET])

    resid_train = train[TARGET] - model.predict(train[features])
    lo_q, hi_q = np.quantile(resid_train, [0.10, 0.90])

    pred = model.predict(test[features])
    base_test = baseline_preds(test, params)
    y = test[TARGET].to_numpy()
    mae_model = float(np.mean(np.abs(y - pred)))
    mae_base = {k: float(np.mean(np.abs(y - v.to_numpy()))) for k, v in base_test.items()}
    ci = cluster_bootstrap_diff(y - pred, y - base_test[best_base].to_numpy(), test["pitcher"].to_numpy())
    gain = 1 - mae_model / mae_base[best_base]
    coverage = float(np.mean((y >= pred + lo_q) & (y <= pred + hi_q)))
    confirmed = bool(mae_model < mae_base[best_base] and ci[1] < 0 and gain >= PRACTICAL_GAIN)

    # permutation importance-lite: MAE increase when a feature is shuffled (test set)
    rng = np.random.default_rng(SEED)
    importance = {}
    for f in features:
        Xp = test[features].copy()
        Xp[f] = rng.permutation(Xp[f].to_numpy())
        importance[f] = float(np.mean(np.abs(y - model.predict(Xp))) - mae_model)

    results = {
        "n_train": int(len(train)), "n_test": int(len(test)),
        "n_test_pitchers": int(test["pitcher"].nunique()),
        "shrunk_params": params,
        "baseline_mae_train": {k: round(float(v), 5) for k, v in base_train.items()},
        "best_baseline": best_base,
        "model_cv_mae_train": {k: round(v, 5) for k, v in cv_scores.items()},
        "model": model_name,
        "mae_test_model": round(mae_model, 5),
        "mae_test_baselines": {k: round(v, 5) for k, v in mae_base.items()},
        "mae_diff_vs_best_baseline": round(mae_model - mae_base[best_base], 5),
        "mae_diff_ci95": [round(ci[0], 5), round(ci[1], 5)],
        "relative_gain": round(gain, 4),
        "practical_threshold": PRACTICAL_GAIN,
        "interval_80": [round(float(lo_q), 4), round(float(hi_q), 4)],
        "interval_80_coverage_test": round(coverage, 3),
        "permutation_importance_test": dict(sorted(
            ((k, round(v, 5)) for k, v in importance.items()), key=lambda kv: -kv[1])),
        "label": "CONFIRMED" if confirmed else "NOT CONFIRMED",
    }
    # production rule: if the model is not CONFIRMED, the app uses the simpler best baseline
    if confirmed:
        prod, plo, phi = pred, lo_q, hi_q
    else:
        resid_b = train[TARGET] - baseline_preds(train, params)[best_base]
        plo, phi = np.quantile(resid_b, [0.10, 0.90])
        prod = base_test[best_base].to_numpy()
    results["production"] = model_name if confirmed else best_base
    results["interval_80"] = [round(float(plo), 4), round(float(phi), 4)]
    results["interval_80_coverage_test"] = round(float(np.mean((y >= prod + plo) & (y <= prod + phi))), 3)
    results["model_interval_80_coverage_test"] = round(coverage, 3)
    out = test.copy()
    out["model_pred"] = pred
    out["pred"] = prod
    out["pred_lo"] = prod + plo
    out["pred_hi"] = prod + phi
    out["baseline"] = base_test[best_base].to_numpy()
    return results, out, model, params


def season_end_projection(gold, model, params, features, season):
    """Projection for each starter's next start using everything up to his last start of `season`.
    Built by treating the end of the season as a hypothetical next start."""
    s = gold[gold["season"] == season].sort_values(["game_date", "game_pk"])
    league = s["csw_n"].sum() / s["pitches"].sum()
    rows = []
    for pid, g in s.groupby("pitcher"):
        if len(g) < MIN_PRIOR_STARTS + 1:
            continue
        last = g.iloc[-1]
        allp, allc = g["pitches"].sum(), g["csw_n"].sum()
        tail3 = g.tail(3)
        r = {
            "pitcher": pid, "player_name": last["player_name"], "team": last["team"],
            "starts": len(g), "std_csw": allc / allp, "std_pitches": allp,
            "last_csw": last["csw_rate"], "last3_csw": tail3["csw_n"].sum() / tail3["pitches"].sum(),
            "prev_csw": last["prev_csw"], "prev_pitches": last["prev_pitches"] if pd.notna(last["prev_pitches"]) else 0,
            "std_whiff": g["whiff_n"].sum() / max(g["swing_n"].sum(), 1),
            "std_zone": g["zone_n"].sum() / allp,
            "std_chase": g["chase_n"].sum() / max(g["out_zone_n"].sum(), 1),
            "opp_csw_std": league,      # neutral (league-average) opponent
            "league_csw_std": league,
            "velo_trend": tail3["fb_velo"].mean() - (last["prev_fb_velo"] if pd.notna(last["prev_fb_velo"]) else g["fb_velo"].mean()),
            "velo_dip_last": last["fb_velo"] - g["fb_velo"].mean(),
            "days_since_prev_game": 5.0, "last_pitches": last["pitches"], "age": last["age"],
            "prior_starts": len(g), "fb_velo_season": g["fb_velo"].mean(),
            "last_start_date": str(last["game_date"])[:10],
        }
        rows.append(r)
    p = pd.DataFrame(rows)
    p["proj_csw"] = model.predict(p[features])
    p["shrunk_csw"] = shrunk(p, params["prior_w"], params["k"])
    return p


def main():
    gold = pd.read_parquet(DATA / "starts.parquet")
    d = prepare(gold)
    train, test = d[d["season"] == 2025], d[d["season"] == 2026]
    results, bt, model, params = evaluate(train, test)
    # secondary (descriptive) check: how much worse is reacting to the last start?
    y = bt[TARGET].to_numpy()
    base = baseline_preds(bt, params)
    for name in ("last_start", "league_average", "season_to_date"):
        lo, hi = cluster_bootstrap_diff(y - base[name].to_numpy(), y - base[results["best_baseline"]].to_numpy(),
                                        bt["pitcher"].to_numpy())
        results.setdefault("baseline_vs_best_ci95", {})[name] = [round(lo, 5), round(hi, 5)]
    results["universe_excluded_absence"] = int(((gold["absence_flag"] == 1) & (gold["season"] >= 2025)).sum())
    print(json.dumps(results, indent=2))
    (DATA / "next_start_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    for df_ in (bt, gold):
        df_["player_name"] = df_["player_name"].map(lambda n: " ".join(reversed(n.split(", "))) if ", " in n else n)
    cols = ["game_date", "game_pk", "pitcher", "player_name", "team", "opponent", "pitches",
            "csw_rate", "pred", "pred_lo", "pred_hi", "model_pred", "baseline", "whiff_rate", "fb_velo", "xwoba"]
    bt[cols].to_csv(DATA / "next_start_backtest.csv", index=False, float_format="%.4f")

    # refit on 2025+2026 for the season-end projection shown in the app
    final = make_models()[results["model"]].fit(d[FEATURES], d[TARGET])
    proj = season_end_projection(gold, final, params, FEATURES, 2026)
    proj["model_csw"] = proj["proj_csw"]
    if results["production"] != results["model"]:
        proj["proj_csw"] = proj["shrunk_csw"]
    proj.to_csv(DATA / "next_start_projection.csv", index=False, float_format="%.4f")
    print(f"projection rows: {len(proj)}")


if __name__ == "__main__":
    main()

"""
EP Player Development Hub v2 — Next-start report per starter (EN/ES)
Emerson Jimenez · Emerson Performance

Every number is computed first (next_start_model.py). This script only turns
those numbers into text. Two modes:

  * template (default): fixed bilingual wording filled with the numbers.
  * --llm: optional rewrite by Claude (needs ANTHROPIC_API_KEY). The LLM gets the
    computed numbers only and must not add any. A validator checks that every number
    in the LLM text appears in the input; if not, the template text is kept.

Reports are generated offline and cached in data/next_start_reports.json, so the
public app never calls an API and never needs a key.

Run:  python src/report.py            (template)
      python src/report.py --llm      (optional LLM rewrite with number check)
"""
import json
import os
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def pct(x, d=1):
    return f"{100 * x:.{d}f}%"


def facts(p, bt, res):
    """All numbers a report may use, already rounded as they will be printed."""
    g = bt[bt["pitcher"] == p["pitcher"]]
    lo, hi = res["interval_80"]
    out = dict(
        name=p["player_name"], team=p["team"], starts=int(p["starts"]),
        csw_season=pct(p["std_csw"]), csw_last3=pct(p["last3_csw"]),
        proj=pct(p["proj_csw"]), proj_lo=pct(p["proj_csw"] + lo), proj_hi=pct(p["proj_csw"] + hi),
        league=pct(p["league_csw_std"]),
        diff_pp=f"{100 * (p['proj_csw'] - p['league_csw_std']):+.1f}",
        velo_season=f"{p['fb_velo_season']:.1f}" if pd.notna(p["fb_velo_season"]) else None,
        velo_trend=f"{p['velo_trend']:+.1f}" if pd.notna(p["velo_trend"]) else None,
        n_bt=int(len(g)),
        n_out=int(((g["csw_rate"] < g["pred_lo"]) | (g["csw_rate"] > g["pred_hi"])).sum()),
        n_below=int((g["csw_rate"] < g["pred_lo"]).sum()),
        n_above=int((g["csw_rate"] > g["pred_hi"]).sum()),
    )
    return out


def template(f, res):
    model_ok = res["label"] == "CONFIRMED"
    tier_en = "above" if float(f["diff_pp"]) > 0.5 else "below" if float(f["diff_pp"]) < -0.5 else "around"
    tier_es = {"above": "por encima del", "below": "por debajo del", "around": "cerca del"}[tier_en]
    velo_en = velo_es = ""
    if f["velo_trend"] is not None:
        v = float(f["velo_trend"])
        if v <= -1.0:
            velo_en = (f" INTERPRETATION: his fastball over the last 3 starts was {f['velo_trend']} mph vs. his 2025 average (or his 2026 average if no 2025 data); "
                       "a drop of 1+ mph late in the season is worth checking for fatigue or physical limitation before the offseason plan.")
            velo_es = (f" INTERPRETACIÓN: su recta en las últimas 3 salidas estuvo {f['velo_trend']} mph frente a su promedio de 2025 (o de 2026 si no hay datos de 2025); "
                       "una caída de 1+ mph al final de la temporada merece revisar fatiga o limitación física antes del plan de offseason.")
    en = (f"CONFIRMED: {f['name']} ({f['team']}) posted a {f['csw_season']} CSW over {f['starts']} starts in 2026 "
          f"({f['csw_last3']} in his last 3). The next-start projection is {f['proj']} (80% range {f['proj_lo']}–{f['proj_hi']}), "
          f"{f['diff_pp']} pp vs. the league ({f['league']}), i.e. {tier_en} average. "
          f"In the 2026 backtest, {f['n_out']} of his {f['n_bt']} starts fell outside the 80% range "
          f"({f['n_below']} below, {f['n_above']} above); with an honest range about 2 in 10 should.")
    es = (f"CONFIRMADO: {f['name']} ({f['team']}) tuvo {f['csw_season']} de CSW en {f['starts']} salidas en 2026 "
          f"({f['csw_last3']} en las últimas 3). La proyección para su próxima salida es {f['proj']} (rango 80% {f['proj_lo']}–{f['proj_hi']}), "
          f"{f['diff_pp']} pp frente a la liga ({f['league']}), es decir {tier_es} promedio. "
          f"En el backtest de 2026, {f['n_out']} de sus {f['n_bt']} salidas quedaron fuera del rango 80% "
          f"({f['n_below']} por debajo, {f['n_above']} por encima); con un rango honesto, cerca de 2 de cada 10 deberían.")
    if not model_ok:
        en += (" The projection is his season CSW blended with 2025 and pulled toward the league average: across MLB, "
               "a 17-feature model did not beat this simple rule, so the app uses the simple rule.")
        es += (" La proyección es su CSW de la temporada combinado con 2025 y acercado al promedio de la liga: en toda la MLB, "
               "un modelo de 17 variables no le ganó a esta regla simple, así que la app usa la regla simple.")
    return {"en": en + velo_en, "es": es + velo_es}


NUM = re.compile(r"[+-]?\d+(?:\.\d+)?")


def numbers_ok(text, source):
    allowed = set(NUM.findall(source)) | {"2", "3", "10", "17", "80", "2025", "2026", "1"}
    return all(n.lstrip("+") in {a.lstrip("+") for a in allowed} for n in NUM.findall(text))


def llm_rewrite(text, lang):
    import anthropic  # optional dependency
    client = anthropic.Anthropic()
    prompt = (f"Rewrite this scouting note in clear, concise {'English' if lang == 'en' else 'Spanish'} for a "
              "front-office reader. Keep the CONFIRMED/INTERPRETATION labels. Do NOT add, change or compute any "
              f"number; use only numbers already in the text.\n\n{text}")
    msg = client.messages.create(model="claude-sonnet-4-5", max_tokens=400,
                                 messages=[{"role": "user", "content": prompt}])
    return msg.content[0].text.strip()


def main(use_llm=False):
    res = json.loads((DATA / "next_start_results.json").read_text(encoding="utf-8"))
    proj = pd.read_csv(DATA / "next_start_projection.csv")
    bt = pd.read_csv(DATA / "next_start_backtest.csv")
    reports, rejected = {}, 0
    for _, p in proj.iterrows():
        f = facts(p, bt, res)
        txt = template(f, res)
        if use_llm and os.environ.get("ANTHROPIC_API_KEY"):
            for lang in ("en", "es"):
                new = llm_rewrite(txt[lang], lang)
                if numbers_ok(new, txt[lang]):
                    txt[lang] = new
                else:
                    rejected += 1
        reports[str(int(p["pitcher"]))] = {"facts": f, **txt}
    (DATA / "next_start_reports.json").write_text(json.dumps(reports, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{len(reports)} reports written" + (f", {rejected} LLM texts rejected by the number check" if use_llm else ""))


if __name__ == "__main__":
    main("--llm" in sys.argv)

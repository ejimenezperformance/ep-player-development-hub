# EP Player Development Hub

**Emerson Jimenez · Emerson Performance**

> From Statcast data to a diagnosis and a 12-week offseason plan — for any MLB hitter or pitcher, in seconds.
> *Del dato de Statcast al diagnóstico y a un plan de offseason de 12 semanas, para cualquier bateador o pitcher de MLB, en segundos.*

**Live app:** _add Streamlit Cloud link here_

---

## What it does / Qué hace

| Tab | English | Español |
|---|---|---|
| **Profile / Perfil** | Percentile vs. MLB on 11–12 Statcast metrics (pitchers compared only within their role: SP or RP). Strengths and gaps. | Percentil vs. MLB en 11–12 métricas de Statcast (pitchers comparados solo con su rol: abridor o relevista). Fortalezas y áreas débiles. |
| **Trend & alerts / Tendencia y alertas** | 2024–2026 values with a traffic-light status calibrated on real league-wide year-to-year change. | Valores 2024–2026 con un semáforo calibrado con el cambio real de un año a otro en toda la liga. |
| **Offseason plan / Plan de offseason** | Rule-based 12-week plan: each block is triggered by a data finding and includes phases, monitoring metric and literature. | Plan de 12 semanas basado en reglas: cada bloque se activa por un hallazgo en los datos e incluye fases, métrica de monitoreo y literatura. |
| **Method / Método** | Data, thresholds, limits. | Datos, umbrales, límites. |

## Method / Método

The app follows the EP four-layer template: **Data → Finding → Interpretation → Training application → Monitoring metric.**

1. **Data (CONFIRMED).** Public Baseball Savant leaderboards, 2024–2026. Hitters ≥100 PA; pitchers ≥100 batters faced.
2. **Percentiles (CONFIRMED).** Rank vs. MLB that season; lower-is-better metrics are flipped so 100 = best.
3. **Alerts (CONFIRMED).** Thresholds are not arbitrary: they are the standard deviation of the real 2024→2025 change of every MLB player with both seasons (n ≈ 330–360). Adverse change > 1 SD = 🔴 Alert, > 0.5 SD = 🟡 Watch. Example: 1 SD of bat speed change = 1.05 mph; of fastball velo = 0.96 mph.
4. **Plan (INTERPRETATION).** A training block is triggered when one of its metrics is below the 35th percentile or carries an alert. Content is fixed, written from S&C literature, selected by rules — no LLM-generated text.

## Limits / Límites

- Statcast measures in-game output, not the cause. A velocity or bat-speed drop can come from fatigue, injury, mechanics or age; the plan is a starting point to confirm with an in-person evaluation (force plates, Blast, Trackman).
- MLB only; no minor-league or private biomechanics data.
- *Statcast mide el resultado en el juego, no la causa. El plan es un punto de partida que el staff debe confirmar con evaluación en persona.*

## Run it / Correrlo

```bash
pip install -r requirements.txt
python build_data.py      # downloads Savant leaderboards and rebuilds data/ (optional: data/ is included)
streamlit run app.py
```

Deploy: push this repo to GitHub → [share.streamlit.io](https://share.streamlit.io) → New app → `app.py`.

## Files

```
app.py            Streamlit app (EN/ES)
plans.py          Rule-based training modules (hitters & pitchers)
build_data.py     Savant download, merge, threshold calibration
data/             hitters.csv, pitchers.csv, meta.json
```

## References

Szymanski et al. 2007 (JSCR) · DeRenne & Szymanski 2009 (Strength Cond J) · MacWilliams et al. 1998 (AJSM) · Kageyama et al. 2014 (JSSM) · Petersen et al. 2011 (AJSM) · van der Horst et al. 2015 (AJSM) · Baseball Savant / MLB Statcast definitions.

---
© Emerson Jimenez · Emerson Performance. Data © MLB Advanced Media, used for non-commercial analysis.

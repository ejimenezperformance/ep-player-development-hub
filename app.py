"""
EP Player Development Hub
Emerson Jimenez · Emerson Performance

From Statcast data to a diagnosis and an offseason plan, for any MLB hitter or pitcher.
Run locally:  streamlit run app.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from plans import HIT_MODULES, PIT_MODULES

DATA = Path(__file__).parent / "data"
NAVY, GOLD, OFF, INK, MUTED = "#0B1B33", "#D4A53A", "#F5F3EC", "#1B2433", "#6B7280"
LOW, MID = "#B5483B", "#C9C6BD"  # diverging poles: warm (bad) ↔ neutral gray ↔ navy (good)

st.set_page_config(page_title="EP Player Development Hub", page_icon="⚾", layout="wide")

# ---------------------------------------------------------------- text
T = {
    "en": dict(
        title="Player Development Hub",
        sub="From Statcast data to a diagnosis and an offseason plan — any MLB hitter or pitcher.",
        type="Player type", hitter="Hitter", pitcher="Pitcher", team="Team", all="All teams",
        player="Player", season="Season", lang="Language",
        tab_profile="Profile", tab_trend="Trend & alerts", tab_plan="Offseason plan", tab_method="Method",
        pct_title="Percentile vs MLB", pct_note_h="Compared with all MLB hitters with ≥{n} PA in {y}.",
        pct_note_p="Compared with MLB {role} with ≥{n} batters faced in {y}.",
        strengths="Strengths (≥ 75th pct)", weak="Gaps (≤ 25th pct)", none="—",
        metric="Metric", value="Value", pct="Pct", change="Change", status="Status",
        red="Alert", yellow="Watch", green="Improved", stable="Stable",
        trend_note="Status compares {y1}→{y2}. Thresholds come from the real year-to-year change of all MLB players 2024→2025: "
                   "a move against the player larger than 1 SD = Alert, 0.5 SD = Watch.",
        no_prev="No previous season with enough playing time — trend not available.",
        pick_metric="Metric to plot",
        plan_head="12-week offseason plan",
        plan_note="Rule-based: each block is triggered by a CONFIRMED data finding (low percentile or alert). "
                  "The training content is an INTERPRETATION based on S&C literature, to be adjusted by the staff "
                  "after an in-person evaluation (force plates, Blast, Trackman).",
        trigger="Data trigger (CONFIRMED)", why="Interpretation", phases="Phases", monitor="Monitoring metric", ref="Reference",
        priority="Priority", no_plan="No clear gap: no metric below the 35th percentile and no alerts. Plan = maintain strengths and monitor.",
        credit="Built by Emerson Jimenez · Emerson Performance · Data: Baseball Savant (MLB Statcast)",
        role={"SP": "starter", "RP": "reliever"}, roles={"SP": "starters", "RP": "relievers"}, built="Data build", pa="PA", bf="BF", age="Age",
        tab_next="Next start (v2)", next_head="What to expect from his next start",
        next_proj="Projected CSW, next start", next_range="80% range", next_league="League",
        next_bt="2026 backtest: projection made before each start vs. what happened",
        next_actual="Actual CSW", next_pred="Projected (before the start)", next_band="80% range",
        next_report="Report", next_none="Next-start projection is available for starters with 4+ starts in 2026.",
        next_valid="League-wide validation (pre-registered, 2026 holdout)",
        next_model="Model", next_base="Best simple baseline", next_mae="Error (MAE, pp of CSW)",
        next_cov="80% range coverage in 2026", next_label="Result",
        next_note="Trained on 2025 only; every 2026 projection uses only information available before that start. "
                  "Starts after a 14+ day absence are excluded. Report text is generated from the computed numbers; it adds none.",
    ),
    "es": dict(
        title="Player Development Hub",
        sub="Del dato de Statcast al diagnóstico y al plan de offseason, para cualquier bateador o pitcher de MLB.",
        type="Tipo de jugador", hitter="Bateador", pitcher="Pitcher", team="Equipo", all="Todos los equipos",
        player="Jugador", season="Temporada", lang="Idioma",
        tab_profile="Perfil", tab_trend="Tendencia y alertas", tab_plan="Plan de offseason", tab_method="Método",
        pct_title="Percentil vs MLB", pct_note_h="Comparado con todos los bateadores de MLB con ≥{n} PA en {y}.",
        pct_note_p="Comparado con los {role} de MLB con ≥{n} bateadores enfrentados en {y}.",
        strengths="Fortalezas (≥ pct 75)", weak="Áreas débiles (≤ pct 25)", none="—",
        metric="Métrica", value="Valor", pct="Pct", change="Cambio", status="Estado",
        red="Alerta", yellow="Vigilar", green="Mejoró", stable="Estable",
        trend_note="El estado compara {y1}→{y2}. Los umbrales salen del cambio real de un año a otro de todos los jugadores de MLB 2024→2025: "
                   "un cambio en contra mayor a 1 DE = Alerta, 0.5 DE = Vigilar.",
        no_prev="No hay temporada anterior con suficiente tiempo de juego; no hay tendencia.",
        pick_metric="Métrica a graficar",
        plan_head="Plan de offseason de 12 semanas",
        plan_note="Basado en reglas: cada bloque se activa por un hallazgo CONFIRMADO en los datos (percentil bajo o alerta). "
                  "El contenido de entrenamiento es una INTERPRETACIÓN basada en literatura de S&C, para que el staff lo ajuste "
                  "después de una evaluación en persona (force plates, Blast, Trackman).",
        trigger="Disparador en los datos (CONFIRMADO)", why="Interpretación", phases="Fases", monitor="Métrica de monitoreo", ref="Referencia",
        priority="Prioridad", no_plan="Sin brecha clara: ninguna métrica bajo el percentil 35 y sin alertas. Plan = mantener fortalezas y monitorear.",
        credit="Creado por Emerson Jimenez · Emerson Performance · Datos: Baseball Savant (MLB Statcast)",
        role={"SP": "abridor", "RP": "relevista"}, roles={"SP": "abridores", "RP": "relevistas"}, built="Datos actualizados", pa="PA", bf="BE", age="Edad",
        tab_next="Próxima salida (v2)", next_head="Qué esperar de su próxima salida",
        next_proj="CSW proyectado, próxima salida", next_range="Rango 80%", next_league="Liga",
        next_bt="Backtest 2026: proyección hecha antes de cada salida vs. lo que pasó",
        next_actual="CSW real", next_pred="Proyectado (antes de la salida)", next_band="Rango 80%",
        next_report="Reporte", next_none="La proyección de próxima salida está disponible para abridores con 4+ salidas en 2026.",
        next_valid="Validación en toda la liga (pre-registrada, 2026 como prueba)",
        next_model="Modelo", next_base="Mejor línea base simple", next_mae="Error (MAE, pp de CSW)",
        next_cov="Cobertura del rango 80% en 2026", next_label="Resultado",
        next_note="Entrenado solo con 2025; cada proyección de 2026 usa solo información disponible antes de esa salida. "
                  "Se excluyen las salidas después de una ausencia de 14+ días. El texto del reporte se genera a partir de los números calculados; no agrega ninguno.",
    ),
}

LABELS = {
    "en": {
        "avg_swing_speed": "Bat speed", "avg_best_speed": "Best-speed EV (top 50%)", "exit_velocity_avg": "Avg exit velocity",
        "hard_hit_percent": "Hard-hit %", "barrel_batted_rate": "Barrel %", "sweet_spot_percent": "Sweet-spot %",
        "oz_swing_percent": "Chase %", "whiff_percent": "Whiff %", "k_percent": "K %", "bb_percent": "BB %",
        "sprint_speed": "Sprint speed", "xwoba": "xwOBA", "fastball_avg_speed": "Fastball velo",
        "fastball_avg_spin": "Fastball spin", "ff_avg_break_z_induced": "4-seam IVB", "arm_angle": "Arm angle",
    },
    "es": {
        "avg_swing_speed": "Bat speed", "avg_best_speed": "EV mejores contactos (top 50%)", "exit_velocity_avg": "Velocidad de salida prom.",
        "hard_hit_percent": "Hard-hit %", "barrel_batted_rate": "Barrel %", "sweet_spot_percent": "Sweet-spot %",
        "oz_swing_percent": "Chase %", "whiff_percent": "Whiff %", "k_percent": "K %", "bb_percent": "BB %",
        "sprint_speed": "Sprint speed", "xwoba": "xwOBA", "fastball_avg_speed": "Velocidad recta",
        "fastball_avg_spin": "Spin recta", "ff_avg_break_z_induced": "IVB recta 4 costuras", "arm_angle": "Ángulo de brazo",
    },
}

# ---------------------------------------------------------------- style
st.markdown(f"""
<style>
  .stApp {{ background:{OFF}; }}
  h1,h2,h3,h4 {{ color:{NAVY}; font-family:'Poppins',sans-serif; }}
  .ep-band {{ background:{NAVY}; color:{OFF}; padding:18px 24px; border-radius:10px; border-bottom:4px solid {GOLD}; margin-bottom:12px; }}
  .ep-band h1 {{ color:{OFF}; margin:0; font-size:1.7rem; }}
  .ep-band p {{ color:#D9D6CC; margin:4px 0 0 0; }}
  .ep-tag {{ color:{GOLD}; font-weight:700; letter-spacing:.08em; font-size:.75rem; }}
  .ep-card {{ background:white; border:1px solid #E4E0D5; border-left:4px solid {GOLD}; border-radius:8px; padding:14px 18px; margin-bottom:12px; }}
  .ep-muted {{ color:{MUTED}; font-size:.85rem; }}
  .ep-head {{ background:white; border:1px solid #E4E0D5; border-radius:8px; padding:12px 18px; margin-bottom:8px; }}
</style>""", unsafe_allow_html=True)


@st.cache_data
def load():
    h = pd.read_csv(DATA / "hitters.csv")
    p = pd.read_csv(DATA / "pitchers.csv")
    meta = json.loads((DATA / "meta.json").read_text())
    return h, p, meta


H, P, META = load()


@st.cache_data
def load_next():
    try:
        res = json.loads((DATA / "next_start_results.json").read_text(encoding="utf-8"))
        proj = pd.read_csv(DATA / "next_start_projection.csv")
        bt = pd.read_csv(DATA / "next_start_backtest.csv", parse_dates=["game_date"])
        rep = json.loads((DATA / "next_start_reports.json").read_text(encoding="utf-8"))
        return res, proj, bt, rep
    except FileNotFoundError:
        return None, None, None, None


NRES, NPROJ, NBT, NREP = load_next()

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    lang = "es" if st.radio("Language / Idioma", ["English", "Español"], horizontal=True) == "Español" else "en"
    t, L = T[lang], LABELS[lang]
    kind = st.radio(t["type"], [t["hitter"], t["pitcher"]], horizontal=True)
    is_hit = kind == t["hitter"]
    df = H if is_hit else P
    metrics = META["hit_metrics"] if is_hit else META["pit_metrics"]
    thresholds = META["hit_thresholds"] if is_hit else META["pit_thresholds"]
    latest = df.sort_values("year").groupby("player_id").tail(1)
    teams = sorted(latest["team"].dropna().unique())
    team = st.selectbox(t["team"], [t["all"]] + teams)
    pool = latest if team == t["all"] else latest[latest["team"] == team]
    pool = pool.sort_values("name")
    names = pool["name"] + " · " + pool["team"]
    default = 0
    for star in (["Juan Soto", "Aaron Judge"] if is_hit else ["Paul Skenes", "Tarik Skubal"]):
        hit = np.where(pool["name"].values == star)[0]
        if len(hit):
            default = int(hit[0])
            break
    choice = st.selectbox(t["player"], names.tolist(), index=default if len(pool) else None)
    pid = int(pool.iloc[names.tolist().index(choice)]["player_id"])
    seasons = sorted(df[df.player_id == pid]["year"].unique(), reverse=True)
    year = st.selectbox(t["season"], seasons)
    st.markdown(f"<div class='ep-muted'>{t['credit']}<br>{t['built']}: {META['built']}</div>", unsafe_allow_html=True)

row = df[(df.player_id == pid) & (df.year == year)].iloc[0]
role = row.get("role", None)

# population for percentiles: same season (and same role for pitchers)
popn = df[df.year == year]
if not is_hit:
    popn = popn[popn.role == role]


def percentile(metric, value, pop):
    vals = pop[metric].dropna()
    if pd.isna(value) or len(vals) < 20:
        return np.nan
    pct = (vals < value).mean() * 100 + (vals == value).mean() * 50
    return pct if metrics[metric]["dir"] >= 0 else 100 - pct


def fmt(metric, v):
    if pd.isna(v):
        return "—"
    d = metrics[metric]["dec"]
    s = f"{v:.{d}f}"
    if metric == "xwoba":
        s = s.lstrip("0")
    u = metrics[metric]["unit"]
    return s + ("" if u in ("", "%") else f" {u}") + ("%" if u == "%" else "")


pcts = {m: percentile(m, row[m], popn) for m, spec in metrics.items() if spec["dir"] != 0}

# trend vs previous season
prev = df[(df.player_id == pid) & (df.year == year - 1)]
prev = prev.iloc[0] if len(prev) else None
status = {}
for m, spec in metrics.items():
    if spec["dir"] == 0 or prev is None or pd.isna(row[m]) or pd.isna(prev[m]):
        continue
    z = (row[m] - prev[m]) / thresholds[m]["sd"] * spec["dir"]
    status[m] = ("red" if z <= -1 else "yellow" if z <= -0.5 else "green" if z >= 0.5 else "stable", row[m] - prev[m], z)

# ---------------------------------------------------------------- header
st.markdown(f"""<div class='ep-band'><span class='ep-tag'>EMERSON PERFORMANCE</span>
<h1>{t['title']}</h1><p>{t['sub']}</p></div>""", unsafe_allow_html=True)

pos = row["position"] if is_hit else t["role"][role]
vol = f"{int(row['pa'])} {t['pa'] if is_hit else t['bf']}"
st.markdown(f"""<div class='ep-head'><span style='font-size:1.5rem;font-weight:700;color:{NAVY}'>{row['name']}</span>
&nbsp;&nbsp;<span class='ep-muted'>{row['team']} · {pos} · {t['age']} {int(row['player_age'])} · {vol} · {year}</span></div>""",
            unsafe_allow_html=True)

show_next = (not is_hit) and NRES is not None
tabs = st.tabs([t["tab_profile"], t["tab_trend"], t["tab_plan"]] + ([t["tab_next"]] if show_next else []) + [t["tab_method"]])
tab1, tab2, tab3, tab4 = tabs[0], tabs[1], tabs[2], tabs[-1]
tab_next = tabs[3] if show_next else None


def pct_color(p):
    """Diverging: warm (low) ↔ neutral gray (50) ↔ navy (high)."""
    def mix(a, b, f):
        a, b = [int(a[i:i + 2], 16) for i in (1, 3, 5)], [int(b[i:i + 2], 16) for i in (1, 3, 5)]
        return "#" + "".join(f"{round(x + (y - x) * f):02X}" for x, y in zip(a, b))
    return mix(MID, LOW, (50 - p) / 50) if p < 50 else mix(MID, NAVY, (p - 50) / 50)


# ---------------------------------------------------------------- tab 1: profile
with tab1:
    c1, c2 = st.columns([3, 2])
    with c1:
        st.subheader(t["pct_title"])
        items = [(m, p) for m, p in pcts.items() if not pd.isna(p)]
        ys = [L[m] for m, _ in items][::-1]
        xs = [p for _, p in items][::-1]
        vals = [fmt(m, row[m]) for m, _ in items][::-1]
        fig = go.Figure(go.Bar(
            x=xs, y=ys, orientation="h", marker=dict(color=[pct_color(p) for p in xs], cornerradius=4),
            text=[f"{p:.0f}" for p in xs], textposition="outside", textfont=dict(color=INK, size=12),
            customdata=vals, hovertemplate="%{y}: %{customdata}<br>Pct %{x:.0f}<extra></extra>", width=0.62,
        ))
        fig.add_vline(x=50, line=dict(color=MUTED, width=1, dash="dot"))
        fig.update_layout(height=34 * len(ys) + 60, margin=dict(l=10, r=30, t=10, b=30), plot_bgcolor="white",
                          paper_bgcolor="rgba(0,0,0,0)", xaxis=dict(range=[0, 108], tickvals=[0, 25, 50, 75, 100],
                          gridcolor="#EEE", color=MUTED), yaxis=dict(color=INK), font=dict(family="Poppins, sans-serif"))
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
        note = t["pct_note_h"].format(n=META["min_pa"], y=year) if is_hit else \
            t["pct_note_p"].format(n=META["min_pa"], y=year, role=t["roles"][role])
        st.markdown(f"<div class='ep-muted'>{note}</div>", unsafe_allow_html=True)
    with c2:
        strong = [f"{L[m]} — {fmt(m, row[m])} (p{p:.0f})" for m, p in sorted(pcts.items(), key=lambda x: -np.nan_to_num(x[1])) if p >= 75]
        weak = [f"{L[m]} — {fmt(m, row[m])} (p{p:.0f})" for m, p in sorted(pcts.items(), key=lambda x: np.nan_to_num(x[1], nan=999)) if p <= 25]
        st.markdown(f"<div class='ep-card'><b>▲ {t['strengths']}</b><br>" + ("<br>".join(strong) or t["none"]) + "</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='ep-card' style='border-left-color:{LOW}'><b>▼ {t['weak']}</b><br>" + ("<br>".join(weak) or t["none"]) + "</div>", unsafe_allow_html=True)
        if not is_hit:
            st.markdown(f"<div class='ep-card'><b>{L['arm_angle']}</b>: {fmt('arm_angle', row['arm_angle'])}</div>", unsafe_allow_html=True)

# ---------------------------------------------------------------- tab 2: trend
ICON = {"red": "🔴", "yellow": "🟡", "green": "🟢", "stable": "⚪"}
with tab2:
    if prev is None:
        st.info(t["no_prev"])
    hist = df[df.player_id == pid].sort_values("year")
    rows = []
    for m, spec in metrics.items():
        if spec["dir"] == 0:
            continue
        r = {t["metric"]: L[m]}
        for _, hr in hist.iterrows():
            r[str(int(hr["year"]))] = fmt(m, hr[m])
        if m in status:
            s, d, _ = status[m]
            dec = spec["dec"]
            r[t["change"]] = f"{d:+.{dec}f}" if m != "xwoba" else f"{d:+.3f}"
            r[t["status"]] = f"{ICON[s]} {t[s]}"
        rows.append(r)
    tbl = pd.DataFrame(rows)
    order = {f"{ICON[k]} {t[k]}": i for i, k in enumerate(["red", "yellow", "green", "stable"])}
    if t["status"] in tbl:
        tbl = tbl.assign(_o=tbl[t["status"]].map(order).fillna(9)).sort_values("_o").drop(columns="_o")
    st.dataframe(tbl, hide_index=True, width="stretch")
    if prev is not None:
        st.markdown(f"<div class='ep-muted'>{t['trend_note'].format(y1=year - 1, y2=year)}</div>", unsafe_allow_html=True)

    opts = [m for m, s in metrics.items() if s["dir"] != 0]
    pick = st.selectbox(t["pick_metric"], opts, format_func=lambda m: L[m])
    lg = df.groupby("year")[pick].median() if is_hit else df[df.role == role].groupby("year")[pick].median()
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=lg.index, y=lg.values, name="MLB median", mode="lines+markers",
                              line=dict(color=MID, width=2, dash="dot"), marker=dict(size=8)))
    fig2.add_trace(go.Scatter(x=hist["year"], y=hist[pick], name=row["name"], mode="lines+markers+text",
                              line=dict(color=NAVY, width=2), marker=dict(size=10, color=GOLD, line=dict(color=NAVY, width=2)),
                              text=[fmt(pick, v) for v in hist[pick]], textposition="top center", textfont=dict(color=INK)))
    fig2.update_layout(height=320, margin=dict(l=10, r=10, t=30, b=30), plot_bgcolor="white", paper_bgcolor="rgba(0,0,0,0)",
                       xaxis=dict(tickvals=META["years"], color=MUTED), yaxis=dict(gridcolor="#EEE", color=MUTED),
                       legend=dict(orientation="h", y=1.12), font=dict(family="Poppins, sans-serif"), hovermode="x unified")
    st.plotly_chart(fig2, width="stretch", config={"displayModeBar": False})

# ---------------------------------------------------------------- tab 3: plan
with tab3:
    st.subheader(t["plan_head"])
    st.markdown(f"<div class='ep-muted'>{t['plan_note']}</div><br>", unsafe_allow_html=True)
    modules = HIT_MODULES if is_hit else PIT_MODULES
    chosen = []
    bs_flag = is_hit and (pcts.get("avg_swing_speed", 50) < 40 or
                          status.get("avg_swing_speed", ("stable",))[0] in ("red", "yellow"))
    for mod in modules:
        score, why_data = 0, []
        trig = list(mod["triggers"])
        if mod["key"] == "rot_power" and bs_flag:  # bat speed is the limiter: contact losses are evidence here
            trig += ["exit_velocity_avg", "hard_hit_percent", "barrel_batted_rate"]
        for m in trig:
            p = pcts.get(m, np.nan)
            if not pd.isna(p) and p < 35:
                score += 2 if p < 20 else 1
                why_data.append(f"{L[m]} {fmt(m, row[m])} → p{p:.0f}")
            if m in status and status[m][0] in ("red", "yellow"):
                s, d, _ = status[m]
                score += 2 if s == "red" else 1
                why_data.append(f"{L[m]} {d:+.{metrics[m]['dec'] if m != 'xwoba' else 3}f} {year - 1}→{year} ({ICON[s]} {t[s]})")
        # contact-quality block only when bat speed is not the limiter
        if mod["key"] == "contact_quality" and bs_flag:
            score = 0
        if score > 0:
            chosen.append((score, mod, why_data))
    chosen.sort(key=lambda x: -x[0])
    if not chosen:
        st.success(t["no_plan"])
    for i, (score, mod, why_data) in enumerate(chosen, 1):
        level = "●●●" if score >= 4 else "●●○" if score >= 2 else "●○○"
        phases = "".join(f"<li>{ph}</li>" for ph in mod["phases"][lang])
        st.markdown(f"""<div class='ep-card'>
<span class='ep-tag'>{t['priority']} {i} · {mod['area'][lang].upper()} · {level}</span>
<h4 style='margin:4px 0 8px 0'>{mod['title'][lang]}</h4>
<b>{t['trigger']}:</b> {' · '.join(why_data)}<br>
<b>{t['why']}:</b> {mod['why'][lang]}<br>
<b>{t['phases']}:</b><ul style='margin:4px 0'>{phases}</ul>
<b>{t['monitor']}:</b> {mod['monitor'][lang]}<br>
<span class='ep-muted'>{t['ref']}: {mod['ref']}</span></div>""", unsafe_allow_html=True)


# ---------------------------------------------------------------- tab next: next start (v2)
if tab_next is not None:
    with tab_next:
        st.subheader(t["next_head"])
        pr = NPROJ[NPROJ.pitcher == pid]
        if not len(pr):
            st.info(t["next_none"])
        else:
            pr = pr.iloc[0]
            lo, hi = NRES["interval_80"]
            c1, c2, c3 = st.columns(3)
            c1.metric(t["next_proj"], f"{100 * pr.proj_csw:.1f}%",
                      f"{100 * (pr.proj_csw - pr.league_csw_std):+.1f} pp vs {t['next_league'].lower()}")
            c2.metric(t["next_range"], f"{100 * (pr.proj_csw + lo):.1f}% – {100 * (pr.proj_csw + hi):.1f}%")
            c3.metric(t["next_league"], f"{100 * pr.league_csw_std:.1f}%")
            rep = NREP.get(str(pid))
            if rep:
                st.markdown(f"<div class='ep-card'><b>{t['next_report']}</b><br>{rep[lang]}</div>", unsafe_allow_html=True)

            g = NBT[NBT.pitcher == pid].sort_values("game_date")
            if len(g):
                st.markdown(f"**{t['next_bt']}**")
                out = (g.csw_rate < g.pred_lo) | (g.csw_rate > g.pred_hi)
                fig3 = go.Figure()
                fig3.add_trace(go.Scatter(x=list(g.game_date) + list(g.game_date[::-1]),
                                          y=list(100 * g.pred_hi) + list(100 * g.pred_lo[::-1]),
                                          fill="toself", fillcolor="rgba(11,27,51,0.10)", line=dict(width=0),
                                          name=t["next_band"], hoverinfo="skip"))
                fig3.add_trace(go.Scatter(x=g.game_date, y=100 * g.pred, name=t["next_pred"], mode="lines",
                                          line=dict(color=NAVY, width=2)))
                fig3.add_trace(go.Scatter(x=g.game_date, y=100 * g.csw_rate, name=t["next_actual"], mode="markers",
                                          marker=dict(size=[12 if o else 8 for o in out],
                                                      color=[LOW if o else GOLD for o in out],
                                                      symbol=["diamond" if o else "circle" for o in out],
                                                      line=dict(color=NAVY, width=1.5)),
                                          customdata=g.opponent, hovertemplate="%{x|%b %d} vs %{customdata}: %{y:.1f}%<extra></extra>"))
                fig3.update_layout(height=340, margin=dict(l=10, r=10, t=30, b=30), plot_bgcolor="white",
                                   paper_bgcolor="rgba(0,0,0,0)", yaxis=dict(title="CSW %", gridcolor="#EEE", color=MUTED),
                                   xaxis=dict(color=MUTED), legend=dict(orientation="h", y=1.14),
                                   font=dict(family="Poppins, sans-serif"))
                st.plotly_chart(fig3, width="stretch", config={"displayModeBar": False})

        st.markdown(f"**{t['next_valid']}**")
        MNAME = {'en': {'league_average': 'League average', 'last_start': 'Last start only', 'season_to_date': 'Season-to-date', 'shrunk_season_to_date': 'Shrunk season-to-date (used in app)'}, 'es': {'league_average': 'Promedio de la liga', 'last_start': 'Solo la última salida', 'season_to_date': 'Temporada hasta la fecha', 'shrunk_season_to_date': 'Temporada ajustada (usada en la app)'}}
        rows_v = [{t["metric"]: MNAME[lang][k], t["next_mae"]: round(100 * v, 2)}
                  for k, v in NRES["mae_test_baselines"].items()]
        rows_v.append({t["metric"]: f"{t['next_model']} ({NRES['model']}, 17 var.)", t["next_mae"]: round(100 * NRES["mae_test_model"], 2)})
        vt = pd.DataFrame(rows_v).sort_values(t["next_mae"])
        st.dataframe(vt, hide_index=True, width="stretch")
        ci = NRES["mae_diff_ci95"]
        st.markdown(f"<div class='ep-card'><b>{t['next_label']}: {NRES['label']}</b> · "
                    f"Δ MAE {100 * NRES['mae_diff_vs_best_baseline']:+.2f} pp (95% CI {100 * ci[0]:+.2f} to {100 * ci[1]:+.2f}) · "
                    f"{t['next_cov']}: {100 * NRES['interval_80_coverage_test']:.0f}% · n = {NRES['n_test']}</div>",
                    unsafe_allow_html=True)
        st.markdown(f"<div class='ep-muted'>{t['next_note']}</div>", unsafe_allow_html=True)

# ---------------------------------------------------------------- tab 4: method
with tab4:
    if lang == "es":
        st.markdown(f"""
**Datos.** Leaderboards públicos de Baseball Savant (Statcast) {META['years'][0]}–{META['years'][-1]}. Bateadores con ≥{META['min_pa']} PA y pitchers con ≥{META['min_pa']} bateadores enfrentados por temporada. Los pitchers se comparan solo con su mismo rol (abridor o relevista).

**Percentiles.** Posición del jugador frente a la MLB de esa temporada. En métricas donde menos es mejor (chase, whiff, K% del bateador; BB%, hard-hit, barrels y xwOBA permitidos del pitcher) el percentil se invierte, así que siempre 100 = mejor.

**Alertas.** Para no marcar ruido como señal, los umbrales salen del cambio real 2024→2025 de todos los jugadores con las dos temporadas. Un cambio en contra mayor a 1 desviación estándar = Alerta; mayor a 0.5 = Vigilar.

**Plan.** Basado en reglas, sin texto generado por IA. Un bloque se activa si alguna de sus métricas está bajo el percentil 35 o tiene alerta. El orden de prioridad suma la severidad de los disparadores.

**Límites.** Statcast mide el resultado en el juego, no la causa. Una caída de velocidad puede venir de fatiga, lesión, mecánica o edad: el plan es un punto de partida que debe confirmarse con evaluación en persona. No incluye ligas menores ni datos biomecánicos privados.
""")
    else:
        st.markdown(f"""
**Data.** Public Baseball Savant (Statcast) leaderboards, {META['years'][0]}–{META['years'][-1]}. Hitters with ≥{META['min_pa']} PA and pitchers with ≥{META['min_pa']} batters faced per season. Pitchers are compared only with their own role (starter or reliever).

**Percentiles.** Player rank vs. MLB that season. For lower-is-better metrics (hitter chase, whiff, K%; pitcher BB%, hard-hit, barrels and xwOBA allowed) the percentile is flipped, so 100 = best.

**Alerts.** To avoid flagging noise, thresholds come from the real 2024→2025 change of every player with both seasons. A move against the player larger than 1 standard deviation = Alert; larger than 0.5 = Watch.

**Plan.** Rule-based, no AI-generated text. A block is triggered when one of its metrics is below the 35th percentile or carries an alert. Priority order adds up trigger severity.

**Limits.** Statcast measures in-game output, not the cause. A velocity drop can come from fatigue, injury, mechanics or age: the plan is a starting point to confirm with an in-person evaluation. Minor leagues and private biomechanics data are not included.
""")
    th = pd.DataFrame([{t["metric"]: L[m], "1 SD": round(v["sd"], 3), "n": v["n"]} for m, v in thresholds.items()])
    st.dataframe(th, hide_index=True, width="stretch")

st.markdown(f"<hr><div class='ep-muted' style='text-align:center'>{t['credit']}</div>", unsafe_allow_html=True)

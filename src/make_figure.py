"""README figure: next-start CSW error by method (2026 holdout). EP brand, direct labels."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
NAVY, GOLD, OFF, INK, MUTED, WARM = "#0B1B33", "#D4A53A", "#F5F3EC", "#1B2433", "#6B7280", "#B5483B"

r = json.loads((ROOT / "data" / "next_start_results.json").read_text(encoding="utf-8"))
items = [
    ("Last start only / Solo la última salida", r["mae_test_baselines"]["last_start"], WARM),
    ("League average / Promedio de la liga", r["mae_test_baselines"]["league_average"], "#9A968C"),
    ("Season-to-date / Temporada a la fecha", r["mae_test_baselines"]["season_to_date"], "#9A968C"),
    ("17-variable model / Modelo de 17 variables", r["mae_test_model"], NAVY),
    ("Shrunk season-to-date / Temporada ajustada", r["mae_test_baselines"]["shrunk_season_to_date"], GOLD),
]
fig, ax = plt.subplots(figsize=(9, 4.6), dpi=160)
fig.patch.set_facecolor(OFF); ax.set_facecolor(OFF)
ys = range(len(items))[::-1]
for y, (lab, v, c) in zip(ys, items):
    ax.barh(y, 100 * v, color=c, height=0.6, edgecolor=NAVY if c == GOLD else "none", linewidth=1.2)
    ax.text(100 * v + 0.05, y, f"{100 * v:.2f} pp", va="center", color=INK, fontsize=11, fontweight="bold")
    ax.text(0.08, y, lab, va="center", color="white" if c in (NAVY, WARM) else INK, fontsize=10)
ax.set_yticks([]); ax.set_xlim(0, 6.2)
ax.set_xlabel("Average error predicting next-start CSW% (pp) · lower is better / Error promedio (pp) · menos es mejor",
              color=MUTED, fontsize=9)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color(MUTED); ax.tick_params(colors=MUTED)
fig.suptitle("Reacting to a starter's last start is worse than assuming he is league average",
             x=0.02, ha="left", color=NAVY, fontsize=13, fontweight="bold", y=0.98)
ax.set_title(f"2026 holdout · {r['n_test']:,} MLB starts · {r['n_test_pitchers']} starters · trained on 2025 only · "
             "Emerson Performance", loc="left", color=MUTED, fontsize=9)
fig.tight_layout()
fig.savefig(ROOT / "figures" / "next_start_error_by_method.png", facecolor=OFF)
print("saved")

"""
EP Player Development Hub v2 — Silver & Gold layers (DuckDB)
Emerson Jimenez · Emerson Performance

bronze (pitch-level Parquet, one file per day)
  -> silver: pitches with clean flags (CSW, whiff, swing, zone, fastball)
  -> gold:   one row per start (starter = first pitcher used by his team)
             + features built ONLY from information available before that start

Run:  python src/build_gold.py
Out:  data/starts.parquet   (one row per start, with features and the target)
"""
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[1]
BRONZE = ROOT / "data_raw" / "bronze" / "*.parquet"
OUT = ROOT / "data" / "starts.parquet"

MIN_PITCHES = 40       # a "start" needs 40+ pitches (drops openers and very early exits)
ABSENCE_DAYS = 14      # 14+ days since previous MLB game = return from IL/absence -> excluded as target

SQL = f"""
-- ───────────────────────── SILVER ─────────────────────────
CREATE OR REPLACE TABLE pitches AS
SELECT
    game_pk, CAST(game_date AS DATE) AS game_date, YEAR(CAST(game_date AS DATE)) AS season,
    pitcher, player_name, p_throws, age_pit,
    CASE WHEN inning_topbot = 'Top' THEN home_team ELSE away_team END AS pitching_team,
    CASE WHEN inning_topbot = 'Top' THEN away_team ELSE home_team END AS batting_team,
    at_bat_number, pitch_number, pitch_type, release_speed, release_extension,
    pitcher_days_since_prev_game,
    description,
    (description = 'called_strike')::INT AS called_strike,
    (description IN ('swinging_strike','swinging_strike_blocked','foul_tip'))::INT AS whiff,
    (description IN ('swinging_strike','swinging_strike_blocked','foul_tip','foul',
                     'hit_into_play','foul_bunt','missed_bunt','bunt_foul_tip'))::INT AS swing,
    (zone BETWEEN 1 AND 9)::INT AS in_zone,
    (pitch_type IN ('FF','SI'))::INT AS is_fastball,
    woba_denom,
    CASE WHEN woba_denom = 1 THEN COALESCE(estimated_woba_using_speedangle, woba_value) END AS xwoba_pa
FROM read_parquet('{BRONZE.as_posix()}', union_by_name = true)
WHERE game_type = 'R' AND pitcher IS NOT NULL;

-- starter = pitcher who threw the first pitch for his team in that game
CREATE OR REPLACE TABLE starters AS
SELECT game_pk, pitching_team, arg_min(pitcher, at_bat_number * 100 + pitch_number) AS pitcher
FROM pitches GROUP BY game_pk, pitching_team;

-- ───────────────────────── GOLD: one row per start ─────────────────────────
CREATE OR REPLACE TABLE starts AS
SELECT
    p.game_pk, p.game_date, p.season, p.pitcher,
    any_value(p.player_name) AS player_name,
    any_value(p.pitching_team) AS team,
    any_value(p.batting_team) AS opponent,
    max(p.age_pit) AS age,
    max(p.pitcher_days_since_prev_game) AS days_since_prev_game,
    count(*) AS pitches,
    sum(called_strike + whiff) AS csw_n,
    sum(whiff) AS whiff_n, sum(swing) AS swing_n,
    sum(in_zone) AS zone_n,
    sum(CASE WHEN in_zone = 0 THEN swing ELSE 0 END) AS chase_n,
    sum(1 - in_zone) AS out_zone_n,
    avg(CASE WHEN is_fastball = 1 THEN release_speed END) AS fb_velo,
    avg(CASE WHEN is_fastball = 1 THEN release_extension END) AS fb_ext,
    sum(woba_denom) AS pa_n,
    sum(xwoba_pa) AS xwoba_sum
FROM pitches p
JOIN starters s USING (game_pk, pitching_team, pitcher)
GROUP BY p.game_pk, p.game_date, p.season, p.pitcher
HAVING count(*) >= {MIN_PITCHES};

-- opponent strength known BEFORE the game: batting team's CSW faced, season-to-date
CREATE OR REPLACE TABLE team_day AS
SELECT season, batting_team AS team, game_date,
       sum(called_strike + whiff) AS csw_n, count(*) AS n
FROM pitches GROUP BY ALL;

CREATE OR REPLACE TABLE team_std AS
SELECT season, team, game_date,
       sum(csw_n) OVER w / NULLIF(sum(n) OVER w, 0) AS opp_csw_std,
       sum(n) OVER w AS opp_pitches_std
FROM team_day
WINDOW w AS (PARTITION BY season, team ORDER BY game_date
             ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING);

-- league CSW season-to-date (before the game date)
CREATE OR REPLACE TABLE league_day AS
SELECT season, game_date, sum(called_strike + whiff) AS csw_n, count(*) AS n
FROM pitches GROUP BY ALL;
CREATE OR REPLACE TABLE league_std AS
SELECT season, game_date,
       sum(csw_n) OVER w / NULLIF(sum(n) OVER w, 0) AS league_csw_std
FROM league_day
WINDOW w AS (PARTITION BY season ORDER BY game_date
             ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING);

-- prior full season, per pitcher (all appearances)
CREATE OR REPLACE TABLE prior_season AS
SELECT season + 1 AS season, pitcher,
       sum(called_strike + whiff) / count(*) AS prev_csw,
       count(*) AS prev_pitches,
       avg(CASE WHEN is_fastball = 1 THEN release_speed END) AS prev_fb_velo
FROM pitches GROUP BY ALL;

-- features from previous starts in the same season (strictly before the start)
CREATE OR REPLACE TABLE features AS
SELECT
    s.*,
    s.csw_n / s.pitches AS csw_rate,                         -- TARGET
    s.whiff_n / NULLIF(s.swing_n, 0) AS whiff_rate,
    s.xwoba_sum / NULLIF(s.pa_n, 0) AS xwoba,
    row_number() OVER w0 - 1 AS prior_starts,
    sum(s.csw_n) OVER w / NULLIF(sum(s.pitches) OVER w, 0) AS std_csw,
    sum(s.pitches) OVER w AS std_pitches,
    sum(s.whiff_n) OVER w / NULLIF(sum(s.swing_n) OVER w, 0) AS std_whiff,
    sum(s.zone_n) OVER w / NULLIF(sum(s.pitches) OVER w, 0) AS std_zone,
    sum(s.chase_n) OVER w / NULLIF(sum(s.out_zone_n) OVER w, 0) AS std_chase,
    sum(s.csw_n) OVER w3 / NULLIF(sum(s.pitches) OVER w3, 0) AS last3_csw,
    lag(s.csw_n / s.pitches) OVER w0 AS last_csw,
    lag(s.pitches) OVER w0 AS last_pitches,
    lag(s.fb_velo) OVER w0 AS last_fb_velo,
    avg(s.fb_velo) OVER w AS std_fb_velo,
    avg(s.fb_velo) OVER w3 AS last3_fb_velo
FROM starts s
WINDOW
    w0 AS (PARTITION BY s.season, s.pitcher ORDER BY s.game_date, s.game_pk),
    w  AS (PARTITION BY s.season, s.pitcher ORDER BY s.game_date, s.game_pk
           ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING),
    w3 AS (PARTITION BY s.season, s.pitcher ORDER BY s.game_date, s.game_pk
           ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING);

CREATE OR REPLACE TABLE gold AS
SELECT f.*,
       ps.prev_csw, ps.prev_pitches, ps.prev_fb_velo,
       t.opp_csw_std, l.league_csw_std,
       f.last3_fb_velo - COALESCE(ps.prev_fb_velo, f.std_fb_velo) AS velo_trend,
       (f.days_since_prev_game >= {ABSENCE_DAYS})::INT AS absence_flag
FROM features f
LEFT JOIN prior_season ps USING (season, pitcher)
LEFT JOIN team_std t ON t.season = f.season AND t.team = f.opponent AND t.game_date = f.game_date
LEFT JOIN league_std l ON l.season = f.season AND l.game_date = f.game_date
ORDER BY f.game_date, f.game_pk;
"""


def main():
    OUT.parent.mkdir(exist_ok=True)
    con = duckdb.connect()
    con.execute(SQL)
    con.execute(f"COPY gold TO '{OUT.as_posix()}' (FORMAT PARQUET)")
    n = con.execute("SELECT season, count(*), count(DISTINCT pitcher) FROM gold GROUP BY 1 ORDER BY 1").fetchall()
    for season, starts, pitchers in n:
        print(f"{season}: {starts} starts, {pitchers} starters")


if __name__ == "__main__":
    main()

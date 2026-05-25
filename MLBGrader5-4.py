# @title ⚾ MLB Daily Picks Grader — Run Morning After Games — 5-4 Baseline
import pandas as pd
import numpy as np
import requests
import time
import math
import re
import unicodedata
import os, json
import atexit
from datetime import datetime, timedelta
from itertools import combinations
import pytz
import gspread
from google.auth import default
from google.oauth2.service_account import Credentials
from run_logger import RunLogger

def get_gspread_client():
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]
    svc_json = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON") or os.environ.get("GSPREAD_SERVICE_ACCOUNT_JSON")
    if svc_json:
        creds = Credentials.from_service_account_info(json.loads(svc_json), scopes=scopes)
        print("✅ Google auth via service account env")
        return gspread.authorize(creds)
    try:
        from google.colab import auth as colab_auth
        print("Authenticating with Google...")
        colab_auth.authenticate_user()
        creds, _ = default(scopes=scopes)
        print("✅ Google auth via Colab")
        return gspread.authorize(creds)
    except Exception as e:
        raise RuntimeError("Google auth unavailable. Set GOOGLE_SERVICE_ACCOUNT_JSON or run in Colab.") from e

gc = get_gspread_client()

SHEET_NAME = 'MLB_Dashboard_Data'
SHEET_ID = '1AAwSwFCGIqS6JGdYTdkSau91BtnM_sMdWl2By5A9nFQ'
MLB_API = "https://statsapi.mlb.com/api/v1"
SNAPSHOT_DATE = "2026-05-04"
sh = gc.open_by_key(SHEET_ID)
print(f"✅ Connected to Google Sheet: {SHEET_ID}")
runlog = RunLogger(gc, SHEET_ID, sport='MLB', kind='grader')
atexit.register(runlog.finalize_and_write)

eastern = pytz.timezone('US/Eastern')
now_est = datetime.now(eastern)
today_str = now_est.strftime('%Y-%m-%d')
timestamp_est = now_est.strftime('%Y-%m-%d %I:%M:%S %p EST')
RETRY_DNP_LOOKBACK_DAYS = 7

def safe_float(val, default=None):
    if val is None:
        return default
    if isinstance(val, str):
        val = val.strip().replace(',', '')
        if not val or val.upper() in {'N/A', 'NA', 'NONE', 'NULL', 'DNP'}:
            return default
    try:
        num = float(val)
        if math.isnan(num) or math.isinf(num):
            return default
        return num
    except (TypeError, ValueError):
        return default

def innings_to_outs(ip_val):
    if ip_val is None:
        return 0
    s = str(ip_val).strip()
    if not s or s.upper() in {'N/A', 'NA', 'NONE', 'NULL', 'DNP'}:
        return 0
    try:
        whole_str, frac_str = s.split('.', 1)
        whole = int(whole_str or 0)
        frac = int(frac_str[:1] or 0)
        if len(frac_str) > 1 or frac > 2:
            num = safe_float(s, 0)
            return int(round((num or 0) * 3))
        frac = max(0, min(frac, 2))
        return whole * 3 + frac
    except ValueError:
        num = safe_float(s, 0)
        return int(round((num or 0) * 3))

def outs_to_ip_decimal(outs):
    outs = safe_float(outs, 0)
    if outs is None:
        return 0
    return round((outs or 0) / 3, 3)

def normalize_person_name(name):
    text = unicodedata.normalize('NFKD', str(name or ''))
    text = ''.join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower()
    text = re.sub(r"[’'`\\.]", "", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def find_box_score(box_lookup, player, date):
    box = box_lookup.get((player, date))
    if box is not None:
        return box
    player_norm = normalize_person_name(player)
    for (bn, bd), bv in box_lookup.items():
        if bd == date and normalize_person_name(bn) == player_norm:
            return bv
    return None

def grade_pick(actual, line_val, lean):
    if actual is None or line_val is None:
        return '', ''
    if actual == line_val:
        return 'PUSH', 'PUSH'
    if lean in ('UNDER', 'FADE'):
        return ('YES', 'HIT') if actual < line_val else ('NO', 'MISS')
    return ('YES', 'HIT') if actual > line_val else ('NO', 'MISS')

def combo_leg_label(row):
    return f"{row.get('player', '?')} {row.get('prop_type', '?')} {row.get('lean', '?')} {row.get('line', '?')}"

def print_winning_combo_tracker(df_all, dates_to_grade):
    if 'DATE' not in df_all.columns or 'HIT' not in df_all.columns:
        return
    hit_df = df_all[df_all['HIT'] == 'YES'].copy()
    if hit_df.empty:
        return
    hit_df['DATE'] = hit_df['DATE'].astype(str)
    target_dates = {str(d) for d in dates_to_grade}
    hit_df = hit_df[hit_df['DATE'].isin(target_dates)]
    if hit_df.empty:
        return
    hit_df['_run'] = pd.to_numeric(hit_df['RUN_NUMBER'], errors='coerce') if 'RUN_NUMBER' in hit_df.columns else np.nan
    print("\n   Winning Combo Tracker:")
    for date in sorted(hit_df['DATE'].unique()):
        date_df = hit_df[hit_df['DATE'] == date]
        run_vals = sorted(date_df['_run'].dropna().astype(int).unique()) if pd.Series(date_df['_run']).notna().any() else [None]
        for run_no in run_vals:
            grp = date_df if run_no is None else date_df[date_df['_run'] == run_no]
            labels = [combo_leg_label(row) for _, row in grp.iterrows()]
            if len(labels) < 2:
                continue
            combos2 = list(combinations(labels, 2))
            combos3 = list(combinations(labels, 3)) if len(labels) >= 3 else []
            header = f"   {date}" + (f" / Run {run_no}" if run_no is not None else "")
            print(f"{header}: {len(combos2)} winning 2-leg, {len(combos3)} winning 3-leg")
            if combos2:
                print(f"      2-leg ex: {' + '.join(combos2[0])}")
            if combos3:
                print(f"      3-leg ex: {' + '.join(combos3[0])}")

def print_clv_summary(df_all):
    needed = {'CLV_OPEN_LINE', 'CLV_LATEST_LINE', 'lean', 'HIT'}
    if not needed.issubset(df_all.columns):
        return
    clv_df = df_all[df_all['HIT'].isin(['YES', 'NO'])].copy()
    if clv_df.empty:
        return
    clv_df['open_line'] = pd.to_numeric(clv_df['CLV_OPEN_LINE'], errors='coerce')
    clv_df['latest_line'] = pd.to_numeric(clv_df['CLV_LATEST_LINE'], errors='coerce')
    clv_df = clv_df.dropna(subset=['open_line', 'latest_line'])
    if clv_df.empty:
        return
    clv_df['lean_norm'] = clv_df['lean'].fillna('').astype(str).str.upper().replace({'FADE': 'UNDER'})
    clv_df['clv_edge'] = np.where(clv_df['lean_norm'] == 'UNDER', clv_df['open_line'] - clv_df['latest_line'], clv_df['latest_line'] - clv_df['open_line'])
    print("\n   CLV Summary:")
    pos_df = clv_df[clv_df['clv_edge'] > 0]
    neg_df = clv_df[clv_df['clv_edge'] <= 0]
    if not pos_df.empty:
        pos_hits = len(pos_df[pos_df['HIT'] == 'YES'])
        print(f"   Positive CLV: {pos_hits}-{len(pos_df)-pos_hits} ({pos_hits/len(pos_df)*100:.0f}%) | Avg {pos_df['clv_edge'].mean():+.2f}")
    if not neg_df.empty:
        neg_hits = len(neg_df[neg_df['HIT'] == 'YES'])
        print(f"   Flat/Negative CLV: {neg_hits}-{len(neg_df)-neg_hits} ({neg_hits/len(neg_df)*100:.0f}%) | Avg {neg_df['clv_edge'].mean():+.2f}")

def write_dataframe_sheet(spreadsheet, sheet_name, df):
    if df is None or df.empty:
        print(f"   ⏭️  {sheet_name}: No data — skipped")
        return
    df_clean = df.copy().replace([np.inf, -np.inf], '')
    df_clean = df_clean.fillna('')
    for col in df_clean.columns:
        df_clean[col] = df_clean[col].apply(lambda x: x.item() if hasattr(x, 'item') else x)
    try:
        try:
            ws_out = spreadsheet.worksheet(sheet_name)
            ws_out.clear()
        except gspread.exceptions.WorksheetNotFound:
            ws_out = spreadsheet.add_worksheet(title=sheet_name, rows=max(len(df_clean) + 1, 100), cols=max(len(df_clean.columns), 26))
        needed_rows = len(df_clean) + 1
        needed_cols = len(df_clean.columns)
        if ws_out.row_count < needed_rows or ws_out.col_count < needed_cols:
            ws_out.resize(rows=max(needed_rows, ws_out.row_count), cols=max(needed_cols, ws_out.col_count))
        ws_out.update([df_clean.columns.tolist()] + df_clean.values.tolist(), value_input_option='RAW')
        print(f"   ✅ {sheet_name}: {len(df_clean)} rows × {len(df_clean.columns)} cols")
    except Exception as e:
        print(f"   ❌ {sheet_name}: {e}")

def score_bucket(val):
    if pd.isna(val):
        return None
    if val < 40:
        return '00-39'
    if val < 60:
        return '40-59'
    if val < 80:
        return '60-79'
    return '80-100'

def build_score_backtest(df_graded):
    score_cols = ['H_EDGE_SCORE', 'POWER_EDGE_SCORE', 'P_SO_EDGE_SCORE', 'P_ER_RISK_SCORE']
    if df_graded is None or df_graded.empty:
        return pd.DataFrame(columns=['SCORE', 'BUCKET', 'PROP_TYPE', 'CONFIDENCE', 'PICKS', 'WINS', 'HIT_RATE', 'MEAN_SCORE', 'LAST_UPDATED'])
    df_graded = df_graded[df_graded['HIT'].isin(['YES', 'NO'])].copy()
    if df_graded.empty:
        return pd.DataFrame(columns=['SCORE', 'BUCKET', 'PROP_TYPE', 'CONFIDENCE', 'PICKS', 'WINS', 'HIT_RATE', 'MEAN_SCORE', 'LAST_UPDATED'])
    for col in score_cols:
        if col not in df_graded.columns:
            df_graded[col] = np.nan
        df_graded[col] = pd.to_numeric(df_graded[col], errors='coerce')
    df_graded['WIN'] = (df_graded['HIT'] == 'YES').astype(int)

    rows = []
    for score_col in score_cols:
        df_score = df_graded.dropna(subset=[score_col]).copy()
        if df_score.empty:
            continue
        df_score['BUCKET'] = df_score[score_col].map(score_bucket)
        for bucket, grp in df_score.groupby('BUCKET'):
            rows.append({
                'SCORE': score_col,
                'BUCKET': bucket,
                'PROP_TYPE': 'ALL',
                'CONFIDENCE': 'ALL',
                'PICKS': len(grp),
                'WINS': int(grp['WIN'].sum()),
                'HIT_RATE': round(grp['WIN'].mean(), 3),
                'MEAN_SCORE': round(grp[score_col].mean(), 1),
            })
        if 'prop_type' in df_score.columns:
            for (bucket, prop), grp in df_score.groupby(['BUCKET', 'prop_type']):
                if len(grp) < 3:
                    continue
                rows.append({
                    'SCORE': score_col,
                    'BUCKET': bucket,
                    'PROP_TYPE': prop,
                    'CONFIDENCE': 'ALL',
                    'PICKS': len(grp),
                    'WINS': int(grp['WIN'].sum()),
                    'HIT_RATE': round(grp['WIN'].mean(), 3),
                    'MEAN_SCORE': round(grp[score_col].mean(), 1),
                })
        if 'confidence' in df_score.columns:
            for (bucket, conf), grp in df_score.groupby(['BUCKET', 'confidence']):
                if len(grp) < 3:
                    continue
                rows.append({
                    'SCORE': score_col,
                    'BUCKET': bucket,
                    'PROP_TYPE': 'ALL',
                    'CONFIDENCE': conf,
                    'PICKS': len(grp),
                    'WINS': int(grp['WIN'].sum()),
                    'HIT_RATE': round(grp['WIN'].mean(), 3),
                    'MEAN_SCORE': round(grp[score_col].mean(), 1),
                })
    df_backtest = pd.DataFrame(rows)
    if df_backtest.empty:
        return pd.DataFrame(columns=['SCORE', 'BUCKET', 'PROP_TYPE', 'CONFIDENCE', 'PICKS', 'WINS', 'HIT_RATE', 'MEAN_SCORE', 'LAST_UPDATED'])
    df_backtest['LAST_UPDATED'] = timestamp_est
    return df_backtest.sort_values(['SCORE', 'BUCKET', 'PROP_TYPE', 'CONFIDENCE']).reset_index(drop=True)

# --- 2. LOAD DAILY_PICKS ---
print("\nLoading Daily_Picks...")
try:
    ws = sh.worksheet('Daily_Picks')
    all_rows = ws.get_all_values()
except Exception as e:
    print(f"❌ Could not find Daily_Picks sheet: {e}")
    raise

if len(all_rows) <= 1:
    print("⚠️ No picks to grade — sheet is empty.")
    headers = ['DATE', 'HIT']
    df_picks = pd.DataFrame(columns=headers)
else:
    headers = all_rows[0]
    data = all_rows[1:]
    df_picks = pd.DataFrame(data, columns=headers)
    print(f"📋 Found {len(df_picks)} total picks across {df_picks['DATE'].nunique()} dates")

# --- 3. FIND UNGRADED PICKS ---
hit_series = df_picks['HIT'].fillna('').astype(str).str.strip()
date_series = pd.to_datetime(df_picks['DATE'], errors='coerce')
today_ts = pd.to_datetime(today_str)
retry_cutoff = today_ts - pd.Timedelta(days=RETRY_DNP_LOOKBACK_DAYS)
retry_dnp_mask = (hit_series == 'DNP') & date_series.notna() & (date_series >= retry_cutoff) & (date_series <= today_ts)
blank_ungraded_mask = (hit_series == '') & date_series.notna() & (date_series < today_ts)
ungraded = df_picks[blank_ungraded_mask | retry_dnp_mask].copy()

if ungraded.empty:
    blanks_today = int(((hit_series == '') & date_series.notna() & (date_series >= today_ts)).sum())
    if blanks_today > 0:
        print(f"⏳ {blanks_today} ungraded picks from today ({today_str}) — games haven't finished yet. Run tomorrow.")
    else:
        print("✅ All picks are already graded! Nothing to do.")
    dates_to_grade = []
else:
    dates_to_grade = sorted(ungraded['DATE'].unique())
    retry_ct = int(retry_dnp_mask.sum())
    if retry_ct > 0:
        print(f"🎯 {len(ungraded)} gradeable picks from: {', '.join(dates_to_grade)} ({retry_ct} recent DNP retries)")
    else:
        print(f"🎯 {len(ungraded)} gradeable picks from: {', '.join(dates_to_grade)}")

# --- 4. FETCH MLB BOX SCORES ---
# We need actual game stats for each player on each date.
# Strategy: use the MLB Stats API gameLog endpoint for each player we need to grade.
# More efficient: fetch from our own Batter_Game_Logs sheet (already has the data).

print("\nFetching box score data...")

# First try: load from our own sheets (fastest, already computed)
box_lookup = {}
sheets_loaded = False

try:
    print("   📊 Loading from Batter_Game_Logs sheet...")
    ws_logs = sh.worksheet('Batter_Game_Logs')
    log_rows = ws_logs.get_all_records()
    df_bat_logs = pd.DataFrame(log_rows)

    if len(df_bat_logs) > 0:
        # Build lookup: (player_name, game_date) → stats dict
        for _, row in df_bat_logs.iterrows():
            name = str(row.get('player_name', ''))
            date = str(row.get('game_date', ''))
            if not name or not date:
                continue
            key = (name, date)
            box_lookup[key] = {
                'H': safe_float(row.get('H'), 0),
                'HR': safe_float(row.get('HR'), 0),
                'RBI': safe_float(row.get('RBI'), 0),
                'R': safe_float(row.get('R'), 0),
                'SB': safe_float(row.get('SB'), 0),
                'SO': safe_float(row.get('SO'), 0),
                'BB': safe_float(row.get('BB'), 0),
                'TB': safe_float(row.get('TB'), 0),
                'AB': safe_float(row.get('AB'), 0),
                '1B': safe_float(row.get('1B'), 0),
                '2B': safe_float(row.get('2B'), 0),
                '3B': safe_float(row.get('3B'), 0),
                'HBP': safe_float(row.get('HBP'), 0),
                'DK_FP': safe_float(row.get('DK_FP'), 0),
                'UD_FP': safe_float(row.get('UD_FP'), 0),
            }
        print(f"   ✅ Loaded {len(box_lookup)} batter game entries")
        sheets_loaded = True
except Exception as e:
    print(f"   ⚠️ Could not load Batter_Game_Logs: {e}")

# Also load pitcher logs for pitcher props
try:
    print("   📊 Loading from Pitcher_Game_Logs sheet...")
    ws_plogs = sh.worksheet('Pitcher_Game_Logs')
    plog_rows = ws_plogs.get_all_records()
    df_pit_logs = pd.DataFrame(plog_rows)

    if len(df_pit_logs) > 0:
        pitcher_count = 0
        for _, row in df_pit_logs.iterrows():
            name = str(row.get('player_name', ''))
            date = str(row.get('game_date', ''))
            if not name or not date:
                continue
            key = (name, date)
            if key not in box_lookup:  # Don't overwrite batter data
                box_lookup[key] = {}
            # Add pitcher-specific stats (prefix with P_ to match prop naming)
            pitcher_outs = innings_to_outs(row.get('IP'))
            box_lookup[key].update({
                'P_SO': safe_float(row.get('SO'), 0),
                'P_H': safe_float(row.get('H'), 0),
                'P_BB': safe_float(row.get('BB'), 0),
                'P_ER': safe_float(row.get('ER'), 0),
                'IP': outs_to_ip_decimal(pitcher_outs),
                'P_OUTS': pitcher_outs,
                'W': safe_float(row.get('W'), 0),
                'SO': safe_float(row.get('SO'), 0),  # Also store without prefix
                'ER': safe_float(row.get('ER'), 0),
                'DK_FP': safe_float(row.get('DK_FP'), 0),
                'UD_FP': safe_float(row.get('UD_FP'), 0),
            })
            pitcher_count += 1
        print(f"   ✅ Loaded {pitcher_count} pitcher game entries")
except Exception as e:
    print(f"   ⚠️ Could not load Pitcher_Game_Logs: {e}")

box_dates = sorted(set(d for _, d in box_lookup.keys()))
box_date_set = set(box_dates)
grade_dates_available = [d for d in dates_to_grade if d in box_dates]
grade_dates_missing = [d for d in dates_to_grade if d not in box_dates]

if grade_dates_available:
    print(f"   ✅ Data available for: {', '.join(grade_dates_available)}")
if grade_dates_missing:
    print(f"   ⚠️ No data for: {', '.join(grade_dates_missing)} — run the engine first to populate game logs")
        
# Fallback: if sheets didn't load, fetch from MLB API directly
if not sheets_loaded or grade_dates_missing:
    print("   🔄 Falling back to MLB Stats API...")
    # Get unique players we need to grade
    fallback_df = ungraded[ungraded['DATE'].isin(grade_dates_missing if grade_dates_missing else dates_to_grade)][['player', 'prop_type']].drop_duplicates()
    print(f"   Fetching logs for {len(fallback_df)} player/prop combinations...")

    # Search for each player and get their game logs
    SEASON = now_est.year if now_est.month >= 3 else now_est.year - 1

    for _, fallback_row in fallback_df.iterrows():
        player_name = fallback_row['player']
        prop_type = str(fallback_row.get('prop_type', '') or '')
        try:
            # Search for player
            search_url = f"{MLB_API}/people/search?names={requests.utils.quote(player_name)}&sportId=1&limit=1"
            resp = requests.get(search_url, timeout=10).json()
            people = resp.get('people', [])
            if not people:
                continue
            pid = people[0]['id']

            # Get game log
            is_pitcher_prop = prop_type.startswith('P_')
            stat_group = 'pitching' if is_pitcher_prop else 'hitting'
            log_url = f"{MLB_API}/people/{pid}/stats?stats=gameLog&group={stat_group}&season={SEASON}&sportId=1"
            log_resp = requests.get(log_url, timeout=10).json()
            splits = log_resp.get('stats', [{}])[0].get('splits', [])

            for split in splits:
                stat = split.get('stat', {})
                game_date = split.get('date', '')
                if is_pitcher_prop:
                    ip_raw = stat.get('inningsPitched', 0)
                    key = (player_name, game_date)
                    if key not in box_lookup:
                        box_lookup[key] = {}
                    strikeouts = safe_float(stat.get('strikeOuts', 0), 0)
                    hits_allowed = safe_float(stat.get('hits', 0), 0)
                    walks_allowed = safe_float(stat.get('baseOnBalls', 0), 0)
                    earned_runs = safe_float(stat.get('earnedRuns', 0), 0)
                    wins = 1.0 if safe_float(stat.get('wins', 0), 0) else 0.0
                    pitcher_outs = innings_to_outs(ip_raw)
                    ip_val = outs_to_ip_decimal(pitcher_outs)
                    quality_start = 1.0 if pitcher_outs >= 18 and earned_runs <= 3 else 0.0
                    box_lookup[key].update({
                        'P_SO': safe_float(stat.get('strikeOuts', 0), 0),
                        'P_H': safe_float(stat.get('hits', 0), 0),
                        'P_BB': safe_float(stat.get('baseOnBalls', 0), 0),
                        'P_ER': safe_float(stat.get('earnedRuns', 0), 0),
                        'IP': ip_val,
                        'P_OUTS': pitcher_outs,
                        'W': wins,
                        'SO': strikeouts,
                        'ER': earned_runs,
                        'DK_FP': round(
                            ip_val * 2.25 + strikeouts * 2 + wins * 4 +
                            earned_runs * -2 + hits_allowed * -0.6 + walks_allowed * -0.6 +
                            quality_start * 2.5,
                            2
                        ),
                        'UD_FP': round(
                            wins * 5 + quality_start * 5 + strikeouts * 3 + ip_val * 3 - earned_runs * 3,
                            2
                        ),
                    })
                else:
                    h = int(stat.get('hits', 0))
                    hr = int(stat.get('homeRuns', 0))
                    doubles = int(stat.get('doubles', 0))
                    triples = int(stat.get('triples', 0))
                    singles = h - hr - doubles - triples
                    tb = int(stat.get('totalBases', 0))

                    singles_pos = float(max(singles, 0))
                    box_lookup[(player_name, game_date)] = {
                        'H': float(h),
                        'HR': float(hr),
                        'RBI': float(stat.get('rbi', 0)),
                        'R': float(stat.get('runs', 0)),
                        'SB': float(stat.get('stolenBases', 0)),
                        'SO': float(stat.get('strikeOuts', 0)),
                        'BB': float(stat.get('baseOnBalls', 0)),
                        'TB': float(tb),
                        'AB': float(stat.get('atBats', 0)),
                        '1B': singles_pos,
                        '2B': float(doubles),
                        '3B': float(triples),
                        'HBP': float(stat.get('hitByPitch', 0)),
                        'DK_FP': round(singles_pos * 3 + doubles * 5 + triples * 8 + hr * 10 +
                                       float(stat.get('runs', 0)) * 2 + float(stat.get('rbi', 0)) * 2 +
                                       float(stat.get('baseOnBalls', 0)) * 2 + float(stat.get('stolenBases', 0)) * 5 +
                                       float(stat.get('strikeOuts', 0)) * -0.5, 2),
                        'UD_FP': round(singles_pos * 3 + doubles * 6 + triples * 8 + hr * 10 +
                                       float(stat.get('baseOnBalls', 0)) * 3 + float(stat.get('hitByPitch', 0)) * 3 +
                                       float(stat.get('rbi', 0)) * 2 + float(stat.get('runs', 0)) * 2 +
                                       float(stat.get('stolenBases', 0)) * 4, 2),
                    }
            time.sleep(0.2)
        except Exception as e:
            print(f"   ⚠️ Failed to fetch {player_name}: {e}")

    print(f"   ✅ Built {len(box_lookup)} box score entries from API")

print(f"\n📦 Total box score entries available: {len(box_lookup)}")

# Show available dates in box scores


# --- 5. GRADE EACH PICK ---
print("\n" + "=" * 60)
print("📝 GRADING PICKS")
print("=" * 60)

graded = 0
hits = 0
misses = 0
pushes = 0
dnp = 0
not_found = 0

# Map column names to indices for direct cell updates
col_idx = {h: i for i, h in enumerate(headers)}
actual_col = col_idx.get('ACTUAL_STAT')
hit_col = col_idx.get('HIT')
result_col = col_idx.get('RESULT')

if actual_col is None or hit_col is None:
    print("❌ Missing ACTUAL_STAT or HIT columns in Daily_Picks")
    print(f"   Available columns: {headers}")
    raise SystemExit

# Helper for column letters (handles AA, AB, etc.)
def col_letter(idx):
    if idx < 26:
        return chr(65 + idx)
    return chr(64 + idx // 26) + chr(65 + idx % 26)

# Collect cell updates for batch write
updates = []

for idx, pick in ungraded.iterrows():
    player = pick.get('player', '')
    date = pick.get('DATE', '')
    prop_type = pick.get('prop_type', 'H')
    prop_type = 'SO' if prop_type == 'Batter_SO' else prop_type
    prop_type = 'P_OUTS' if prop_type == 'OUTS' else prop_type
    line = pick.get('line', '')
    lean = (pick.get('lean', '') or '').upper()

    if not player or not date:
        continue

    line_val = safe_float(line)

    # Try exact name match first, then case-insensitive
    box = find_box_score(box_lookup, player, date)

    # Row in the sheet (1-indexed, +1 for header)
    sheet_row = int(idx) + 2
    date_has_logs = date in box_date_set

    if box is None:
        if not date_has_logs:
            print(f"   ⏳ {player} ({date}) — logs for this date are not available yet; leaving ungraded for retry")
            continue
        # Player didn't play on a date where logs exist
        updates.append({'range': f'{col_letter(actual_col)}{sheet_row}', 'value': 'DNP'})
        updates.append({'range': f'{col_letter(hit_col)}{sheet_row}', 'value': 'DNP'})
        if result_col is not None:
            updates.append({'range': f'{col_letter(result_col)}{sheet_row}', 'value': 'DNP'})
        dnp += 1
        print(f"   ⬜ {player} ({date}) — DNP / No box score found")
        continue

    # Look up the actual stat value
    actual = box.get(prop_type)
    if actual is None:
        not_found += 1
        print(f"   ❓ {player} ({date}) — prop_type '{prop_type}' not found in box score")
        print(f"      Available stats: {', '.join(box.keys())}")
        continue

    actual = safe_float(actual)
    hit_str, result_str = grade_pick(actual, line_val, lean)
    if hit_str == 'PUSH':
        pushes += 1
    elif hit_str == 'YES':
        hits += 1
    elif hit_str == 'NO':
        misses += 1

    graded += 1

    updates.append({'range': f'{col_letter(actual_col)}{sheet_row}', 'value': str(actual)})
    updates.append({'range': f'{col_letter(hit_col)}{sheet_row}', 'value': hit_str})
    if result_col is not None:
        updates.append({'range': f'{col_letter(result_col)}{sheet_row}', 'value': result_str})

    icon = "✅" if hit_str == "YES" else "❌" if hit_str == "NO" else "➖" if hit_str == "PUSH" else "⬜"
    print(f"   {icon} {player} | {prop_type} {lean} {line} → Actual: {actual} → {hit_str}")

# --- 6. BATCH UPDATE GOOGLE SHEETS ---
if updates:
    print(f"\n📤 Writing {len(updates)} cell updates to Google Sheets...")
    cells = [{'range': u['range'], 'values': [[u['value']]]} for u in updates]
    ws.batch_update(cells)
    print("✅ Sheet updated!")
else:
    print("\n⚠️ No updates to write.")

# --- 7. SCORE BACKTEST ---
print("\n📊 Building Score_Backtest...")
try:
    ws_picks = sh.worksheet('Daily_Picks')
    df_graded = pd.DataFrame(ws_picks.get_all_records())
    if 'HIT' not in df_graded.columns:
        print("   ⚠️ Daily_Picks has no HIT column — skipping Score_Backtest")
    else:
        df_backtest_source = df_graded[df_graded['HIT'].isin(['YES', 'NO'])].copy()
        df_backtest = build_score_backtest(df_backtest_source)
        if df_backtest.empty:
            print("   ⚠️ No graded picks with edge scores yet — Score_Backtest skipped")
        else:
            write_dataframe_sheet(sh, 'Score_Backtest', df_backtest)
            date_count = df_backtest_source['DATE'].nunique() if 'DATE' in df_backtest_source.columns else '?'
            print(f"\n📊 Score backtest: {len(df_backtest)} rows across {date_count} dates, {len(df_backtest_source)} graded picks")
            score_cols = ['H_EDGE_SCORE', 'POWER_EDGE_SCORE', 'P_SO_EDGE_SCORE', 'P_ER_RISK_SCORE']
            for score_col in score_cols:
                overall = df_backtest[
                    (df_backtest['SCORE'] == score_col) &
                    (df_backtest['PROP_TYPE'] == 'ALL') &
                    (df_backtest['CONFIDENCE'] == 'ALL')
                ]
                if not overall.empty:
                    print(f"   {score_col}:")
                    for _, r in overall.iterrows():
                        print(f"      {r['BUCKET']}: {r['HIT_RATE'] * 100:.0f}% ({int(r['WINS'])}/{int(r['PICKS'])})")
except Exception as e:
    print(f"   ⚠️ Score_Backtest failed: {e}")

# --- 7. SUMMARY ---
total_decided = hits + misses
hit_rate = (hits / total_decided * 100) if total_decided > 0 else 0
runlog.hits = hits
runlog.misses = misses
runlog.dnp_count = dnp
runlog.not_found_count = not_found
runlog.picks_graded = hits + misses

print("\n" + "=" * 60)
print("📊 GRADING COMPLETE")
print("=" * 60)
print(f"   ✅ Hits:      {hits}")
print(f"   ❌ Misses:    {misses}")
print(f"   ➖ Pushes:    {pushes}")
print(f"   ⬜ DNP:       {dnp}")
print(f"   ❓ Not found: {not_found}")
print(f"   📈 Hit Rate:  {hits}/{total_decided} ({hit_rate:.1f}%)")
print(f"   📋 Dates:     {', '.join(dates_to_grade)}")
print("=" * 60)

# --- 8. SHOW CUMULATIVE RECORD ---
print("\n📊 Cumulative Record (all graded picks):")
ws_fresh = sh.worksheet('Daily_Picks')
all_fresh = ws_fresh.get_all_records()
df_all = pd.DataFrame(all_fresh)

if 'HIT' in df_all.columns:
    total_yes = len(df_all[df_all['HIT'] == 'YES'])
    total_no = len(df_all[df_all['HIT'] == 'NO'])
    total_push = len(df_all[df_all['HIT'] == 'PUSH'])
    total_dnp = len(df_all[df_all['HIT'] == 'DNP'])
    total_dec = total_yes + total_no
    cum_rate = (total_yes / total_dec * 100) if total_dec > 0 else 0

    print(f"   Record: {total_yes}-{total_no} ({cum_rate:.1f}%)")
    print(f"   Pushes: {total_push} | DNPs: {total_dnp}")

    if 'lean' in df_all.columns:
        print("\n   By Side:")
        side_series = df_all['lean'].fillna('').astype(str).str.upper().replace({'FADE': 'UNDER'})
        for side in ['OVER', 'UNDER']:
            side_df = df_all[side_series == side]
            side_yes = len(side_df[side_df['HIT'] == 'YES'])
            side_no = len(side_df[side_df['HIT'] == 'NO'])
            side_dec = side_yes + side_no
            if side_dec > 0:
                print(f"   {side}: {side_yes}-{side_no} ({side_yes/side_dec*100:.0f}%)")

    # By confidence tier
    print("\n   By Confidence:")
    for tier in ['SMASH', 'STRONG', 'LEAN']:
        tier_df = df_all[df_all['confidence'].str.upper() == tier]
        tier_yes = len(tier_df[tier_df['HIT'] == 'YES'])
        tier_no = len(tier_df[tier_df['HIT'] == 'NO'])
        tier_dec = tier_yes + tier_no
        if tier_dec > 0:
            print(f"   {tier}: {tier_yes}-{tier_no} ({tier_yes/tier_dec*100:.0f}%)")

    # By prop type
    if 'prop_type' in df_all.columns:
        print("\n   By Prop Type:")
        for ptype in sorted(df_all['prop_type'].unique()):
            if not ptype:
                continue
            p_df = df_all[df_all['prop_type'] == ptype]
            p_yes = len(p_df[p_df['HIT'] == 'YES'])
            p_no = len(p_df[p_df['HIT'] == 'NO'])
            p_dec = p_yes + p_no
            if p_dec > 0:
                print(f"   {ptype}: {p_yes}-{p_no} ({p_yes/p_dec*100:.0f}%)")

    # By date
    print("\n   By Date:")
    for date in sorted(df_all['DATE'].unique()):
        d_df = df_all[df_all['DATE'] == date]
        d_yes = len(d_df[d_df['HIT'] == 'YES'])
        d_no = len(d_df[d_df['HIT'] == 'NO'])
        d_dec = d_yes + d_no
        if d_dec > 0:
            print(f"   {date}: {d_yes}-{d_no} ({d_yes/d_dec*100:.0f}%)")
        else:
            d_dnp = len(d_df[d_df['HIT'] == 'DNP'])
            d_empty = len(d_df[d_df['HIT'].isin(['', None])])
            print(f"   {date}: ungraded ({d_empty}) / DNP ({d_dnp})")

    if 'RUN_NUMBER' in df_all.columns:
        print("\n   By Run Number:")
        run_series = pd.to_numeric(df_all['RUN_NUMBER'], errors='coerce')
        for run_no in sorted(run_series.dropna().astype(int).unique()):
            r_df = df_all[run_series == run_no]
            r_yes = len(r_df[r_df['HIT'] == 'YES'])
            r_no = len(r_df[r_df['HIT'] == 'NO'])
            r_dec = r_yes + r_no
            if r_dec > 0:
                print(f"   Run {run_no}: {r_yes}-{r_no} ({r_yes/r_dec*100:.0f}%)")

    print_clv_summary(df_all)
    print_winning_combo_tracker(df_all, dates_to_grade)

print("\n⚾ Done! Run this every morning after games.")
print("💡 Tip: The cumulative hit rate feeds back into the +EV engine — the more data, the sharper the edge calculations.")

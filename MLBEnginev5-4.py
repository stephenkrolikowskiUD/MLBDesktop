# @title ⚾ MLB Dashboard Engine (v1.3.0 — Underdog Scoring) — 5-4 Baseline
!pip install pybaseball requests pandas numpy gspread google-auth gspread-dataframe google-genai
import pandas as pd
import numpy as np
import requests
import json
import time
import math
import re
import unicodedata
import os
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import pytz
import gspread
from gspread_dataframe import set_with_dataframe
from google.auth import default
from google.oauth2.service_account import Credentials
from google import genai
from google.genai import types
import warnings
warnings.filterwarnings('ignore')

# --- 1. AUTHENTICATION & SETUP ---
print("Authenticating with Google...")

SHEET_NAME = 'MLB_Dashboard_Data'
MLB_API = "https://statsapi.mlb.com/api/v1"
SNAPSHOT_DATE = "2026-05-04"

now_est = datetime.now(pytz.timezone('US/Eastern'))
today_str = now_est.strftime('%Y-%m-%d')
timestamp_est = now_est.strftime('%Y-%m-%d %I:%M:%S %p EST')
eastern = pytz.timezone('US/Eastern')

def derive_mlb_season_context(now=None):
    now = now or datetime.now(eastern)
    season = now.year if now.month >= 3 else now.year - 1
    opening_day = eastern.localize(datetime(season, 3, 25))
    in_season = now >= opening_day or now.month >= 4
    schedule_date = now.strftime('%Y-%m-%d') if in_season else f'{season}-07-02'
    return season, opening_day, schedule_date, in_season

def innings_to_outs(ip_val):
    if pd.isna(ip_val):
        return 0
    s = str(ip_val).strip()
    if not s:
        return 0
    try:
        whole_str, frac_str = s.split('.', 1)
    except ValueError:
        try:
            return int(round(float(s) * 3))
        except (TypeError, ValueError):
            return 0
    try:
        whole = int(whole_str or 0)
        frac = int(frac_str[:1] or 0)
    except ValueError:
        return 0
    if len(frac_str) > 1 or frac > 2:
        try:
            return int(round(float(s) * 3))
        except (TypeError, ValueError):
            return 0
    frac = max(0, min(frac, 2))
    return whole * 3 + frac

def outs_to_ip(outs):
    try:
        outs = int(outs or 0)
    except (TypeError, ValueError):
        return 0.0
    whole, rem = divmod(outs, 3)
    return float(f"{whole}.{rem}")

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
        colab_auth.authenticate_user()
        creds, _ = default(scopes=scopes)
        print("✅ Google auth via Colab")
        return gspread.authorize(creds)
    except Exception as e:
        raise RuntimeError("Google auth unavailable. Set GOOGLE_SERVICE_ACCOUNT_JSON or run in Colab.") from e

def load_secret(name, prompt_text=None, allow_missing=False):
    env_val = os.environ.get(name)
    if env_val:
        print(f"🔐 Loaded {name} from environment!")
        return env_val
    try:
        from google.colab import userdata
        colab_val = userdata.get(name)
        if colab_val:
            print(f"🔐 Loaded {name} from Colab userdata!")
            return colab_val
    except Exception:
        pass
    if allow_missing:
        return None
    import getpass
    return getpass.getpass(prompt_text or f"Paste your {name}: ")

def normalize_player_name(name):
    text = unicodedata.normalize('NFKD', str(name or ''))
    text = ''.join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower()
    text = re.sub(r"[’'`\.]", "", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def normalize_prop_metric(metric):
    text = str(metric or '').strip().upper()
    text = re.sub(r"\s+", "", text)
    if text == 'BATTER_SO':
        return 'SO'
    return text

def normalize_confidence(val):
    conf = str(val or '').strip().upper()
    return conf if conf in {'SMASH', 'STRONG', 'LEAN'} else 'LEAN'

def parse_gemini_json_array(raw):
    cleaned = str(raw or '').strip()
    if cleaned.startswith('```'):
        cleaned = cleaned.split('\n', 1)[1] if '\n' in cleaned else cleaned[3:]
        cleaned = cleaned.rsplit('```', 1)[0].strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        last_complete = cleaned.rfind('}')
        if last_complete > 0:
            return json.loads(cleaned[:last_complete + 1] + ']')
        raise

def promote_consensus_confidence(confidence, consensus_count):
    conf = normalize_confidence(confidence)
    if consensus_count < 2:
        return conf
    if conf == 'LEAN':
        return 'STRONG'
    if conf == 'STRONG':
        return 'SMASH'
    return conf

def build_consensus_pick_pool(pick_lists):
    grouped = {}
    for run_idx, picks in enumerate(pick_lists, start=1):
        for pick in picks or []:
            player_key = normalize_player_name(pick.get('player', ''))
            prop_key = normalize_prop_metric(pick.get('prop_type', ''))
            lean_key = str(pick.get('lean', '') or '').strip().upper()
            if not player_key or not prop_key or not lean_key:
                continue
            key = (player_key, prop_key, lean_key)
            entry = grouped.setdefault(key, {'pick': dict(pick), 'count': 0, 'runs': [], 'best_rank': 999})
            if run_idx not in entry['runs']:
                entry['runs'].append(run_idx)
                entry['count'] += 1
            try:
                rank_val = int(float(pick.get('rank', 999)))
            except (TypeError, ValueError):
                rank_val = 999
            if rank_val < entry['best_rank']:
                entry['pick'] = dict(pick)
                entry['best_rank'] = rank_val
    merged = []
    for entry in grouped.values():
        pick = dict(entry['pick'])
        pick['CONSENSUS_COUNT'] = entry['count']
        pick['CONSENSUS_RUNS'] = ','.join(str(r) for r in entry['runs'])
        pick['CONSENSUS_TAG'] = f"CONSENSUS {entry['count']}/3" if entry['count'] >= 2 else ''
        pick['confidence'] = promote_consensus_confidence(pick.get('confidence'), entry['count'])
        merged.append(pick)
    merged.sort(key=lambda pk: (-int(pk.get('CONSENSUS_COUNT', 1)), float(pk.get('rank', 999) or 999)))
    for idx, pick in enumerate(merged, start=1):
        pick['rank'] = idx
    return merged

def normalize_game_date(val):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return ''
    text = str(val).strip()
    if not text:
        return ''
    if ' ' in text:
        text = text.split(' ', 1)[0]
    try:
        return pd.to_datetime(text, errors='coerce').strftime('%Y-%m-%d')
    except Exception:
        return text[:10]

def load_existing_log_sheet(sheet_name, keep_cols, numeric_cols):
    try:
        ws = sh.worksheet(sheet_name)
        rows = ws.get_all_records()
    except Exception:
        return pd.DataFrame(columns=keep_cols)
    if not rows:
        return pd.DataFrame(columns=keep_cols)
    df = pd.DataFrame(rows)
    for col in keep_cols:
        if col not in df.columns:
            df[col] = np.nan
    df = df[keep_cols].copy()
    if 'game_date' in df.columns:
        df['game_date'] = df['game_date'].map(normalize_game_date)
    if 'player_id' in df.columns:
        df['player_id'] = pd.to_numeric(df['player_id'], errors='coerce')
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    return df

def load_existing_daily_picks(sheet, target_date):
    try:
        ws = sheet.worksheet('Daily_Picks')
        rows = ws.get_all_records()
    except Exception:
        return pd.DataFrame()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if 'DATE' not in df.columns:
        return pd.DataFrame()
    df['DATE'] = df['DATE'].map(normalize_game_date)
    return df[df['DATE'] == target_date].copy()

def refresh_clv_frame(df_existing, target_date, props_df, timestamp_label):
    if df_existing is None or df_existing.empty or props_df is None or props_df.empty:
        return df_existing
    df_existing = df_existing.copy()
    for col in ['CLV_OPEN_LINE', 'CLV_LATEST_LINE', 'CLV_DELTA', 'CLV_LAST_UPDATE']:
        if col not in df_existing.columns:
            df_existing[col] = ''
    line_map = {}
    for _, prop in props_df.iterrows():
        try:
            latest_line = float(prop.get('DK_LINE'))
        except (TypeError, ValueError):
            continue
        key = (
            normalize_player_name(prop.get('PLAYER_NAME', '')),
            str(prop.get('METRIC', '')).strip().upper(),
        )
        line_map[key] = latest_line
    date_series = df_existing.get('DATE', pd.Series(dtype=object)).map(normalize_game_date)
    for idx, row in df_existing[date_series == target_date].iterrows():
        metric = str(row.get('prop_type', '')).strip().upper()
        if metric == 'BATTER_SO':
            metric = 'SO'
        key = (normalize_player_name(row.get('player', '')), metric)
        if key not in line_map:
            continue
        latest_line = line_map[key]
        open_raw = row.get('CLV_OPEN_LINE', '') or row.get('line', '')
        try:
            open_line = float(open_raw)
        except (TypeError, ValueError):
            open_line = None
        if open_line is not None:
            df_existing.at[idx, 'CLV_OPEN_LINE'] = open_line
            df_existing.at[idx, 'CLV_DELTA'] = round(latest_line - open_line, 1)
        df_existing.at[idx, 'CLV_LATEST_LINE'] = latest_line
        df_existing.at[idx, 'CLV_LAST_UPDATE'] = timestamp_label
    return df_existing

def build_batter_sample_flags(log_df, ref_date=None):
    cols = ['player_id', 'player_name', 'L5_GAMES_PLAYED', 'GAMES_LAST_7D', 'LIMITED_SAMPLE', 'RETURNING']
    if log_df is None or log_df.empty:
        return pd.DataFrame(columns=cols)
    ref_ts = pd.to_datetime(ref_date or datetime.now(pytz.timezone('US/Eastern')).strftime('%Y-%m-%d'))
    last7_cutoff = ref_ts - pd.Timedelta(days=6)
    rows = []
    for player_id, group in log_df.groupby('player_id'):
        grp = group.sort_values('game_date').copy()
        if grp.empty:
            continue
        ud_vals = pd.to_numeric(grp['UD_FP'], errors='coerce').dropna()
        l5_games = int(min(5, len(grp)))
        season_avg = float(ud_vals.mean()) if len(ud_vals) else 0.0
        l5_avg = float(ud_vals.tail(5).mean()) if len(ud_vals) else 0.0
        game_dates = pd.to_datetime(grp['game_date'], errors='coerce')
        games_last_7d = int(((game_dates >= last7_cutoff) & (game_dates <= ref_ts)).sum())
        limited_sample = l5_games < 3
        returning = bool(season_avg > 0 and l5_avg < (0.7 * season_avg) and games_last_7d < 4)
        rows.append({
            'player_id': player_id,
            'player_name': grp['player_name'].iloc[-1],
            'L5_GAMES_PLAYED': l5_games,
            'GAMES_LAST_7D': games_last_7d,
            'LIMITED_SAMPLE': limited_sample,
            'RETURNING': returning,
        })
    return pd.DataFrame(rows, columns=cols)

SEASON, OPENING_DAY, schedule_date, IN_SEASON = derive_mlb_season_context(now_est)
gc = get_gspread_client()

if IN_SEASON:
    print(f"🟢 Regular season mode — {SEASON}")
else:
    print(f"🟡 Pre-season detected — using {SEASON} data for testing")

print(f"📅 Schedule date: {schedule_date}")
print(f"📆 Season: {SEASON}")

try:
    sh = gc.open(SHEET_NAME)
    print(f"✅ Connected to Google Sheet: {SHEET_NAME}")
except Exception as e:
    print(f"❌ Error: {e}")

ODDS_API_KEY = load_secret('ODDS_API_KEY', '🔑 Paste your Odds API Key: ')
OPENWEATHER_API_KEY = load_secret('OPENWEATHER_API_KEY', '🌤️ Paste your OpenWeather API Key: ')
GEMINI_API_KEY = load_secret('GEMINI_API_KEY', allow_missing=True)
if GEMINI_API_KEY:
    print("🔐 Gemini API key ready!")
else:
    print("⚠️ No Gemini API key found — AI picks will be skipped.")

# --- 2. FETCH ALL MLB TEAMS ---
print("\nFetching MLB teams...")
teams_resp = requests.get(f"{MLB_API}/teams?sportId=1&season={SEASON}").json()
team_list = []
for team in teams_resp.get('teams', []):
    team_list.append({
        'team_id': team['id'],
        'team_name': team['name'],
        'team_abbr': team.get('abbreviation', ''),
        'venue_name': team.get('venue', {}).get('name', ''),
        'venue_id': team.get('venue', {}).get('id', '')
    })
df_teams = pd.DataFrame(team_list)
team_id_to_abbr = dict(zip(df_teams['team_id'], df_teams['team_abbr']))
print(f"✅ Loaded {len(df_teams)} MLB teams")

# --- 3. FETCH BATTER GAME LOGS (PARALLEL) ---
print(f"\nFetching batter game logs for {SEASON} season...")
print("⚡ Using parallel fetching...")

def get_qualified_batters(season):
    url = f"{MLB_API}/stats?stats=season&group=hitting&sportId=1&season={season}&limit=300"
    resp = requests.get(url).json()
    batters = []
    days_into_season = (now_est - OPENING_DAY).days if now_est >= OPENING_DAY else 0
    min_pa = 1 if days_into_season <= 14 else min(max(days_into_season, 1), 100)
    print(f"   PA threshold: {min_pa} (day {days_into_season} of season)")
    for split in resp.get('stats', [{}])[0].get('splits', []):
        stat = split.get('stat', {})
        player = split.get('player', {})
        team = split.get('team', {})
        if int(stat.get('plateAppearances', 0)) >= min_pa:
            batters.append({
                'player_id': player.get('id'),
                'player_name': player.get('fullName', ''),
                'team_id': team.get('id'),
                'team_abbr': team_id_to_abbr.get(team.get('id'), ''),
                'pa': int(stat.get('plateAppearances', 0))
            })
    return batters

def get_player_game_log(player_id, season):
    url = f"{MLB_API}/people/{player_id}/stats?stats=gameLog&group=hitting&season={season}&sportId=1"
    try:
        resp = requests.get(url, timeout=10).json()
        splits = resp.get('stats', [{}])[0].get('splits', [])
        games = []
        for split in splits:
            stat = split.get('stat', {})
            opp_id = split.get('opponent', {}).get('id')
            games.append({
                'player_id': player_id,
                'game_date': split.get('date', ''),
                'team_abbr': team_id_to_abbr.get(split.get('team', {}).get('id'), ''),
                'opp_abbr': team_id_to_abbr.get(opp_id, split.get('opponent', {}).get('abbreviation', '')),
                'home_away': 'Home' if split.get('isHome', True) else 'Away',
                'AB': int(stat.get('atBats', 0)),
                'H': int(stat.get('hits', 0)),
                'HR': int(stat.get('homeRuns', 0)),
                'RBI': int(stat.get('rbi', 0)),
                'R': int(stat.get('runs', 0)),
                'SB': int(stat.get('stolenBases', 0)),
                'SO': int(stat.get('strikeOuts', 0)),
                'BB': int(stat.get('baseOnBalls', 0)),
                'TB': int(stat.get('totalBases', 0)),
                '2B': int(stat.get('doubles', 0)),
                '3B': int(stat.get('triples', 0)),
                'HBP': int(stat.get('hitByPitch', 0)),  # v1.3.0: Added for Underdog scoring
            })
        return games
    except Exception:
        return []

BATTER_LOG_BASE_COLS = ['player_id', 'player_name', 'game_date', 'team_abbr', 'opp_abbr', 'home_away',
                        'AB', 'H', 'HR', 'RBI', 'R', 'SB', 'SO', 'BB', 'TB', '2B', '3B', 'HBP']
BATTER_LOG_NUMERIC_COLS = ['player_id', 'AB', 'H', 'HR', 'RBI', 'R', 'SB', 'SO', 'BB', 'TB', '2B', '3B', 'HBP']
existing_batter_logs = load_existing_log_sheet('Batter_Game_Logs', BATTER_LOG_BASE_COLS, BATTER_LOG_NUMERIC_COLS)
latest_batter_date_by_pid = {}
if len(existing_batter_logs) > 0:
    latest_batter_date_by_pid = existing_batter_logs.dropna(subset=['player_id']).groupby('player_id')['game_date'].max().to_dict()
    latest_seed_date = max(latest_batter_date_by_pid.values()) if latest_batter_date_by_pid else ''
    if latest_seed_date:
        print(f"♻️ Seeded Batter_Game_Logs through {latest_seed_date} ({len(existing_batter_logs)} existing rows)")
else:
    print("🆕 No existing Batter_Game_Logs seed found — full batter fetch")

qualified_batters = get_qualified_batters(SEASON)
print(f"✅ Found {len(qualified_batters)} qualified batters")

def fetch_one_batter_log(batter):
    logs = get_player_game_log(batter['player_id'], SEASON)
    cutoff = latest_batter_date_by_pid.get(batter['player_id'])
    if cutoff:
        logs = [log for log in logs if normalize_game_date(log.get('game_date')) > cutoff]
    for log in logs:
        log['player_name'] = batter['player_name']
        log['game_date'] = normalize_game_date(log['game_date'])
    return logs

all_game_logs = []
start_time = time.time()
with ThreadPoolExecutor(max_workers=15) as executor:
    futures = {executor.submit(fetch_one_batter_log, b): b for b in qualified_batters}
    done_count = 0
    for future in as_completed(futures):
        result = future.result()
        all_game_logs.extend(result)
        done_count += 1
        if done_count % 50 == 0:
            print(f"   Fetched {done_count}/{len(qualified_batters)} players...")

elapsed = time.time() - start_time
new_batter_logs = pd.DataFrame(all_game_logs, columns=BATTER_LOG_BASE_COLS)
combined_batter_logs = pd.concat([existing_batter_logs, new_batter_logs], ignore_index=True)
if len(combined_batter_logs) > 0:
    combined_batter_logs['game_date'] = combined_batter_logs['game_date'].map(normalize_game_date)
    dedupe_cols = ['player_id', 'game_date', 'opp_abbr', 'home_away', 'AB', 'H', 'HR', 'RBI', 'R', 'SB', 'SO', 'BB', 'TB', '2B', '3B', 'HBP']
    combined_batter_logs = combined_batter_logs.drop_duplicates(subset=dedupe_cols, keep='last')
    combined_batter_logs['game_date'] = pd.to_datetime(combined_batter_logs['game_date'], errors='coerce')
    df_logs = combined_batter_logs.sort_values(['player_id', 'game_date']).reset_index(drop=True)
else:
    df_logs = combined_batter_logs
print(f"✅ Fetched {len(new_batter_logs)} new batter logs; {len(df_logs)} combined logs across {df_logs['player_name'].nunique() if len(df_logs) > 0 else 0} players in {elapsed:.1f}s")

# --- 4. CALCULATE METRICS & ROLLING AVERAGES ---
print("\nCalculating metrics and rolling averages...")

df_logs['1B'] = df_logs['H'] - df_logs['HR'] - df_logs['2B'] - df_logs['3B']
df_logs['DK_FP'] = (
    df_logs['1B'] * 3 + df_logs['2B'] * 5 + df_logs['3B'] * 8 +
    df_logs['HR'] * 10 + df_logs['R'] * 2 + df_logs['RBI'] * 2 +
    df_logs['BB'] * 2 + df_logs['SB'] * 5 + df_logs['SO'] * -0.5
).round(2)

# v1.3.0: Underdog Fantasy scoring: 1B=3, 2B=6, 3B=8, HR=10, BB=3, HBP=3, RBI=2, R=2, SB=4
df_logs['UD_FP'] = (
    df_logs['1B'] * 3 + df_logs['2B'] * 6 + df_logs['3B'] * 8 +
    df_logs['HR'] * 10 + df_logs['BB'] * 3 + df_logs['HBP'] * 3 +
    df_logs['RBI'] * 2 + df_logs['R'] * 2 + df_logs['SB'] * 4
).round(2)

# v1.3.0: Added 1B, 3B, HBP, UD_FP to rolling averages
metrics = ['H', 'HR', 'RBI', 'R', 'SB', 'SO', 'BB', 'TB', '2B', '3B', '1B', 'HBP', 'AB', 'DK_FP', 'UD_FP']
windows = {'L7': 7, 'L14': 14, 'L30': 30}

df_logs = df_logs.set_index('game_date').sort_index()
for m in metrics:
    grp = df_logs.groupby('player_id')[m]
    df_logs[f'Seas_{m}'] = grp.transform(lambda x: x.expanding().mean()).round(3)
    for label, w in windows.items():
        df_logs[f'{label}_{m}'] = grp.transform(lambda x: x.rolling(w, min_periods=1).mean()).round(3)

for label, w in windows.items():
    grp_h = df_logs.groupby('player_id')['H'].transform(lambda x: x.rolling(w, min_periods=1).sum())
    grp_ab = df_logs.groupby('player_id')['AB'].transform(lambda x: x.rolling(w, min_periods=1).sum())
    df_logs[f'{label}_AVG'] = np.where(grp_ab > 0, (grp_h / grp_ab).round(3), 0)

seas_h = df_logs.groupby('player_id')['H'].transform(lambda x: x.expanding().sum())
seas_ab = df_logs.groupby('player_id')['AB'].transform(lambda x: x.expanding().sum())
df_logs['Seas_AVG'] = np.where(seas_ab > 0, (seas_h / seas_ab).round(3), 0)

df_logs = df_logs.reset_index()
df_sample_flags = build_batter_sample_flags(df_logs, today_str)
if not df_sample_flags.empty:
    df_logs = df_logs.merge(df_sample_flags, on=['player_id', 'player_name'], how='left')
    df_logs['LIMITED_SAMPLE'] = df_logs['LIMITED_SAMPLE'].fillna(False)
    df_logs['RETURNING'] = df_logs['RETURNING'].fillna(False)
    print(f"✅ Sample flags built — {int(df_sample_flags['LIMITED_SAMPLE'].sum())} LIMITED_SAMPLE, {int(df_sample_flags['RETURNING'].sum())} RETURNING")
else:
    df_logs['L5_GAMES_PLAYED'] = 0
    df_logs['GAMES_LAST_7D'] = 0
    df_logs['LIMITED_SAMPLE'] = False
    df_logs['RETURNING'] = False
df_logs['game_date'] = df_logs['game_date'].dt.strftime('%Y-%m-%d')
df_logs['LAST_UPDATED'] = timestamp_est
print(f"✅ Metrics calculated — {len(df_logs.columns)} columns total")
print(f"   📊 New columns: UD_FP, Seas_UD_FP, L7_UD_FP, L14_UD_FP, L30_UD_FP + 1B/3B/HBP rolling avgs")

# --- 5. LHP/RHP SPLITS (PARALLEL) ---
print("\nCalculating LHP/RHP splits...")

def get_player_splits(player_id, season):
    url = f"{MLB_API}/people/{player_id}/stats?stats=statSplits&group=hitting&season={season}&sportId=1&sitCodes=vl,vr"
    try:
        resp = requests.get(url, timeout=10).json()
        splits_data = {}
        for stat_group in resp.get('stats', []):
            for split in stat_group.get('splits', []):
                code = split.get('split', {}).get('code', '')
                stat = split.get('stat', {})
                ab = int(stat.get('atBats', 0))
                if ab == 0:
                    continue
                hand = 'vs_LHP' if code == 'vl' else 'vs_RHP' if code == 'vr' else None
                if not hand:
                    continue
                splits_data[hand] = {
                    'AB': ab, 'H': int(stat.get('hits', 0)),
                    'HR': int(stat.get('homeRuns', 0)), 'RBI': int(stat.get('rbi', 0)),
                    'TB': int(stat.get('totalBases', 0)),
                    'AVG': float(stat.get('avg', 0) or 0), 'OPS': float(stat.get('ops', 0) or 0),
                    'SO': int(stat.get('strikeOuts', 0)), 'BB': int(stat.get('baseOnBalls', 0)),
                }
        return splits_data
    except Exception:
        return {}

def fetch_one_batter_splits(batter):
    splits = get_player_splits(batter['player_id'], SEASON)
    row = {'player_id': batter['player_id'], 'player_name': batter['player_name'], 'team_abbr': batter['team_abbr']}
    for hand in ['vs_LHP', 'vs_RHP']:
        for k, v in splits.get(hand, {}).items():
            row[f'{hand}_{k}'] = v
    return row

splits_rows = []
start_time = time.time()
with ThreadPoolExecutor(max_workers=15) as executor:
    futures = {executor.submit(fetch_one_batter_splits, b): b for b in qualified_batters}
    done_count = 0
    for future in as_completed(futures):
        splits_rows.append(future.result())
        done_count += 1
        if done_count % 50 == 0:
            print(f"   Fetched splits {done_count}/{len(qualified_batters)}...")

elapsed = time.time() - start_time
df_splits = pd.DataFrame(splits_rows)
df_splits['LAST_UPDATED'] = timestamp_est
print(f"✅ LHP/RHP splits for {len(df_splits)} players in {elapsed:.1f}s")

# --- 6. HOME/AWAY SPLITS ---
print("\nCalculating Home/Away splits...")

split_metrics = ['H', 'HR', 'RBI', 'R', 'SB', 'SO', 'BB', 'TB', 'DK_FP', 'UD_FP', 'AB']
df_logs_temp = df_logs.copy()

ha_mean = df_logs_temp.groupby(['player_id', 'home_away'])[split_metrics].mean().round(3)
ha_count = df_logs_temp.groupby(['player_id', 'home_away'])['AB'].count().rename('GAMES')
df_home_away = ha_mean.join(ha_count).reset_index()

ha_pivot = df_home_away.pivot(index='player_id', columns='home_away', values=split_metrics)
ha_pivot.columns = [f'{stat}_{loc}' for stat, loc in ha_pivot.columns]
ha_count_pivot = df_home_away.pivot(index='player_id', columns='home_away', values='GAMES')
ha_count_pivot.columns = [f'{c}_GAMES' for c in ha_count_pivot.columns]
ha_pivot = ha_pivot.join(ha_count_pivot)

for m in split_metrics:
    for loc in ['Home', 'Away']:
        col = f'{m}_{loc}'
        if col not in ha_pivot.columns:
            ha_pivot[col] = np.nan
for loc in ['Home', 'Away']:
    gcol = f'{loc}_GAMES'
    if gcol not in ha_pivot.columns:
        ha_pivot[gcol] = np.nan

for m in split_metrics:
    hc, ac = f'{m}_Home', f'{m}_Away'
    if hc in ha_pivot.columns and ac in ha_pivot.columns:
        ha_pivot[f'{m}_SPLIT_DIFF'] = (ha_pivot[hc] - ha_pivot[ac]).where(
            ha_pivot[hc].notna() & ha_pivot[ac].notna(), other=np.nan).round(3)

player_names = df_logs_temp.groupby('player_id')['player_name'].first()
ha_pivot = ha_pivot.reset_index().merge(player_names, on='player_id', how='left')
ha_pivot = ha_pivot.reindex(sorted(ha_pivot.columns), axis=1)
cols = ['player_id', 'player_name'] + [c for c in ha_pivot.columns if c not in ['player_id', 'player_name']]
ha_pivot = ha_pivot[cols]
ha_pivot['LAST_UPDATED'] = timestamp_est
print(f"✅ Home/Away splits for {ha_pivot['player_name'].nunique()} players")

# --- 7. TONIGHT'S SCHEDULE & STARTING PITCHERS ---
print("\nFetching tonight's schedule and starting pitchers...")

games_tonight = []
pitcher_map = {}
venue_coords_dynamic = {}

DOMED_VENUES = {
    'Tropicana Field', 'Globe Life Field', 'loanDepot park',
    'Minute Maid Park', 'Daikin Park',
    'Rogers Centre', 'T-Mobile Park',
    'Chase Field', 'American Family Field'
}

try:
    sched_url = f"{MLB_API}/schedule?sportId=1&date={schedule_date}&hydrate=probablePitcher,team,venue&gameType=R"
    sched_resp = requests.get(sched_url).json()

    for date in sched_resp.get('dates', []):
        for game in date.get('games', []):
            home = game['teams']['home']
            away = game['teams']['away']
            venue = game.get('venue', {})
            home_abbr = home['team'].get('abbreviation', '')
            away_abbr = away['team'].get('abbreviation', '')
            home_pitcher = home.get('probablePitcher', {})
            away_pitcher = away.get('probablePitcher', {})

            venue_name = venue.get('name', '')
            venue_lat = venue.get('location', {}).get('defaultCoordinates', {}).get('latitude')
            venue_lon = venue.get('location', {}).get('defaultCoordinates', {}).get('longitude')

            if venue_lat and venue_lon:
                venue_coords_dynamic[venue_name] = (venue_lat, venue_lon)

            games_tonight.append({
                'game_pk': game['gamePk'], 'home_abbr': home_abbr, 'away_abbr': away_abbr,
                'home_team_id': home['team']['id'], 'away_team_id': away['team']['id'],
                'venue_name': venue_name, 'venue_id': venue.get('id', ''),
                'venue_lat': venue_lat, 'venue_lon': venue_lon,
                'game_time': game.get('gameDate', ''),
                'home_pitcher_id': home_pitcher.get('id'),
                'home_pitcher_name': home_pitcher.get('fullName', 'TBD'),
                'away_pitcher_id': away_pitcher.get('id'),
                'away_pitcher_name': away_pitcher.get('fullName', 'TBD'),
            })

            if home_pitcher.get('id'):
                pitcher_map[away_abbr] = {'opp_pitcher_id': home_pitcher['id'], 'opp_pitcher_name': home_pitcher.get('fullName', 'TBD')}
            if away_pitcher.get('id'):
                pitcher_map[home_abbr] = {'opp_pitcher_id': away_pitcher['id'], 'opp_pitcher_name': away_pitcher.get('fullName', 'TBD')}

    print(f"⚾ Found {len(games_tonight)} games on {schedule_date}")
    for g in games_tonight:
        print(f"   {g['away_abbr']} ({g['away_pitcher_name']}) @ {g['home_abbr']} ({g['home_pitcher_name']}) — {g['venue_name']}")
    if venue_coords_dynamic:
        print(f"📍 Dynamic venue coordinates loaded for {len(venue_coords_dynamic)} venues")
except Exception as e:
    print(f"❌ Schedule fetch failed: {e}")

VENUE_COORDS_FALLBACK = {
    'Angel Stadium': (33.8003, -117.8827), 'Busch Stadium': (38.6226, -90.1928),
    'Chase Field': (33.4455, -112.0667), 'Citi Field': (40.7571, -73.8458),
    'Citizens Bank Park': (39.9061, -75.1665), 'Comerica Park': (42.3390, -83.0485),
    'Coors Field': (39.7559, -104.9942), 'Dodger Stadium': (34.0739, -118.2400),
    'UNIQLO FIELD AT DODGER STADIUM': (34.0739, -118.2400),
    'Fenway Park': (42.3467, -71.0972), 'Globe Life Field': (32.7473, -97.0845),
    'Great American Ball Park': (39.0974, -84.5082),
    'Guaranteed Rate Field': (41.8299, -87.6338), 'Rate Field': (41.8299, -87.6338),
    'Kauffman Stadium': (39.0517, -94.4803), 'loanDepot park': (25.7781, -80.2196),
    'Minute Maid Park': (29.7573, -95.3555), 'Daikin Park': (29.7573, -95.3555),
    'Nationals Park': (38.8730, -77.0074),
    'Oakland Coliseum': (37.7516, -122.2005), 'Sutter Health Park': (38.5802, -121.5101), 'Oracle Park': (37.7786, -122.3893),
    'Oriole Park at Camden Yards': (39.2838, -76.6216), 'Petco Park': (32.7076, -117.1570),
    'PNC Park': (40.4469, -80.0058), 'Progressive Field': (41.4962, -81.6852),
    'Rogers Centre': (43.6414, -79.3894), 'T-Mobile Park': (47.5914, -122.3325),
    'Target Field': (44.9818, -93.2775), 'Tropicana Field': (27.7682, -82.6534),
    'Truist Park': (33.8908, -84.4678), 'Wrigley Field': (41.9484, -87.6553),
    'Yankee Stadium': (40.8296, -73.9262), 'American Family Field': (43.0280, -87.9712),
    'George M. Steinbrenner Field': (27.9789, -82.5034), 'Salt River Fields': (33.5453, -111.8847),
}

def get_venue_coords(venue_name):
    if venue_name in venue_coords_dynamic:
        return venue_coords_dynamic[venue_name]
    if venue_name in VENUE_COORDS_FALLBACK:
        return VENUE_COORDS_FALLBACK[venue_name]
    vl = venue_name.lower()
    for k, v in VENUE_COORDS_FALLBACK.items():
        if k.lower() == vl:
            return v
    return None

def get_pitcher_hand(pitcher_id):
    try:
        resp = requests.get(f"{MLB_API}/people/{pitcher_id}", timeout=10).json()
        return resp.get('people', [{}])[0].get('pitchHand', {}).get('code', 'R')
    except Exception:
        return 'R'

for team_abbr, info in pitcher_map.items():
    if info.get('opp_pitcher_id'):
        info['opp_pitcher_hand'] = get_pitcher_hand(info['opp_pitcher_id'])
        time.sleep(0.1)

print(f"✅ Pitcher handedness fetched for {len(pitcher_map)} teams")
for team, info in pitcher_map.items():
    print(f"   {team} faces {info['opp_pitcher_name']} ({info.get('opp_pitcher_hand','?')}HP)")

# --- 7.5 TWO-WAY PLAYER DETECTION (data-driven, future-proof) ---
# A two-way player is anyone who is BOTH a probable SP tonight AND a qualified batter.
# Ohtani is the known case, but this generalizes to any future two-way player.
tonight_sp_names = set()
for g in games_tonight:
    if g.get('home_pitcher_name') and g['home_pitcher_name'] != 'TBD':
        tonight_sp_names.add(g['home_pitcher_name'])
    if g.get('away_pitcher_name') and g['away_pitcher_name'] != 'TBD':
        tonight_sp_names.add(g['away_pitcher_name'])

qualified_batter_names = {b['player_name'] for b in qualified_batters}
two_way_tonight = tonight_sp_names & qualified_batter_names

if two_way_tonight:
    print(f"⚠️  TWO-WAY PLAYER ALERT: {', '.join(sorted(two_way_tonight))} pitching AND on qualified batter list")
    print(f"   → Batter props will carry LINEUP RISK flag (confirm lineup before betting)")
else:
    print(f"ℹ️  No two-way players pitching tonight")

# --- 8. BUILD TONIGHT'S BATTER SHEET (ROSTER-SAFE + EARLY-SEASON EXPANSION) ---
print("\nBuilding tonight's batter sheet...")

active_team_abbrs = set()
active_team_ids = {}
for g in games_tonight:
    active_team_abbrs.add(g['home_abbr'])
    active_team_abbrs.add(g['away_abbr'])
    active_team_ids[g['home_abbr']] = g['home_team_id']
    active_team_ids[g['away_abbr']] = g['away_team_id']

batter_current_team = {b['player_id']: b['team_abbr'] for b in qualified_batters}

most_recent = df_logs.sort_values('game_date').groupby('player_id').last().reset_index()
most_recent['team_abbr'] = most_recent['player_id'].map(batter_current_team)
most_recent['player_name'] = most_recent['player_id'].map(
    {b['player_id']: b['player_name'] for b in qualified_batters})
most_recent = most_recent[most_recent['team_abbr'].isin(active_team_abbrs)].copy()

print("   🔄 Early-season expansion: fetching full rosters for tonight's teams...")
expansion_batters = []
existing_ids = set(most_recent['player_id'].unique())

for team_abbr, team_id in active_team_ids.items():
    try:
        roster_url = f"{MLB_API}/teams/{team_id}/roster?rosterType=active&season={SEASON}"
        roster_resp = requests.get(roster_url, timeout=10).json()
        for entry in roster_resp.get('roster', []):
            pid = entry.get('person', {}).get('id')
            pname = entry.get('person', {}).get('fullName', '')
            pos_type = entry.get('position', {}).get('type', '')
            if pid and pid not in existing_ids and pos_type != 'Pitcher':
                expansion_batters.append({'player_id': pid, 'player_name': pname, 'team_abbr': team_abbr})
                existing_ids.add(pid)
    except Exception as e:
        print(f"   ⚠️ Roster fetch failed for {team_abbr}: {e}")

if expansion_batters:
    df_expansion = pd.DataFrame(expansion_batters)
    for col in most_recent.columns:
        if col not in df_expansion.columns:
            df_expansion[col] = np.nan
    most_recent = pd.concat([most_recent, df_expansion[most_recent.columns]], ignore_index=True)
    print(f"   ✅ Added {len(expansion_batters)} batters from roster expansion (no game logs yet)")
else:
    print(f"   ℹ️ No additional batters needed")

most_recent['opp_pitcher_name'] = most_recent['team_abbr'].map(
    {k: v['opp_pitcher_name'] for k, v in pitcher_map.items()})
most_recent['opp_pitcher_hand'] = most_recent['team_abbr'].map(
    {k: v.get('opp_pitcher_hand', 'R') for k, v in pitcher_map.items()})
most_recent['opp_abbr_tonight'] = most_recent['team_abbr'].map(
    {g['home_abbr']: g['away_abbr'] for g in games_tonight} |
    {g['away_abbr']: g['home_abbr'] for g in games_tonight})
most_recent['venue_tonight'] = most_recent['team_abbr'].map(
    {g['home_abbr']: g['venue_name'] for g in games_tonight} |
    {g['away_abbr']: g['venue_name'] for g in games_tonight})
most_recent['home_away_tonight'] = most_recent['team_abbr'].map(
    {g['home_abbr']: 'Home' for g in games_tonight} |
    {g['away_abbr']: 'Away' for g in games_tonight})

if 'Seas_OPS' in most_recent.columns:
    elite_ops_series = pd.to_numeric(most_recent['Seas_OPS'], errors='coerce')
else:
    elite_ops_series = pd.Series(dtype=float)
elite_ops_cutoff = max(0.850, float(elite_ops_series.dropna().quantile(0.85))) if not elite_ops_series.dropna().empty else 0.850
most_recent['IBB_RISK'] = False
most_recent['LINEUP_PROTECTION_NOTE'] = ""
most_recent['TEAM_SUPPORT_OPS1'] = np.nan
most_recent['TEAM_SUPPORT_OPS2'] = np.nan

for idx, row in most_recent.iterrows():
    ops = pd.to_numeric(pd.Series([row.get('Seas_OPS')]), errors='coerce').iloc[0]
    if pd.isna(ops) or ops < elite_ops_cutoff:
        continue
    teammates = most_recent[(most_recent['team_abbr'] == row.get('team_abbr')) & (most_recent['player_id'] != row.get('player_id'))]
    if 'Seas_OPS' in teammates.columns:
        teammate_ops = pd.to_numeric(teammates['Seas_OPS'], errors='coerce')
    else:
        teammate_ops = pd.Series(dtype=float)
    support_ops = sorted(teammate_ops.dropna().tolist(), reverse=True)
    top1 = support_ops[0] if len(support_ops) > 0 else np.nan
    top2 = support_ops[1] if len(support_ops) > 1 else np.nan
    most_recent.at[idx, 'TEAM_SUPPORT_OPS1'] = top1
    most_recent.at[idx, 'TEAM_SUPPORT_OPS2'] = top2
    weak_support = (pd.isna(top1) or top1 < 0.760) and (pd.isna(top2) or top2 < 0.720)
    if weak_support:
        top1_txt = f"{top1:.3f}" if pd.notna(top1) else "n/a"
        top2_txt = f"{top2:.3f}" if pd.notna(top2) else "n/a"
        most_recent.at[idx, 'IBB_RISK'] = True
        most_recent.at[idx, 'LINEUP_PROTECTION_NOTE'] = f"LINEUP RISK: weak lineup protection (support OPS {top1_txt}/{top2_txt})"

df_splits_merge = df_splits.copy()
for col in df_splits_merge.columns:
    if col not in ['player_id', 'player_name', 'team_abbr', 'LAST_UPDATED']:
        df_splits_merge.rename(columns={col: f'SPLIT_{col}'}, inplace=True)

most_recent = most_recent.merge(
    df_splits_merge[['player_id'] + [c for c in df_splits_merge.columns if 'SPLIT_' in c]],
    on='player_id', how='left')
ha_merge_cols = ['player_id', 'Home_GAMES', 'Away_GAMES', 'H_Home', 'H_Away', 'TB_Home', 'TB_Away',
                 'HR_Home', 'HR_Away', 'RBI_Home', 'RBI_Away', 'UD_FP_Home', 'UD_FP_Away']
ha_merge_cols = [c for c in ha_merge_cols if c in ha_pivot.columns]
if ha_merge_cols:
    most_recent = most_recent.merge(ha_pivot[ha_merge_cols], on='player_id', how='left')

split_stats = ['AVG', 'OPS', 'H', 'HR', 'TB', 'RBI', 'SO', 'BB']
for stat in split_stats:
    most_recent[f'vs_OPP_{stat}'] = np.where(
        most_recent['opp_pitcher_hand'] == 'L',
        most_recent.get(f'SPLIT_vs_LHP_{stat}', np.nan),
        most_recent.get(f'SPLIT_vs_RHP_{stat}', np.nan))

rolling_cols = [c for c in most_recent.columns if any(c.startswith(p) for p in ['L7_', 'L14_', 'L30_', 'Seas_'])]
ha_prompt_cols = ['Home_GAMES', 'Away_GAMES', 'H_Home', 'H_Away', 'TB_Home', 'TB_Away',
                  'HR_Home', 'HR_Away', 'RBI_Home', 'RBI_Away', 'UD_FP_Home', 'UD_FP_Away']
final_cols = (
    ['player_name', 'team_abbr', 'opp_abbr_tonight', 'opp_pitcher_name', 'opp_pitcher_hand', 'venue_tonight', 'home_away_tonight'] +
    [f'vs_OPP_{s}' for s in split_stats] + ha_prompt_cols + rolling_cols +
    ['IBB_RISK', 'LINEUP_PROTECTION_NOTE', 'TEAM_SUPPORT_OPS1', 'TEAM_SUPPORT_OPS2', 'LAST_UPDATED'])
final_cols = [c for c in final_cols if c in most_recent.columns]
df_tonight = most_recent[final_cols].copy()
df_tonight = df_tonight.sort_values('player_name').reset_index(drop=True)

game_log_count = df_tonight['L7_H'].notna().sum() if 'L7_H' in df_tonight.columns else 0
roster_only_count = len(df_tonight) - game_log_count
print(f"✅ Tonight's batter sheet built — {len(df_tonight)} active batters")
print(f"   ({game_log_count} with game logs, {roster_only_count} roster-only)")
print(f"   Columns: {len(df_tonight.columns)}")

# --- 8.5 BATTER vs STARTING PITCHER (CAREER) ---
print("\nFetching career batter vs SP stats...")
vs_sp_rows = []

def fetch_batter_vs_pitcher(batter_name, batter_id, pitcher_id, pitcher_name):
    url = f"{MLB_API}/people/{batter_id}/stats?stats=vsPlayer&group=hitting&opposingPlayerId={pitcher_id}"
    try:
        resp = requests.get(url, timeout=10).json()
        splits = resp.get('stats', [{}])[0].get('splits', [])
        if not splits:
            return None
        stat = splits[0].get('stat', {})
        ab = int(stat.get('atBats', 0))
        if ab == 0:
            return None
        return {
            'player_name': batter_name, 'opp_pitcher_name': pitcher_name,
            'AB': ab, 'H': int(stat.get('hits', 0)),
            'HR': int(stat.get('homeRuns', 0)), 'RBI': int(stat.get('rbi', 0)),
            'BB': int(stat.get('baseOnBalls', 0)), 'SO': int(stat.get('strikeOuts', 0)),
            'TB': int(stat.get('totalBases', 0)),
            'AVG': stat.get('avg', '.000'), 'OPS': stat.get('ops', '.000'),
        }
    except:
        return None

# Build batter→pitcher pairs from tonight's matchups
batter_pitcher_pairs = []
for b in qualified_batters:
    team = b['team_abbr']
    if team in pitcher_map and pitcher_map[team].get('opp_pitcher_id'):
        if team in active_team_abbrs:
            batter_pitcher_pairs.append((b['player_name'], b['player_id'],
                pitcher_map[team]['opp_pitcher_id'], pitcher_map[team]['opp_pitcher_name']))

print(f"   Fetching {len(batter_pitcher_pairs)} batter vs SP matchups...")
start_time = time.time()
with ThreadPoolExecutor(max_workers=15) as executor:
    futures = {executor.submit(fetch_batter_vs_pitcher, *pair): pair for pair in batter_pitcher_pairs}
    done_count = 0
    for future in as_completed(futures):
        result = future.result()
        if result:
            vs_sp_rows.append(result)
        done_count += 1
        if done_count % 50 == 0:
            print(f"   Fetched {done_count}/{len(batter_pitcher_pairs)}...")

elapsed = time.time() - start_time
df_vs_sp = pd.DataFrame(vs_sp_rows)
if len(df_vs_sp) > 0:
    df_vs_sp['LAST_UPDATED'] = timestamp_est
print(f"✅ Career vs SP data for {len(df_vs_sp)} matchups in {elapsed:.1f}s")

# --- 9. WEATHER ---
print("\nFetching weather for tonight's venues...")

def get_weather(lat, lon, api_key):
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}&units=imperial"
        resp = requests.get(url, timeout=10).json()
        return {
            'temp_f': round(resp['main']['temp'], 1), 'feels_like_f': round(resp['main']['feels_like'], 1),
            'humidity': resp['main']['humidity'], 'wind_mph': round(resp['wind']['speed'], 1),
            'wind_dir': resp['wind'].get('deg', 0), 'condition': resp['weather'][0]['main'],
            'description': resp['weather'][0]['description'], 'clouds_pct': resp['clouds']['all'],
        }
    except Exception as e:
        print(f"   ⚠️ Weather fetch failed: {e}")
        return None

def wind_direction_label(degrees):
    dirs = ['N','NNE','NE','ENE','E','ESE','SE','SSE','S','SSW','SW','WSW','W','WNW','NW','NNW']
    return dirs[round(degrees / 22.5) % 16]

weather_rows = []
seen_venues = set()
for game in games_tonight:
    venue = game['venue_name']
    if venue in seen_venues:
        continue
    seen_venues.add(venue)
    is_dome = venue.lower() in {v.lower() for v in DOMED_VENUES}
    coords = get_venue_coords(venue)
    if coords and OPENWEATHER_API_KEY:
        wx = get_weather(coords[0], coords[1], OPENWEATHER_API_KEY)
        if wx:
            weather_rows.append({
                'venue_name': venue, 'home_abbr': game['home_abbr'], 'away_abbr': game['away_abbr'],
                'is_dome': is_dome, 'temp_f': wx['temp_f'], 'feels_like_f': wx['feels_like_f'],
                'humidity': wx['humidity'], 'wind_mph': wx['wind_mph'],
                'wind_dir': wind_direction_label(wx['wind_dir']), 'wind_deg': wx['wind_dir'],
                'condition': wx['condition'], 'description': wx['description'], 'clouds_pct': wx['clouds_pct'],
                'weather_note': 'Dome/Roof — weather may not apply' if is_dome else '',
            })
            status = "🏟️ DOME" if is_dome else f"🌡️ {wx['temp_f']}°F"
            print(f"   {venue}: {status}, {wx['wind_mph']}mph {wind_direction_label(wx['wind_dir'])}, {wx['description']}")
    else:
        weather_rows.append({'venue_name': venue, 'home_abbr': game['home_abbr'], 'away_abbr': game['away_abbr'],
            'is_dome': is_dome, 'weather_note': 'Dome/Roof' if is_dome else 'No coordinates available'})
        print(f"   {venue}: {'🏟️ DOME' if is_dome else '⚠️ No coordinates'}")
    time.sleep(0.25)

df_weather = pd.DataFrame(weather_rows)
df_weather['LAST_UPDATED'] = timestamp_est
print(f"✅ Weather data for {len(df_weather)} venues")

# --- 10. BETTING ODDS ---
ODDS_SPORT = 'baseball_mlb'
ODDS_REGIONS = 'us'
ODDS_MARKETS = 'h2h,spreads,totals'

TEAM_NAME_TO_ABBR = {
    'Arizona Diamondbacks': 'AZ', 'Atlanta Braves': 'ATL', 'Baltimore Orioles': 'BAL',
    'Boston Red Sox': 'BOS', 'Chicago Cubs': 'CHC', 'Chicago White Sox': 'CWS',
    'Cincinnati Reds': 'CIN', 'Cleveland Guardians': 'CLE', 'Colorado Rockies': 'COL',
    'Detroit Tigers': 'DET', 'Houston Astros': 'HOU', 'Kansas City Royals': 'KC',
    'Los Angeles Angels': 'LAA', 'Los Angeles Dodgers': 'LAD', 'Miami Marlins': 'MIA',
    'Milwaukee Brewers': 'MIL', 'Minnesota Twins': 'MIN', 'New York Mets': 'NYM',
    'New York Yankees': 'NYY', 'Oakland Athletics': 'OAK', 'Philadelphia Phillies': 'PHI',
    'Pittsburgh Pirates': 'PIT', 'San Diego Padres': 'SD', 'San Francisco Giants': 'SF',
    'Seattle Mariners': 'SEA', 'St. Louis Cardinals': 'STL', 'Tampa Bay Rays': 'TB',
    'Texas Rangers': 'TEX', 'Toronto Blue Jays': 'TOR', 'Washington Nationals': 'WSH',
}

def fetch_odds(api_key):
    url = f"https://api.the-odds-api.com/v4/sports/{ODDS_SPORT}/odds/"
    params = {'apiKey': api_key, 'regions': ODDS_REGIONS, 'markets': ODDS_MARKETS, 'oddsFormat': 'american'}
    try:
        resp = requests.get(url, params=params, timeout=15)
        print(f"   API status: {resp.status_code}")
        print(f"   Quota: {resp.headers.get('x-requests-remaining', '?')} remaining / {resp.headers.get('x-requests-used', '?')} used")
        if resp.status_code != 200:
            return []
        return resp.json()
    except Exception as e:
        print(f"   ❌ Odds fetch failed: {e}")
        return []

def extract_odds(events, preferred_book='draftkings', fallback_book='fanduel'):
    rows = []
    for event in events:
        home, away = event.get('home_team', ''), event.get('away_team', '')
        bookmakers = event.get('bookmakers', [])
        book = next((b for b in bookmakers if b['key'] == preferred_book), None)
        if not book:
            book = next((b for b in bookmakers if b['key'] == fallback_book), None)
        if not book and bookmakers:
            book = bookmakers[0]
        if not book:
            continue
        row = {'home_team': home, 'away_team': away, 'commence_time': event.get('commence_time', ''), 'bookmaker': book.get('title', '')}
        for market in book.get('markets', []):
            mkey = market['key']
            for o in market.get('outcomes', []):
                if mkey == 'h2h':
                    if o['name'] == home: row['home_ml'] = o['price']
                    elif o['name'] == away: row['away_ml'] = o['price']
                elif mkey == 'spreads':
                    if o['name'] == home: row['home_spread'], row['home_spread_odds'] = o.get('point', ''), o['price']
                    elif o['name'] == away: row['away_spread'], row['away_spread_odds'] = o.get('point', ''), o['price']
                elif mkey == 'totals':
                    if o['name'] == 'Over': row['total'], row['over_odds'] = o.get('point', ''), o['price']
                    elif o['name'] == 'Under': row['under_odds'] = o['price']
        rows.append(row)
    return rows

if (now_est >= OPENING_DAY or now_est.month >= 4) and ODDS_API_KEY:
    print("\nFetching betting odds...")
    raw_odds = fetch_odds(ODDS_API_KEY)
    odds_rows = extract_odds(raw_odds)
    df_odds = pd.DataFrame(odds_rows)
    if len(df_odds) > 0:
        df_odds['total'] = pd.to_numeric(df_odds.get('total', pd.Series()), errors='coerce')
        df_odds['home_spread'] = pd.to_numeric(df_odds.get('home_spread', pd.Series()), errors='coerce')
        df_odds['home_implied_total'] = ((df_odds['total'] - df_odds['home_spread']) / 2).round(2)
        df_odds['away_implied_total'] = ((df_odds['total'] + df_odds['home_spread']) / 2).round(2)
        df_odds['home_abbr'] = df_odds['home_team'].map(TEAM_NAME_TO_ABBR)
        df_odds['away_abbr'] = df_odds['away_team'].map(TEAM_NAME_TO_ABBR)
        if active_team_abbrs:
            before_games = len(df_odds)
            df_odds = df_odds[
                df_odds['home_abbr'].isin(active_team_abbrs) &
                df_odds['away_abbr'].isin(active_team_abbrs)
            ].copy()
            print(f"🎯 Odds aligned to tonight's slate: {before_games} -> {len(df_odds)} games")
        df_odds['LAST_UPDATED'] = timestamp_est
        print(f"✅ Odds fetched for {len(df_odds)} games")
    else:
        df_odds = pd.DataFrame()
else:
    print("\n⏭️ Odds skipped")
    df_odds = pd.DataFrame()

df_schedule = pd.DataFrame(games_tonight)
df_schedule['LAST_UPDATED'] = timestamp_est

pitcher_rows_out = []
for team_abbr, info in pitcher_map.items():
    pitcher_rows_out.append({'team_abbr': team_abbr, 'opp_pitcher_id': info.get('opp_pitcher_id', ''),
        'opp_pitcher_name': info.get('opp_pitcher_name', 'TBD'), 'opp_pitcher_hand': info.get('opp_pitcher_hand', '')})
df_pitchers = pd.DataFrame(pitcher_rows_out)
df_pitchers['LAST_UPDATED'] = timestamp_est

# --- 10.5 DK PLAYER PROPS ---
print("\nFetching Live DraftKings Player Props...")

PROP_SPORT = 'baseball_mlb'
PROP_BOOKMAKER = 'draftkings'
MARKET_BATCHES = [
    'batter_hits,batter_total_bases,batter_home_runs,batter_rbis,batter_runs_scored',
    'batter_stolen_bases,batter_strikeouts,batter_walks,batter_singles,batter_doubles',
    'pitcher_strikeouts,pitcher_hits_allowed,pitcher_walks,pitcher_earned_runs,pitcher_outs',
]

market_mapping = {
    'batter_hits': 'H', 'batter_total_bases': 'TB', 'batter_home_runs': 'HR',
    'batter_rbis': 'RBI', 'batter_runs_scored': 'R', 'batter_stolen_bases': 'SB',
    'batter_strikeouts': 'Batter_SO', 'batter_walks': 'BB',
    'batter_singles': '1B', 'batter_doubles': '2B',
    'pitcher_strikeouts': 'P_SO', 'pitcher_hits_allowed': 'P_H',
    'pitcher_walks': 'P_BB', 'pitcher_earned_runs': 'P_ER', 'pitcher_outs': 'P_OUTS'}

BINARY_PROP_MARKETS = {
    'batter_home_runs': 0.5,
    'batter_stolen_bases': 0.5,
    'batter_singles': 0.5,
    'batter_doubles': 0.5,
}

def parse_prop_outcomes_for_market(mkt, metric_name):
    parsed_rows = []
    pd_dict = {}
    market_key = mkt.get('key', '')
    for oc in mkt.get('outcomes', []):
        player_name = (
            oc.get('description') or
            oc.get('participant') or
            oc.get('player') or
            ''
        )
        bet_type = str(oc.get('name', '')).strip()
        line_val = oc.get('point')
        odds_val = oc.get('price')

        if not player_name or odds_val is None:
            continue
        if line_val is None and market_key in BINARY_PROP_MARKETS:
            line_val = BINARY_PROP_MARKETS[market_key]
        if line_val is None:
            continue

        if player_name not in pd_dict:
            pd_dict[player_name] = {
                'PLAYER_NAME': player_name,
                'METRIC': metric_name,
                'DK_LINE': line_val,
                'OVER_ODDS': None,
                'UNDER_ODDS': None
            }

        if bet_type in {'Over', 'Yes'}:
            pd_dict[player_name]['OVER_ODDS'] = odds_val
        elif bet_type in {'Under', 'No'}:
            pd_dict[player_name]['UNDER_ODDS'] = odds_val
        else:
            # Some books occasionally emit a player label/name variant here; keep row but don't classify odds.
            continue

    parsed_rows.extend(pd_dict.values())
    return parsed_rows

df_props = pd.DataFrame()

if (now_est >= OPENING_DAY or now_est.month >= 4) and ODDS_API_KEY and len(games_tonight) > 0:
    try:
        ev_url = f'https://api.the-odds-api.com/v4/sports/{PROP_SPORT}/events'
        ev_resp = requests.get(ev_url, params={'apiKey': ODDS_API_KEY}, timeout=15)
        if ev_resp.status_code != 200:
            print(f"   ❌ Failed to fetch events: {ev_resp.text[:200]}")
        else:
            ev_data = ev_resp.json()
            tonight_team_names = {t['team_name'].lower() for t in team_list if t['team_abbr'] in active_team_abbrs}
            tonight_ids = [e['id'] for e in ev_data
                if e.get('home_team', '').lower() in tonight_team_names
                or e.get('away_team', '').lower() in tonight_team_names]
            print(f"   🏟️ Found {len(tonight_ids)} events for props...")

            prop_list = []
            dk_raw_market_keys = set()
            dk_parsed_market_keys = set()
            api_errors = 0
            for eid in tonight_ids:
                p_url = f'https://api.the-odds-api.com/v4/sports/{PROP_SPORT}/events/{eid}/odds'
                for batch in MARKET_BATCHES:
                    p_params = {
                        'apiKey': ODDS_API_KEY,
                        'regions': 'us',
                        'markets': batch,
                        'bookmakers': PROP_BOOKMAKER,
                        'oddsFormat': 'american'
                    }
                    try:
                        p_resp = requests.get(p_url, params=p_params, timeout=15)
                        if p_resp.status_code != 200:
                            api_errors += 1
                            if api_errors <= 3:
                                print(f"   ⚠️ DK props API {p_resp.status_code} for {eid} [{batch}]: {p_resp.text[:100]}")
                            continue
                        p_data = p_resp.json()
                        for bk in p_data.get('bookmakers', []):
                            if eid == tonight_ids[0]:
                                market_debug = [
                                    f"{mkt.get('key')} ({len(mkt.get('outcomes', []))})"
                                    for mkt in bk.get('markets', [])
                                ]
                                print(f"   🔎 DK sample bookmaker={bk.get('key')} markets: {', '.join(market_debug)}")
                            if bk['key'] != PROP_BOOKMAKER:
                                continue
                            for mkt in bk.get('markets', []):
                                dk_raw_market_keys.add(mkt.get('key'))
                                mn = market_mapping.get(mkt['key'])
                                if not mn:
                                    continue
                                parsed_rows = parse_prop_outcomes_for_market(mkt, mn)
                                if parsed_rows:
                                    dk_parsed_market_keys.add(mkt.get('key'))
                                    prop_list.extend(parsed_rows)
                    except Exception as e:
                        print(f"   ⚠️ Props error for {eid} [{batch}]: {e}")
                        continue
                time.sleep(0.5)

            if tonight_ids:
                parsed_metrics_seen = {row.get('METRIC') for row in prop_list if row.get('METRIC')}
                if {'Batter_SO', 'HR'} - parsed_metrics_seen:
                    probe_url = f'https://api.the-odds-api.com/v4/sports/{PROP_SPORT}/events/{tonight_ids[0]}/odds'
                    probe_params = {
                        'apiKey': ODDS_API_KEY,
                        'regions': 'us',
                        'markets': 'batter_home_runs_alternate,batter_strikeouts_alternate',
                        'bookmakers': 'draftkings,fanduel',
                        'oddsFormat': 'american'
                    }
                    try:
                        probe_resp = requests.get(probe_url, params=probe_params, timeout=15)
                        if probe_resp.status_code == 200:
                            probe_data = probe_resp.json()
                            print("   🔎 Alternate-market probe on first event:")
                            for bk in probe_data.get('bookmakers', []):
                                market_debug = [
                                    f"{mkt.get('key')} ({len(mkt.get('outcomes', []))})"
                                    for mkt in bk.get('markets', [])
                                ]
                                print(f"   🔎 ALT bookmaker={bk.get('key')} markets: {', '.join(market_debug)}")
                        else:
                            print(f"   ⚠️ Alternate-market probe API {probe_resp.status_code}: {probe_resp.text[:120]}")
                    except Exception as e:
                        print(f"   ⚠️ Alternate-market probe failed: {e}")

            df_props = pd.DataFrame(prop_list)
            if not df_props.empty:
                df_props = df_props.dropna(subset=['DK_LINE'])
                df_props['LAST_UPDATED'] = timestamp_est
                remaining = p_resp.headers.get('x-requests-remaining', '?') if 'p_resp' in dir() else '?'
                print(f"   📊 API Quota remaining: {remaining}")
                print(f"   ✅ Fetched {len(df_props)} player props across {df_props['METRIC'].nunique()} markets!")
                print(f"   Batter props: {len(df_props[~df_props['METRIC'].str.startswith('P_')])}")
                print(f"   Pitcher props: {len(df_props[df_props['METRIC'].str.startswith('P_')])}")
                if api_errors:
                    print(f"   ⚠️ DK props API errors: {api_errors}")
                if dk_raw_market_keys != dk_parsed_market_keys:
                    raw_unparsed = sorted(dk_raw_market_keys - dk_parsed_market_keys)
                    if raw_unparsed:
                        print(f"   ⚠️ DK returned but did not fully parse markets: {', '.join(raw_unparsed)}")

                # --- FANDUEL FALLBACK: Fill gaps for missing markets ---
                dk_metrics = set(df_props['METRIC'].unique())
                all_metrics = set(market_mapping.values())
                missing = all_metrics - dk_metrics
                if missing:
                    print(f"\n   🔄 FanDuel fallback for missing markets: {', '.join(sorted(missing))}")
                    fd_list = []
                    fd_raw_market_keys = set()
                    fd_parsed_market_keys = set()
                    fd_api_errors = 0
                    for eid in tonight_ids:
                        p_url = f'https://api.the-odds-api.com/v4/sports/{PROP_SPORT}/events/{eid}/odds'
                        for batch in MARKET_BATCHES:
                            batch_markets = [m for m in batch.split(',') if market_mapping.get(m) in missing]
                            if not batch_markets:
                                continue
                            p_params = {
                                'apiKey': ODDS_API_KEY,
                                'regions': 'us',
                                'markets': ','.join(batch_markets),
                                'bookmakers': 'fanduel',
                                'oddsFormat': 'american'
                            }
                            try:
                                p_resp = requests.get(p_url, params=p_params, timeout=15)
                                if p_resp.status_code != 200:
                                    fd_api_errors += 1
                                    if fd_api_errors <= 3:
                                        print(f"   ⚠️ FD props API {p_resp.status_code} for {eid} [{','.join(batch_markets)}]: {p_resp.text[:100]}")
                                    continue
                                p_data = p_resp.json()
                                for bk in p_data.get('bookmakers', []):
                                    if eid == tonight_ids[0]:
                                        market_debug = [
                                            f"{mkt.get('key')} ({len(mkt.get('outcomes', []))})"
                                            for mkt in bk.get('markets', [])
                                        ]
                                        print(f"   🔎 FD sample bookmaker={bk.get('key')} markets: {', '.join(market_debug)}")
                                    if bk['key'] != 'fanduel':
                                        continue
                                    for mkt in bk.get('markets', []):
                                        fd_raw_market_keys.add(mkt.get('key'))
                                        mn = market_mapping.get(mkt['key'])
                                        if not mn or mn not in missing:
                                            continue
                                        parsed_rows = parse_prop_outcomes_for_market(mkt, mn)
                                        if parsed_rows:
                                            fd_parsed_market_keys.add(mkt.get('key'))
                                            fd_list.extend(parsed_rows)
                            except Exception:
                                continue
                        time.sleep(0.5)

                    if fd_list:
                        df_fd = pd.DataFrame(fd_list).dropna(subset=['DK_LINE'])
                        df_fd['LAST_UPDATED'] = timestamp_est
                        df_props = pd.concat([df_props, df_fd], ignore_index=True)
                        print(f"   ✅ FanDuel added {len(df_fd)} props for: {', '.join(sorted(missing & set(df_fd['METRIC'].unique())))}")
                        print(f"   📊 Total props now: {len(df_props)} across {df_props['METRIC'].nunique()} markets")
                        if fd_api_errors:
                            print(f"   ⚠️ FanDuel props API errors: {fd_api_errors}")
                        raw_unparsed_fd = sorted(fd_raw_market_keys - fd_parsed_market_keys)
                        if raw_unparsed_fd:
                            print(f"   ⚠️ FanDuel returned but did not fully parse markets: {', '.join(raw_unparsed_fd)}")
                    else:
                        if fd_api_errors:
                            print(f"   ⚠️ FanDuel props API errors: {fd_api_errors}")
                        print(f"   ⚠️ Odds API returned no {sorted(missing)} data from FanDuel either (book may not expose these markets via Odds API even when shown in app)")
                else:
                    print(f"   ✅ DK covered all markets — no FanDuel fallback needed")
            else:
                print("   ⚠️ No player props returned.")
    except Exception as e:
        print(f"   ❌ Failed to fetch player props: {e}")
else:
    if len(games_tonight) == 0:
        print("   ⚠️ No games tonight — skipping props.")
    else:
        print("   ⏭️ Props skipped")

# --- 10.6 GEMINI AI PICKS ---
print("\n" + "=" * 60)
existing_daily_picks = load_existing_daily_picks(sh, schedule_date)
seen_pick_keys = set()
if len(existing_daily_picks) > 0:
    for _, row in existing_daily_picks.iterrows():
        key = (
            normalize_player_name(str(row.get('player', ''))),
            str(row.get('prop_type', '')).strip().upper(),
            str(row.get('lean', '')).strip().upper(),
        )
        seen_pick_keys.add(key)
existing_run_numbers = pd.to_numeric(existing_daily_picks.get('RUN_NUMBER', pd.Series(dtype=float)), errors='coerce').dropna().astype(int)
today_run_number = int(existing_run_numbers.max()) + 1 if not existing_run_numbers.empty else 1
def generate_gemini_picks():
    print("\n" + "=" * 60)
    print("🤖 GEMINI AI DAILY PICKS GENERATOR")
    print("=" * 60)

    df_picks = pd.DataFrame()
    if not GEMINI_API_KEY:
        print("⚠️ No Gemini API key — skipping AI picks.")
        return df_picks
    if len(games_tonight) == 0:
        print("⚠️ No games tonight — skipping AI picks.")
        return df_picks

    def implied_prob_american(odds):
        try:
            odds = float(odds)
        except (TypeError, ValueError):
            return None
        if odds > 0:
            return 100.0 / (odds + 100.0)
        if odds < 0:
            return abs(odds) / (abs(odds) + 100.0)
        return None

    def weather_note_for_venue(venue_name):
        wx = next((w for w in weather_rows if w.get('venue_name') == venue_name), {})
        if not wx:
            return "No major weather edge."
        if wx.get('is_dome'):
            return "Dome — no weather factor."
        notes = []
        temp = pd.to_numeric(wx.get('temp_f'), errors='coerce')
        wind = pd.to_numeric(wx.get('wind_mph'), errors='coerce')
        if venue_name == 'Coors Field':
            notes.append('Coors boost')
        if pd.notna(temp) and temp >= 75:
            notes.append('Warm-weather boost')
        if pd.notna(wind) and wind >= 10:
            notes.append(f"Wind {int(wind)} mph")
        if notes:
            return "; ".join(notes)[:40]
        return str(wx.get('description', 'No major weather edge.'))[:40]

    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        games_context = []
        seen_matchups = set()
        for g in games_tonight:
            key = tuple(sorted([g['home_abbr'], g['away_abbr']]))
            if key in seen_matchups:
                continue
            seen_matchups.add(key)
            wx = next((w for w in weather_rows if w.get('venue_name') == g['venue_name']), {})
            games_context.append({
                'matchup': f"{g['away_abbr']} @ {g['home_abbr']}",
                'venue': g['venue_name'],
                'home_pitcher': g['home_pitcher_name'],
                'away_pitcher': g['away_pitcher_name'],
                'home_pitcher_hand': pitcher_map.get(g['away_abbr'], {}).get('opp_pitcher_hand', '?'),
                'away_pitcher_hand': pitcher_map.get(g['home_abbr'], {}).get('opp_pitcher_hand', '?'),
                'temp_f': wx.get('temp_f', 'N/A'),
                'wind_mph': wx.get('wind_mph', 'N/A'),
                'wind_dir': wx.get('wind_dir', 'N/A'),
                'is_dome': wx.get('is_dome', False),
                'condition': wx.get('description', 'N/A'),
            })

        status_cols = [c for c in ['status', 'STATUS', 'injury_status', 'INJURY_STATUS', 'roster_status', 'ROSTER_STATUS'] if c in df_tonight.columns]
        hard_out = {"OUT", "O", "DOUBTFUL", "D", "INACTIVE", "SUSPENDED"}
        top_batters = df_tonight.copy()
        top_batters['_player_norm'] = top_batters['player_name'].map(normalize_player_name)
        print(f"   Total tonight batters: {len(top_batters)}")
        print(f"   Batters before dedupe: {len(top_batters)}")
        top_batters = top_batters[top_batters['_player_norm'] != ""].drop_duplicates(subset=['_player_norm']).copy()
        print(f"   Batters after dedupe: {len(top_batters)}")
        print(f"   Batters before availability filter: {len(top_batters)}")
        if status_cols:
            status_col = status_cols[0]
            top_batters['_status_clean'] = top_batters[status_col].astype(str).str.strip().str.upper()
            top_batters = top_batters[~top_batters['_status_clean'].isin(hard_out)].copy()
        else:
            print("   Availability filter skipped — no batter status column found")
        print(f"   Batters after availability filter: {len(top_batters)}")

        top_pitchers = df_pitcher_tonight.copy()
        if not top_pitchers.empty:
            top_pitchers['_player_norm'] = top_pitchers['player_name'].map(normalize_player_name)
            top_pitchers = top_pitchers[top_pitchers['_player_norm'] != ""].drop_duplicates(subset=['_player_norm']).copy()
        print(f"   Total tonight pitchers: {len(top_pitchers)}")

        df_batter_props = pd.DataFrame()
        df_pitcher_props = pd.DataFrame()
        prop_rows_by_key = {}
        valid_prop_keys = set()

        props_df = df_props.copy() if isinstance(df_props, pd.DataFrame) else pd.DataFrame()
        if not props_df.empty:
            props_df['PLAYER_NORM'] = props_df['PLAYER_NAME'].map(normalize_player_name)
            props_df['PROMPT_METRIC'] = props_df['METRIC'].map(normalize_prop_metric)
            df_batter_props = props_df[
                props_df['PLAYER_NAME'].notna() &
                props_df['METRIC'].notna() &
                ~props_df['METRIC'].astype(str).str.startswith('P_')
            ].copy()
            df_pitcher_props = props_df[
                props_df['PLAYER_NAME'].notna() &
                props_df['METRIC'].notna() &
                props_df['METRIC'].astype(str).str.startswith('P_')
            ].copy()
            for _, row in pd.concat([df_batter_props, df_pitcher_props], ignore_index=True).iterrows():
                key = (row['PLAYER_NORM'], row['PROMPT_METRIC'])
                if key[0] and key[1]:
                    valid_prop_keys.add(key)
                    prop_rows_by_key.setdefault(key, []).append(row)

        batter_prop_names = set(df_batter_props['PLAYER_NORM'].dropna()) if not df_batter_props.empty else set()
        pitcher_prop_names = set(df_pitcher_props['PLAYER_NORM'].dropna()) if not df_pitcher_props.empty else set()
        batter_prop_pool = top_batters[top_batters['_player_norm'].isin(batter_prop_names)].copy() if not top_batters.empty else top_batters
        pitcher_prop_pool = top_pitchers[top_pitchers['_player_norm'].isin(pitcher_prop_names)].copy() if not top_pitchers.empty else top_pitchers
        print(f"   Batters after props-only filtering: {len(batter_prop_pool)}")
        print(f"   Pitchers after props-only filtering: {len(pitcher_prop_pool)}")

        fallback_used = []
        batter_pool = batter_prop_pool.copy()
        if batter_pool.empty and not top_batters.empty:
            batter_pool = top_batters.copy()
            fallback_used.append("batters_stats_fallback")
        pitcher_pool = pitcher_prop_pool.copy()
        if pitcher_pool.empty and not top_pitchers.empty:
            pitcher_pool = top_pitchers.copy()
            fallback_used.append("pitchers_stats_fallback")

        batter_pool['ROLE'] = 'BATTER'
        pitcher_pool['ROLE'] = 'PITCHER'
        combined_prop_pool = pd.concat([batter_pool, pitcher_pool], ignore_index=True, sort=False)
        if 'Seas_UD_FP' in combined_prop_pool.columns:
            combined_prop_pool = combined_prop_pool.sort_values('Seas_UD_FP', ascending=False).copy()
        guaranteed_stars = combined_prop_pool.head(15).copy()
        star_top20_norms = set(combined_prop_pool.head(20)['_player_norm'].dropna().tolist())
        rest_pool = combined_prop_pool[~combined_prop_pool['_player_norm'].isin(guaranteed_stars['_player_norm'])].copy()
        gemini_pool = pd.concat([guaranteed_stars, rest_pool], ignore_index=True, sort=False).drop_duplicates(subset=['_player_norm']).head(80).copy()
        gemini_pool['STAR'] = gemini_pool['_player_norm'].isin(star_top20_norms)
        print(f"   Final players sent to Gemini: {len(gemini_pool)} ({len(batter_pool)} batters, {len(pitcher_pool)} pitchers)")
        print(f"   Fallback used: {', '.join(fallback_used) if fallback_used else 'props + valid stats'}")

        returning_player_map = {
            row.get('_player_norm'): bool(row.get('RETURNING', False))
            for _, row in batter_pool.iterrows()
            if row.get('_player_norm')
        }
        valid_player_map = {
            row['_player_norm']: {
                'player_name': row.get('player_name', ''),
                'role': row.get('ROLE', 'BATTER'),
                'team': row.get('team_abbr', ''),
                'opp': row.get('opp_abbr_tonight', ''),
                'opp_pitcher': row.get('opp_pitcher_name', row.get('opp_starter', 'TBD')),
                'venue': row.get('venue_tonight', ''),
                'lineup_risk_note': row.get('LINEUP_PROTECTION_NOTE', ''),
            }
            for _, row in gemini_pool.iterrows()
            if row.get('_player_norm')
        }

        streak_ctx = ""
        player_streak_map = {}
        try:
            streaks = get_streaks()
            streak_lines = [f"{s['player']} — {s['stat']} streak: {s['streak']} games" for s in streaks if s['streak'] >= 3]
            streak_ctx = "\n".join(streak_lines) if streak_lines else "No active streaks tonight."
            for s in streaks:
                if s.get('streak', 0) >= 3:
                    player_streak_map.setdefault(normalize_player_name(s['player']), []).append(f"{s['stat']} x{s['streak']}")
        except Exception:
            streak_ctx = "Streak data unavailable."

        batter_log_player_norm = df_logs['player_name'].map(normalize_player_name) if not df_logs.empty else pd.Series(dtype=str)
        pitcher_log_player_norm = df_pitcher_logs['player_name'].map(normalize_player_name) if not df_pitcher_logs.empty else pd.Series(dtype=str)
        player_lines = []
        for _, p in gemini_pool.iterrows():
            player_norm = p.get('_player_norm') or normalize_player_name(p.get('player_name'))
            if p.get('ROLE') == 'PITCHER':
                ln = f"{p.get('player_name','?')} [PITCHER] ({p.get('team_abbr','?')} vs {p.get('opp_abbr_tonight','?')})"
                if bool(p.get('STAR', False)):
                    ln += " | STAR"
                ln += f" | Venue={p.get('venue_tonight','?')} | Seas: SO={p.get('Seas_SO','')} ER={p.get('Seas_ER','')} IP={p.get('Seas_IP','')} UD={p.get('Seas_UD_FP','')}"
                ln += f" | L3: SO={p.get('L3_SO','')} ER={p.get('L3_ER','')} IP={p.get('L3_IP','')} UD={p.get('L3_UD_FP','')}"
                ln += f" | L7: SO={p.get('L7_SO','')} ER={p.get('L7_ER','')} IP={p.get('L7_IP','')} UD={p.get('L7_UD_FP','')}"
                if not df_pitcher_props.empty:
                    player_props = df_pitcher_props[df_pitcher_props['PLAYER_NORM'] == player_norm]
                    if not player_props.empty:
                        prop_lines = []
                        signal_lines = []
                        player_logs = df_pitcher_logs[pitcher_log_player_norm == player_norm].copy()
                        for _, prop in player_props.sort_values(['PROMPT_METRIC', 'DK_LINE']).iterrows():
                            over_odds = prop.get('OVER_ODDS')
                            under_odds = prop.get('UNDER_ODDS')
                            odds_bits = []
                            if pd.notna(over_odds):
                                odds_bits.append(f"O {int(over_odds):+d}")
                            if pd.notna(under_odds):
                                odds_bits.append(f"U {int(under_odds):+d}")
                            odds_str = f" ({', '.join(odds_bits)})" if odds_bits else ""
                            prop_lines.append(f"{prop.get('PROMPT_METRIC')} {prop.get('DK_LINE')}{odds_str}")
                            metric = str(prop.get('PROMPT_METRIC', '')).strip().upper()
                            metric_col = 'IP_OUTS' if metric == 'P_OUTS' else metric.replace('P_', '')
                            line_val = pd.to_numeric(prop.get('DK_LINE'), errors='coerce')
                            if metric_col in player_logs.columns and pd.notna(line_val) and len(player_logs) >= 3:
                                vals = pd.to_numeric(player_logs[metric_col], errors='coerce').dropna()
                                if len(vals) >= 3:
                                    if metric in {'P_H', 'P_BB', 'P_ER'}:
                                        hr = (vals < line_val).mean()
                                        sig_label = "U"
                                    else:
                                        hr = (vals > line_val).mean()
                                        sig_label = "O"
                                    ip = implied_prob_american(over_odds if sig_label == 'O' else under_odds)
                                    edge = (hr - ip) * 100 if ip is not None else None
                                    sig = f"{metric} {sig_label}{line_val:g} HR={hr*100:.0f}%"
                                    if edge is not None:
                                        sig += f" EV={edge:.0f}%"
                                    signal_lines.append(sig)
                        if prop_lines:
                            ln += f" | REAL DK props: {', '.join(prop_lines[:6])}"
                        if signal_lines:
                            ln += f" | Best prop signals: {'; '.join(signal_lines[:3])}"
            else:
                ln = f"{p.get('player_name','?')} [BATTER] ({p.get('team_abbr','?')} vs {p.get('opp_abbr_tonight','?')})"
                if bool(p.get('STAR', False)):
                    ln += " | STAR"
                ln += f" | vs {p.get('opp_pitcher_name','TBD')} ({p.get('opp_pitcher_hand','?')}HP)"
                ln += f" | Seas: H={p.get('Seas_H','')} HR={p.get('Seas_HR','')} RBI={p.get('Seas_RBI','')} TB={p.get('Seas_TB','')} R={p.get('Seas_R','')} UD={p.get('Seas_UD_FP','')}"
                ln += f" | L7: H={p.get('L7_H','')} HR={p.get('L7_HR','')} RBI={p.get('L7_RBI','')} TB={p.get('L7_TB','')} UD={p.get('L7_UD_FP','')}"
                if bool(p.get('RETURNING', False)):
                    ln += f" | SAMPLE FLAG: RETURNING (L5 games={int(p.get('L5_GAMES_PLAYED', 0) or 0)}, last7={int(p.get('GAMES_LAST_7D', 0) or 0)})"
                elif bool(p.get('LIMITED_SAMPLE', False)):
                    ln += f" | SAMPLE FLAG: LIMITED_SAMPLE (L5 games={int(p.get('L5_GAMES_PLAYED', 0) or 0)})"
                if bool(p.get('IBB_RISK', False)):
                    ln += f" | RISK FLAG: {p.get('LINEUP_PROTECTION_NOTE', 'LINEUP RISK')}"
                opp_avg = p.get('vs_OPP_AVG', '')
                opp_ops = p.get('vs_OPP_OPS', '')
                opp_hr = p.get('vs_OPP_HR', '')
                if opp_avg:
                    ln += f" | vs {p.get('opp_pitcher_hand','?')}HP: AVG={opp_avg} OPS={opp_ops} HR={opp_hr}"
                loc = p.get('home_away_tonight', '')
                if loc in ('Home', 'Away'):
                    split_bits = []
                    for stat in ['H', 'TB', 'HR', 'RBI', 'UD_FP']:
                        val = p.get(f'{stat}_{loc}')
                        if pd.notna(val):
                            split_bits.append(f"{stat}={val}")
                if split_bits:
                    ln += f" | Tonight {loc} split: {' '.join(split_bits[:5])}"
                ln += f" | Weather edge: {weather_note_for_venue(p.get('venue_tonight'))}"
                streak_bits = player_streak_map.get(player_norm)
                if streak_bits:
                    ln += f" | Streaks: {', '.join(streak_bits[:3])}"
                if not df_batter_props.empty:
                    player_props = df_batter_props[df_batter_props['PLAYER_NORM'] == player_norm]
                    if not player_props.empty:
                        prop_lines = []
                        signal_lines = []
                        player_logs = df_logs[batter_log_player_norm == player_norm].copy()
                        for _, prop in player_props.sort_values(['PROMPT_METRIC', 'DK_LINE']).iterrows():
                            over_odds = prop.get('OVER_ODDS')
                            under_odds = prop.get('UNDER_ODDS')
                            odds_bits = []
                            if pd.notna(over_odds):
                                odds_bits.append(f"O {int(over_odds):+d}")
                            if pd.notna(under_odds):
                                odds_bits.append(f"U {int(under_odds):+d}")
                            odds_str = f" ({', '.join(odds_bits)})" if odds_bits else ""
                            prop_lines.append(f"{prop.get('PROMPT_METRIC')} {prop.get('DK_LINE')}{odds_str}")
                            metric = str(prop.get('PROMPT_METRIC', '')).strip().upper()
                            metric_col = 'SO' if metric == 'SO' else metric
                            line_val = pd.to_numeric(prop.get('DK_LINE'), errors='coerce')
                            if metric_col in player_logs.columns and pd.notna(line_val) and len(player_logs) >= 3:
                                vals = pd.to_numeric(player_logs[metric_col], errors='coerce').dropna()
                                if len(vals) >= 3:
                                    hr = (vals > line_val).mean()
                                    ip = implied_prob_american(over_odds)
                                    edge = (hr - ip) * 100 if ip is not None else None
                                    if bool(p.get('RETURNING', False)) and edge is not None:
                                        edge *= 0.5
                                    sig = f"{metric} {line_val:g} HR={hr*100:.0f}%"
                                    if edge is not None:
                                        sig += f" EV={edge:.0f}%"
                                    signal_lines.append(sig)
                        if prop_lines:
                            ln += f" | REAL DK props: {', '.join(prop_lines[:8])}"
                        if signal_lines:
                            ln += f" | Best prop signals: {'; '.join(signal_lines[:3])}"
            player_lines.append(ln)

        if gemini_pool.empty:
            print("⚠️ No usable Gemini player pool after fallbacks — skipping AI picks.")
            return df_picks

        games_str = json.dumps(games_context, indent=2, default=str)
        player_ctx = '\n'.join(player_lines[:80])
        two_way_notice = ""
        if two_way_tonight:
            names_str = ', '.join(sorted(two_way_tonight))
            two_way_notice = f"""
⚠️  TWO-WAY PLAYER ALERT: {names_str} is a scheduled starting pitcher AND a qualified batter tonight.
Default assumption: they hit when they pitch. However, teams occasionally scratch two-way players from the batting lineup on pitching days.
IF you pick {names_str} for a BATTER prop, you MUST set injury_context to start with "LINEUP RISK:".
"""

        allowed_prompt_props = ['H', 'R', 'P_SO', 'P_ER', 'P_BB']
        print(f"   📋 Gemini available prop types: {', '.join(allowed_prompt_props)}")
        prompt = f"""You are an expert MLB props analyst. Today is {schedule_date}. MLB Regular Season.
{two_way_notice}
TONIGHT'S GAMES (with pitchers and weather):
{games_str}
PLAYER DATA (batters and pitchers, with season/recent averages, matchup context, prop hit-rate/EV signals, and real DK props):
{player_ctx}
ACTIVE PROP STREAKS:
{streak_ctx}
RULES:
- CRITICAL: ONLY pick players from the PLAYER DATA list above.
- Return EXACTLY 14 ranked picks as a JSON array.
- Confidence tiers: SMASH (top 3-4 highest conviction only), STRONG (next 4-5), LEAN (rest).
- STRONG should require multiple confirming signals: positive EV, strong hit rate, and supportive matchup/split context. If only one signal is strong, use LEAN.
- Players flagged RETURNING have depressed lines due to injury/absence. Their season averages are NOT reliable short-term predictors. Treat with extreme caution — do NOT SMASH these players.
- Players flagged with LINEUP RISK / weak lineup protection may be pitched around late. Lower confidence on H/R props when that warning appears unless the edge is overwhelming.
- STAR players are the top 20 by season UD fantasy points in tonight's valid prop pool.
- Prefer at least 4 of your 14 picks to come from STAR players. Non-stars should fill the remaining slots only when they have exceptional edges or matchup context.
- Available prop types: ONLY the real DK props listed next to each player in PLAYER DATA.
- Batters can only get batter props. Pitchers can only get pitcher props (P_SO, P_H, P_BB, P_ER, P_OUTS).
- For batters, only use: H, R.
- Do NOT pick TB, HR, RBI, SB, 2B, P_H, P_OUTS — confirmed losing props by grader data
- Do NOT pick SB, 2B, 1B, or BB at all. Remove them entirely from consideration.
- Do NOT pick H OVER lines above 0.5 — H OVER 1.5 is a consistent losing pick
- Prioritize H OVER 0.5 and P_SO — highest cumulative hit rates (59% and 73% respectively)
- Do NOT pick UD_FP or H+R+RBI — stick to single-stat props.
- DIVERSIFY prop types: max 3 picks of the same prop type per slate.
- Max 3 pitcher props per slate.
- P_SO is the preferred pitcher market. Include at least 1 pitcher strikeouts (P_SO) pick per slate when a strong pitcher matchup exists.
- HR picks: maximum 1 per slate. HR overs are low-probability — only pick when data strongly supports it.
- H props have a 53% hit rate. Prioritize H OVER 0.5 as the core of every slate.
- When edges are close, prioritize H props over every other batter market.
- Prefer at least 8 of your 14 picks to be H props when valid H markets exist.
- Hard cap TB at 2 picks.
- Keep R props secondary to H and RBI. Use R only when matchup, lineup spot, and recent form strongly agree.
- Include a mix of prop types across H, RBI, R, TB, HR, and the strongest pitcher markets when supported by the listed data.
- CRITICAL: Every pick must match one of the listed REAL DK props for that exact player and line.
- Lines must be real sportsbook lines from the listed REAL DK props. Do NOT invent lines or use player averages.
- Use DK lines when available. NEVER return null for line.
ANALYSIS FACTORS:
- Batter vs LHP/RHP splits, recent form, active prop streaks, home/away split, weather, park, and pitcher quality.
- Coors Field, warm weather (75F+), and strong favorable weather notes should boost H/TB overs.
- Pitcher strikeout form, recent IP/outs workload, earned run prevention, matchup offense, and park.
- Prefer props with strong listed hit-rate / EV signals when the market and matchup agree.
For each pick provide:
- rank (1-14)
- player (exact name from data)
- team (abbreviation)
- game (e.g. "NYY @ TOR")
- opponent (abbreviation)
- opp_pitcher (pitcher name for hitters, opposing starter for pitchers)
- prop_type (real DK prop)
- line (real sportsbook line)
- lean (OVER or UNDER)
- confidence (SMASH, STRONG, or LEAN)
- rationale (1 sentence, under 15 words)
- injury_context (under 10 words)
- venue (stadium name)
- weather_note (under 10 words)
OUTPUT FORMAT — Return ONLY a valid JSON array. No markdown, no backticks, no explanation:
[{{"rank":1,"player":"Aaron Judge","team":"NYY","game":"NYY @ TOR","opponent":"TOR","opp_pitcher":"Jose Berrios","prop_type":"H","line":1.5,"lean":"OVER","confidence":"SMASH","rationale":"Elite recent form, favorable split.","injury_context":"Healthy.","venue":"Rogers Centre","weather_note":"Dome"}}]"""

        consensus_pick_lists = []
        consensus_temps = [0.35, 0.55, 0.75]
        for run_idx, temp in enumerate(consensus_temps, start=1):
            gen_config = types.GenerateContentConfig(temperature=temp, max_output_tokens=8192, response_mime_type="application/json")
            print(f"🤖 Calling Gemini API run {run_idx}/3 (temp={temp:.2f})...")
            raw = client.models.generate_content(
                model='gemini-2.5-flash-lite',
                contents=prompt,
                config=gen_config
            ).text.strip()
            try:
                run_picks = parse_gemini_json_array(raw)
                print(f"   ↳ {len(run_picks)} picks returned")
                consensus_pick_lists.append(run_picks)
            except json.JSONDecodeError:
                print(f"   ⚠️ Run {run_idx} returned malformed JSON — ignoring that pass")
        picks_data = build_consensus_pick_pool(consensus_pick_lists)
        consensus_hits = sum(1 for pk in picks_data if int(pk.get('CONSENSUS_COUNT', 1) or 1) >= 2)
        print(f"🤝 Consensus merge: {len(picks_data)} unique picks, {consensus_hits} appearing in 2+ runs")

        picks_before_filter = len(picks_data)
        print(f"   Gemini picks before post-filter: {picks_before_filter}")
        hr_count = 0
        pitcher_pick_count = 0
        prop_type_counts = {}
        hit_count = 0
        run_prop_count = 0
        filtered = []
        dropped_reasons = []
        for pk in picks_data:
            raw_prop = str(pk.get('prop_type', '')).strip()
            player_norm = normalize_player_name(pk.get('player', ''))
            prompt_metric = normalize_prop_metric(raw_prop)
            player_meta = valid_player_map.get(player_norm)
            if not player_meta:
                dropped_reasons.append(f"{pk.get('player')} — player not in Gemini pool")
                continue
            if prompt_metric in {'SB', '2B', '1B', 'BB', 'TB', 'HR', 'RBI', 'P_H', 'P_OUTS'}:
                dropped_reasons.append(f"{pk.get('player')} — {prompt_metric} removed from slate")
                continue
            if prompt_metric == 'H':
                line_num = pd.to_numeric(pk.get('line'), errors='coerce')
                if pd.notna(line_num) and float(line_num) > 0.5:
                    dropped_reasons.append(f"{pk.get('player')} H {line_num:g} — H lines above 0.5 removed")
                    continue
            if prompt_metric == 'HR':
                hr_count += 1
                if hr_count > 1:
                    dropped_reasons.append(f"{pk.get('player')} — extra HR cap")
                    continue
            metric_cap_key = 'Batter_SO' if prompt_metric == 'SO' else prompt_metric
            line_val = pk.get('line')
            if (player_norm, prompt_metric) not in valid_prop_keys:
                dropped_reasons.append(f"{pk.get('player')} {raw_prop} {line_val} — invalid prop for player")
                continue
            line_matches = pd.DataFrame(prop_rows_by_key.get((player_norm, prompt_metric), []))
            if line_matches.empty:
                dropped_reasons.append(f"{pk.get('player')} {raw_prop} — no DK market after normalization")
                continue
            try:
                desired_line = float(line_val)
            except (TypeError, ValueError):
                desired_line = float(line_matches.iloc[0]['DK_LINE'])
            matched_line = line_matches.loc[
                line_matches['DK_LINE'].astype(float).sub(desired_line).abs().idxmin(), 'DK_LINE'
            ]
            pk['line'] = float(matched_line)
            if prompt_metric == 'SO':
                pk['prop_type'] = 'Batter_SO'
                metric_cap_key = 'Batter_SO'
            if metric_cap_key == 'H' and hit_count >= 8:
                dropped_reasons.append(f"{pk.get('player')} H — hit prop cap")
                continue
            if metric_cap_key != 'H' and prop_type_counts.get(metric_cap_key, 0) >= 3:
                dropped_reasons.append(f"{pk.get('player')} {metric_cap_key} — per-type cap")
                continue
            if metric_cap_key == 'TB' and prop_type_counts.get('TB', 0) >= 2:
                dropped_reasons.append(f"{pk.get('player')} TB — TB cap")
                continue
            if metric_cap_key == 'R' and run_prop_count >= 2:
                dropped_reasons.append(f"{pk.get('player')} R — run prop cap")
                continue
            if metric_cap_key.startswith('P_'):
                if pitcher_pick_count >= 3:
                    dropped_reasons.append(f"{pk.get('player')} {metric_cap_key} — pitcher cap")
                    continue
                pitcher_pick_count += 1
            pk['player'] = player_meta['player_name']
            pk['team'] = pk.get('team') or player_meta['team']
            pk['opponent'] = pk.get('opponent') or player_meta['opp']
            pk['opp_pitcher'] = pk.get('opp_pitcher') or player_meta['opp_pitcher']
            pk['venue'] = pk.get('venue') or player_meta['venue']
            pk['game'] = pk.get('game') or f"{player_meta['team']} @ {player_meta['opp']}"
            pk['weather_note'] = pk.get('weather_note') or weather_note_for_venue(player_meta['venue'])
            lineup_risk_note = str(player_meta.get('lineup_risk_note', '') or '').strip()
            if lineup_risk_note:
                existing_ctx = str(pk.get('injury_context', '') or '').strip()
                if existing_ctx:
                    if not existing_ctx.upper().startswith('LINEUP RISK'):
                        pk['injury_context'] = f"{lineup_risk_note}. {existing_ctx}"
                else:
                    pk['injury_context'] = lineup_risk_note
            prop_type_counts[metric_cap_key] = prop_type_counts.get(metric_cap_key, 0) + 1
            if metric_cap_key == 'H':
                hit_count += 1
            if metric_cap_key == 'R':
                run_prop_count += 1
            filtered.append(pk)
        picks_data = filtered

        print(f"   Gemini picks after post-filter: {len(picks_data)}")
        if dropped_reasons:
            print(f"   🚫 Dropped hallucinated/invalid picks: {len(dropped_reasons)}")
            if len(dropped_reasons) <= 25:
                for msg in dropped_reasons:
                    print(f"      - {msg}")
        print("   Gemini pool summary:")
        print(f"      batters after props filter: {len(batter_prop_pool)}")
        print(f"      pitchers after props filter: {len(pitcher_prop_pool)}")
        print(f"      final sent to Gemini: {len(gemini_pool)}")
        print(f"      picks before post-filter: {picks_before_filter}")
        print(f"      picks after post-filter: {len(picks_data)}")
        for i, pk in enumerate(picks_data):
            pk['rank'] = i + 1
        df_picks = pd.DataFrame(picks_data)
        if not df_picks.empty:
            df_picks['confidence'] = df_picks['confidence'].map(normalize_confidence)
            smash_idx = df_picks.index[df_picks['confidence'] == 'SMASH'].tolist()
            max_smash = min(3, max(1, len(df_picks) // 4 + (1 if len(df_picks) >= 8 else 0)))
            for idx in smash_idx[max_smash:]:
                df_picks.at[idx, 'confidence'] = 'STRONG'
            if 'line' in df_picks.columns:
                df_picks['line'] = df_picks['line'].astype(float).apply(lambda x: round(x - 0.5) + 0.5 if x > 0 else 0.5)
                print("   📐 Lines snapped to real prop lines")
            batter_prop_types = {'H', 'HR', 'RBI', 'R', 'TB', 'H+R+RBI', 'UD_FP', '2B', '3B', '1B', 'BB', 'Batter_SO'}
            if two_way_tonight and 'player' in df_picks.columns:
                for i, row in df_picks.iterrows():
                    if row.get('player') in two_way_tonight and row.get('prop_type') in batter_prop_types:
                        existing_ctx = str(row.get('injury_context', '')).strip()
                        if not existing_ctx.upper().startswith('LINEUP RISK'):
                            df_picks.at[i, 'injury_context'] = f"LINEUP RISK: Pitching tonight — confirm batting lineup before bet. {existing_ctx}".strip()
                            print(f"   🚨 Forced LINEUP RISK flag on {row['player']} ({row.get('prop_type')})")
            returning_mask = df_picks['player'].map(lambda n: returning_player_map.get(normalize_player_name(n), False))
            if returning_mask.any():
                df_picks.loc[returning_mask & (df_picks['confidence'] == 'SMASH'), 'confidence'] = 'STRONG'
            df_picks['matchup'] = df_picks['game']
            df_picks['reasoning'] = df_picks['rationale']
            df_picks['DATE'] = schedule_date
            df_picks['RUN_TIME'] = timestamp_est
            df_picks['RUN_NUMBER'] = today_run_number
            df_picks['LAST_UPDATED'] = timestamp_est
            df_picks['RESULT'] = ''
            df_picks['ACTUAL_STAT'] = np.nan
            df_picks['HIT'] = ''
            df_picks['CLV_OPEN_LINE'] = df_picks['line']
            df_picks['CLV_LATEST_LINE'] = df_picks['line']
            df_picks['CLV_DELTA'] = 0.0
            df_picks['CLV_LAST_UPDATE'] = timestamp_est
            df_picks['DATA_SOURCE'] = 'mixed_props_validated'
            df_picks['source'] = df_picks['DATA_SOURCE']
            if 'CONSENSUS_COUNT' not in df_picks.columns:
                df_picks['CONSENSUS_COUNT'] = 1
            if 'CONSENSUS_RUNS' not in df_picks.columns:
                df_picks['CONSENSUS_RUNS'] = '1'
            if 'CONSENSUS_TAG' not in df_picks.columns:
                df_picks['CONSENSUS_TAG'] = ''
            MIN_DAILY_PICKS = 9
            dedup_keep = []
            duplicate_drop_msgs = []
            duplicate_reserve = []
            for _, row in df_picks.iterrows():
                pick_key = (
                    normalize_player_name(row.get('player', '')),
                    str(row.get('prop_type', '')).strip().upper(),
                    str(row.get('lean', '')).strip().upper(),
                )
                if pick_key in seen_pick_keys:
                    duplicate_drop_msgs.append(f"{row.get('player')} {row.get('prop_type')} {row.get('lean')} — duplicate prior run")
                    print(f"   🔁 Skipping duplicate pick: {row.get('player')} {row.get('prop_type')} {row.get('lean')}")
                    duplicate_reserve.append(row.to_dict())
                    continue
                seen_pick_keys.add(pick_key)
                dedup_keep.append(row.to_dict())
            if len(dedup_keep) < MIN_DAILY_PICKS and duplicate_reserve:
                needed = MIN_DAILY_PICKS - len(dedup_keep)
                restored = duplicate_reserve[:needed]
                dedup_keep.extend(restored)
                print(f"   ♻️ Restored {len(restored)} same-day duplicate pick(s) to keep a minimum of {MIN_DAILY_PICKS} picks")
                for restored_row in restored:
                    existing_ctx = str(restored_row.get('injury_context', '') or '').strip()
                    restored_row['injury_context'] = f"RERUN DUPLICATE. {existing_ctx}".strip()
            if duplicate_drop_msgs:
                dropped_reasons.extend(duplicate_drop_msgs)
            df_picks = pd.DataFrame(dedup_keep)
            col_order = ['DATE', 'RUN_NUMBER', 'RUN_TIME', 'rank', 'game', 'matchup', 'player', 'team', 'opponent', 'opp_pitcher', 'prop_type',
                         'line', 'lean', 'confidence', 'rationale', 'reasoning', 'injury_context', 'venue', 'weather_note', 'DATA_SOURCE', 'source',
                         'CONSENSUS_COUNT', 'CONSENSUS_RUNS', 'CONSENSUS_TAG',
                         'CLV_OPEN_LINE', 'CLV_LATEST_LINE', 'CLV_DELTA', 'CLV_LAST_UPDATE',
                         'RESULT', 'ACTUAL_STAT', 'HIT', 'LAST_UPDATED']
            df_picks = df_picks[[c for c in col_order if c in df_picks.columns]]
            if not df_picks.empty:
                df_picks = df_picks.reset_index(drop=True)
                df_picks['rank'] = range(1, len(df_picks) + 1)
            prop_dist = df_picks['prop_type'].fillna('').astype(str).str.upper().value_counts().to_dict()
            lean_series = df_picks['lean'].fillna('').astype(str).str.upper().replace({'FADE': 'UNDER'})
            conf_series = df_picks['confidence'].fillna('').astype(str).str.upper()
            star_ct = int(df_picks['player'].map(lambda n: normalize_player_name(n) in star_top20_norms).sum())
            returning_ct = int(df_picks['player'].map(lambda n: returning_player_map.get(normalize_player_name(n), False)).sum())
            print("📊 Final pick distribution:")
            print(f"   Prop types: {prop_dist}")
            print(f"   Lean: {int((lean_series == 'OVER').sum())} OVER / {int((lean_series == 'UNDER').sum())} UNDER")
            print(f"   Confidence: {int((conf_series == 'SMASH').sum())} SMASH / {int((conf_series == 'STRONG').sum())} STRONG / {int((conf_series == 'LEAN').sum())} LEAN")
            print(f"   Stars: {star_ct}")
            print(f"   Returning: {returning_ct}")
            print(f"   Dropped: {len(dropped_reasons)} — {', '.join(dropped_reasons[:10]) if dropped_reasons else 'none'}")
        print(f"\n✅ Generated {len(df_picks)} picks across {df_picks['game'].nunique() if not df_picks.empty and 'game' in df_picks.columns else 0} games!")
        if not df_picks.empty:
            print(f"🏆 #1 Pick: {df_picks.iloc[0]['player']} — {df_picks.iloc[0]['prop_type']} {df_picks.iloc[0]['lean']} {df_picks.iloc[0]['line']} ({df_picks.iloc[0]['confidence']})")
            smash_ct = len(df_picks[df_picks['confidence'] == 'SMASH'])
            pitcher_ct = int(df_picks['prop_type'].astype(str).str.startswith('P_').sum())
            print(f"💪 {smash_ct} SMASH picks | {pitcher_ct} pitcher props | {len(df_picks) - smash_ct} standard picks")
        else:
            print("⚠️ No validated Gemini picks were produced after post-filtering.")
    except json.JSONDecodeError as e:
        print(f"❌ Failed to parse Gemini response: {e}")
    except Exception as e:
        print(f"❌ AI Picks generation failed: {e}")
    return df_picks

# --- 12. PITCHER GAME LOGS ---
print(f"\nFetching pitcher game logs for {SEASON} season...")
print("⚡ Using parallel fetching...")

df_pitcher_logs = pd.DataFrame()
df_pitcher_tonight = pd.DataFrame()
p_ha_pivot = pd.DataFrame()

def get_qualified_pitchers(season):
    url = f"{MLB_API}/stats?stats=season&group=pitching&sportId=1&season={season}&limit=300"
    resp = requests.get(url).json()
    pitchers_list = []
    days_into_season = (now_est - OPENING_DAY).days if now_est >= OPENING_DAY else 0
    min_ip = 0.1 if days_into_season <= 14 else min(max(days_into_season * 0.3, 1), 50)
    min_ip_outs = max(1, int(math.ceil(min_ip * 3)))
    print(f"   IP threshold: {min_ip} (day {days_into_season} of season)")
    for split in resp.get('stats', [{}])[0].get('splits', []):
        stat = split.get('stat', {})
        player = split.get('player', {})
        team = split.get('team', {})
        ip_outs = innings_to_outs(stat.get('inningsPitched', '0'))
        ip = round(ip_outs / 3, 3)
        if ip_outs >= min_ip_outs:
            pitchers_list.append({
                'player_id': player.get('id'), 'player_name': player.get('fullName', ''),
                'team_id': team.get('id'), 'team_abbr': team_id_to_abbr.get(team.get('id'), ''), 'ip': ip, 'ip_outs': ip_outs})
    return pitchers_list

def get_pitcher_game_log(player_id, season):
    url = f"{MLB_API}/people/{player_id}/stats?stats=gameLog&group=pitching&season={season}&sportId=1"
    try:
        resp = requests.get(url, timeout=10).json()
        splits = resp.get('stats', [{}])[0].get('splits', [])
        games = []
        for split in splits:
            stat = split.get('stat', {})
            opp_id = split.get('opponent', {}).get('id')
            ip_outs = innings_to_outs(stat.get('inningsPitched', '0'))
            games.append({
                'player_id': player_id, 'game_date': split.get('date', ''),
                'team_abbr': team_id_to_abbr.get(split.get('team', {}).get('id'), ''),
                'opp_abbr': team_id_to_abbr.get(opp_id, split.get('opponent', {}).get('abbreviation', '')),
                'home_away': 'Home' if split.get('isHome', True) else 'Away',
                'IP': round(ip_outs / 3, 3), 'IP_DISPLAY': outs_to_ip(ip_outs), 'IP_OUTS': ip_outs,
                'H': int(stat.get('hits', 0)), 'R': int(stat.get('runs', 0)),
                'ER': int(stat.get('earnedRuns', 0)), 'HR': int(stat.get('homeRuns', 0)),
                'BB': int(stat.get('baseOnBalls', 0)), 'SO': int(stat.get('strikeOuts', 0)),
                'W': 1 if stat.get('wins', 0) else 0, 'L': 1 if stat.get('losses', 0) else 0,
                'PC': int(stat.get('pitchesThrown', 0) or stat.get('numberOfPitches', 0) or 0),
                'GS': 1 if stat.get('gamesStarted', 0) else 0,
                'HBP': int(stat.get('hitByPitch', 0)), 'ERA': float(stat.get('era', '0') or '0'),
            })
        return games
    except Exception:
        return []

PITCHER_LOG_BASE_COLS = ['player_id', 'player_name', 'game_date', 'team_abbr', 'opp_abbr', 'home_away',
                         'IP', 'IP_DISPLAY', 'IP_OUTS', 'H', 'R', 'ER', 'HR', 'BB', 'SO', 'W', 'L', 'PC', 'GS', 'HBP', 'ERA']
PITCHER_LOG_NUMERIC_COLS = ['player_id', 'IP', 'IP_DISPLAY', 'IP_OUTS', 'H', 'R', 'ER', 'HR', 'BB', 'SO', 'W', 'L', 'PC', 'GS', 'HBP', 'ERA']
existing_pitcher_logs = load_existing_log_sheet('Pitcher_Game_Logs', PITCHER_LOG_BASE_COLS, PITCHER_LOG_NUMERIC_COLS)
latest_pitcher_date_by_pid = {}
if len(existing_pitcher_logs) > 0:
    latest_pitcher_date_by_pid = existing_pitcher_logs.dropna(subset=['player_id']).groupby('player_id')['game_date'].max().to_dict()
    latest_pitcher_seed_date = max(latest_pitcher_date_by_pid.values()) if latest_pitcher_date_by_pid else ''
    if latest_pitcher_seed_date:
        print(f"♻️ Seeded Pitcher_Game_Logs through {latest_pitcher_seed_date} ({len(existing_pitcher_logs)} existing rows)")
else:
    print("🆕 No existing Pitcher_Game_Logs seed found — full pitcher fetch")

qualified_pitchers = get_qualified_pitchers(SEASON)
print(f"✅ Found {len(qualified_pitchers)} qualified pitchers")

def fetch_one_pitcher_log(pitcher):
    logs = get_pitcher_game_log(pitcher['player_id'], SEASON)
    cutoff = latest_pitcher_date_by_pid.get(pitcher['player_id'])
    if cutoff:
        logs = [log for log in logs if normalize_game_date(log.get('game_date')) > cutoff]
    for log in logs:
        log['player_name'] = pitcher['player_name']
        log['game_date'] = normalize_game_date(log['game_date'])
    return logs

all_pitcher_logs = []
start_time = time.time()
with ThreadPoolExecutor(max_workers=15) as executor:
    futures = {executor.submit(fetch_one_pitcher_log, p): p for p in qualified_pitchers}
    done_count = 0
    for future in as_completed(futures):
        all_pitcher_logs.extend(future.result())
        done_count += 1
        if done_count % 50 == 0:
            print(f"   Fetched {done_count}/{len(qualified_pitchers)} pitchers...")

elapsed = time.time() - start_time
new_pitcher_logs = pd.DataFrame(all_pitcher_logs, columns=PITCHER_LOG_BASE_COLS)
combined_pitcher_logs = pd.concat([existing_pitcher_logs, new_pitcher_logs], ignore_index=True)
if len(combined_pitcher_logs) > 0:
    combined_pitcher_logs['game_date'] = combined_pitcher_logs['game_date'].map(normalize_game_date)
    dedupe_cols = ['player_id', 'game_date', 'opp_abbr', 'home_away', 'IP_OUTS', 'H', 'R', 'ER', 'HR', 'BB', 'SO', 'W', 'L', 'PC', 'GS', 'HBP']
    combined_pitcher_logs = combined_pitcher_logs.drop_duplicates(subset=dedupe_cols, keep='last')
    combined_pitcher_logs['game_date'] = pd.to_datetime(combined_pitcher_logs['game_date'], errors='coerce')
    df_pitcher_logs = combined_pitcher_logs.sort_values(['player_id', 'game_date']).reset_index(drop=True)
    print(f"✅ Fetched {len(new_pitcher_logs)} new pitcher logs; {len(df_pitcher_logs)} combined pitcher logs across {df_pitcher_logs['player_name'].nunique()} pitchers in {elapsed:.1f}s")
else:
    df_pitcher_logs = combined_pitcher_logs
    print("⚠️ No pitcher game logs found")

# --- 13. PITCHER METRICS ---
if len(df_pitcher_logs) > 0:
    print("\nCalculating pitcher metrics and rolling averages...")
    if 'IP_OUTS' not in df_pitcher_logs.columns:
        df_pitcher_logs['IP_OUTS'] = df_pitcher_logs['IP'].apply(innings_to_outs)
    df_pitcher_logs['DK_FP'] = (
        df_pitcher_logs['IP'] * 2.25 + df_pitcher_logs['SO'] * 2 + df_pitcher_logs['W'] * 4 +
        df_pitcher_logs['ER'] * -2 + df_pitcher_logs['H'] * -0.6 + df_pitcher_logs['BB'] * -0.6 +
        df_pitcher_logs['HBP'] * -0.6 +
        (df_pitcher_logs['IP_OUTS'] >= 18).astype(int) * (df_pitcher_logs['ER'] <= 3).astype(int) * 2.5).round(2)
    df_pitcher_logs['CG'] = (df_pitcher_logs['IP_OUTS'] >= 27).astype(int)

    # v1.3.0: Quality Start column + Underdog Fantasy scoring
    df_pitcher_logs['QS'] = ((df_pitcher_logs['IP_OUTS'] >= 18) & (df_pitcher_logs['ER'] <= 3)).astype(int)

    # Underdog: W=5, QS=5, K=3, IP=3, ER=-3
    df_pitcher_logs['UD_FP'] = (
        df_pitcher_logs['W'] * 5 + df_pitcher_logs['QS'] * 5 +
        df_pitcher_logs['SO'] * 3 + df_pitcher_logs['IP'] * 3 +
        df_pitcher_logs['ER'] * -3
    ).round(2)

    # v1.3.0: Added QS, UD_FP to rolling averages
    p_metrics = ['IP', 'H', 'ER', 'HR', 'BB', 'SO', 'PC', 'DK_FP', 'UD_FP', 'QS', 'R', 'W', 'L']
    p_windows = {'L3': 3, 'L7': 7, 'L15': 15}

    df_pitcher_logs = df_pitcher_logs.set_index('game_date').sort_index()
    for m in p_metrics:
        grp = df_pitcher_logs.groupby('player_id')[m]
        df_pitcher_logs[f'Seas_{m}'] = grp.transform(lambda x: x.expanding().mean()).round(3)
        for label, w in p_windows.items():
            df_pitcher_logs[f'{label}_{m}'] = grp.transform(lambda x: x.rolling(w, min_periods=1).mean()).round(3)

    for label, w in p_windows.items():
        roll_er = df_pitcher_logs.groupby('player_id')['ER'].transform(lambda x: x.rolling(w, min_periods=1).sum())
        roll_ip = df_pitcher_logs.groupby('player_id')['IP'].transform(lambda x: x.rolling(w, min_periods=1).sum())
        df_pitcher_logs[f'{label}_ERA'] = np.where(roll_ip > 0, (roll_er * 9 / roll_ip).round(2), 0)

    seas_er = df_pitcher_logs.groupby('player_id')['ER'].transform(lambda x: x.expanding().sum())
    seas_ip = df_pitcher_logs.groupby('player_id')['IP'].transform(lambda x: x.expanding().sum())
    df_pitcher_logs['Seas_ERA'] = np.where(seas_ip > 0, (seas_er * 9 / seas_ip).round(2), 0)

    for label, w in p_windows.items():
        roll_h = df_pitcher_logs.groupby('player_id')['H'].transform(lambda x: x.rolling(w, min_periods=1).sum())
        roll_bb = df_pitcher_logs.groupby('player_id')['BB'].transform(lambda x: x.rolling(w, min_periods=1).sum())
        roll_ip = df_pitcher_logs.groupby('player_id')['IP'].transform(lambda x: x.rolling(w, min_periods=1).sum())
        df_pitcher_logs[f'{label}_WHIP'] = np.where(roll_ip > 0, ((roll_h + roll_bb) / roll_ip).round(2), 0)

    seas_h = df_pitcher_logs.groupby('player_id')['H'].transform(lambda x: x.expanding().sum())
    seas_bb = df_pitcher_logs.groupby('player_id')['BB'].transform(lambda x: x.expanding().sum())
    df_pitcher_logs['Seas_WHIP'] = np.where(seas_ip > 0, ((seas_h + seas_bb) / seas_ip).round(2), 0)

    for label, w in p_windows.items():
        roll_so = df_pitcher_logs.groupby('player_id')['SO'].transform(lambda x: x.rolling(w, min_periods=1).sum())
        roll_ip = df_pitcher_logs.groupby('player_id')['IP'].transform(lambda x: x.rolling(w, min_periods=1).sum())
        df_pitcher_logs[f'{label}_K9'] = np.where(roll_ip > 0, (roll_so * 9 / roll_ip).round(2), 0)

    seas_so = df_pitcher_logs.groupby('player_id')['SO'].transform(lambda x: x.expanding().sum())
    df_pitcher_logs['Seas_K9'] = np.where(seas_ip > 0, (seas_so * 9 / seas_ip).round(2), 0)

    df_pitcher_logs = df_pitcher_logs.reset_index()
    df_pitcher_logs['game_date'] = df_pitcher_logs['game_date'].dt.strftime('%Y-%m-%d')
    df_pitcher_logs['LAST_UPDATED'] = timestamp_est
    print(f"✅ Pitcher metrics calculated — {len(df_pitcher_logs.columns)} columns total")
    print(f"   📊 New columns: QS, UD_FP, Seas_UD_FP, L3_UD_FP, L7_UD_FP, Seas_QS, L3_QS, L7_QS")

# --- 14. PITCHER HOME/AWAY SPLITS ---
if len(df_pitcher_logs) > 0:
    print("\nCalculating pitcher Home/Away splits...")
    p_split_metrics = ['IP', 'H', 'ER', 'HR', 'BB', 'SO', 'DK_FP', 'UD_FP', 'PC', 'R']
    df_plogs_temp = df_pitcher_logs.copy()

    p_ha_mean = df_plogs_temp.groupby(['player_id', 'home_away'])[p_split_metrics].mean().round(3)
    p_ha_count = df_plogs_temp.groupby(['player_id', 'home_away'])['IP'].count().rename('GAMES')
    df_p_home_away = p_ha_mean.join(p_ha_count).reset_index()

    p_ha_pivot = df_p_home_away.pivot(index='player_id', columns='home_away', values=p_split_metrics)
    p_ha_pivot.columns = [f'{stat}_{loc}' for stat, loc in p_ha_pivot.columns]
    p_ha_count_pivot = df_p_home_away.pivot(index='player_id', columns='home_away', values='GAMES')
    p_ha_count_pivot.columns = [f'{c}_GAMES' for c in p_ha_count_pivot.columns]
    p_ha_pivot = p_ha_pivot.join(p_ha_count_pivot)

    for m in p_split_metrics:
        for loc in ['Home', 'Away']:
            col = f'{m}_{loc}'
            if col not in p_ha_pivot.columns:
                p_ha_pivot[col] = np.nan
    for loc in ['Home', 'Away']:
        gcol = f'{loc}_GAMES'
        if gcol not in p_ha_pivot.columns:
            p_ha_pivot[gcol] = np.nan

    for m in p_split_metrics:
        hc, ac = f'{m}_Home', f'{m}_Away'
        if hc in p_ha_pivot.columns and ac in p_ha_pivot.columns:
            p_ha_pivot[f'{m}_SPLIT_DIFF'] = (p_ha_pivot[hc] - p_ha_pivot[ac]).where(
                p_ha_pivot[hc].notna() & p_ha_pivot[ac].notna(), other=np.nan).round(3)

    p_names = df_plogs_temp.groupby('player_id')['player_name'].first()
    p_ha_pivot = p_ha_pivot.reset_index().merge(p_names, on='player_id', how='left')
    p_ha_pivot = p_ha_pivot.reindex(sorted(p_ha_pivot.columns), axis=1)
    cols = ['player_id', 'player_name'] + [c for c in p_ha_pivot.columns if c not in ['player_id', 'player_name']]
    p_ha_pivot = p_ha_pivot[cols]
    p_ha_pivot['LAST_UPDATED'] = timestamp_est
    print(f"✅ Pitcher Home/Away splits for {p_ha_pivot['player_name'].nunique()} pitchers")

# --- 15. TONIGHT'S PITCHER SHEET (EARLY-SEASON SAFE) ---
if len(games_tonight) > 0:
    print("\nBuilding tonight's pitcher stats sheet...")
    pitcher_current_team = {p['player_id']: p['team_abbr'] for p in qualified_pitchers}

    tonight_sp_ids = set()
    tonight_sp_info = {}
    for g in games_tonight:
        if g.get('home_pitcher_id'):
            tonight_sp_ids.add(g['home_pitcher_id'])
            tonight_sp_info[g['home_pitcher_id']] = {'player_name': g['home_pitcher_name'], 'team_abbr': g['home_abbr']}
        if g.get('away_pitcher_id'):
            tonight_sp_ids.add(g['away_pitcher_id'])
            tonight_sp_info[g['away_pitcher_id']] = {'player_name': g['away_pitcher_name'], 'team_abbr': g['away_abbr']}

    if len(df_pitcher_logs) > 0:
        p_most_recent = df_pitcher_logs.sort_values('game_date').groupby('player_id').last().reset_index()
        p_most_recent['team_abbr'] = p_most_recent['player_id'].map(pitcher_current_team)
        p_most_recent['player_name'] = p_most_recent['player_id'].map(
            {p['player_id']: p['player_name'] for p in qualified_pitchers})
        df_pitcher_tonight = p_most_recent[p_most_recent['player_id'].isin(tonight_sp_ids)].copy()
    else:
        df_pitcher_tonight = pd.DataFrame()

    existing_sp_ids = set(df_pitcher_tonight['player_id'].unique()) if len(df_pitcher_tonight) > 0 else set()
    missing_sp_ids = tonight_sp_ids - existing_sp_ids
    if missing_sp_ids:
        print(f"   🔄 Early-season expansion: adding {len(missing_sp_ids)} starter(s) with no game logs...")
        expansion_rows = []
        for pid in missing_sp_ids:
            info = tonight_sp_info.get(pid, {})
            expansion_rows.append({'player_id': pid, 'player_name': info.get('player_name', 'Unknown'), 'team_abbr': info.get('team_abbr', '')})

        df_sp_expansion = pd.DataFrame(expansion_rows)
        if len(df_pitcher_tonight) > 0:
            for col in df_pitcher_tonight.columns:
                if col not in df_sp_expansion.columns:
                    df_sp_expansion[col] = np.nan
            df_pitcher_tonight = pd.concat([df_pitcher_tonight, df_sp_expansion[df_pitcher_tonight.columns]], ignore_index=True)
        else:
            df_pitcher_tonight = df_sp_expansion.copy()
        for pid in missing_sp_ids:
            info = tonight_sp_info.get(pid, {})
            print(f"      + {info.get('player_name', pid)} ({info.get('team_abbr', '?')})")

    pitcher_to_game = {}
    for g in games_tonight:
        if g.get('home_pitcher_id'):
            pitcher_to_game[g['home_pitcher_id']] = {'opp_abbr': g['away_abbr'], 'venue': g['venue_name'], 'home_away': 'Home', 'opp_pitcher': g['away_pitcher_name']}
        if g.get('away_pitcher_id'):
            pitcher_to_game[g['away_pitcher_id']] = {'opp_abbr': g['home_abbr'], 'venue': g['venue_name'], 'home_away': 'Away', 'opp_pitcher': g['home_pitcher_name']}

    df_pitcher_tonight['opp_abbr_tonight'] = df_pitcher_tonight['player_id'].map({k: v['opp_abbr'] for k, v in pitcher_to_game.items()})
    df_pitcher_tonight['venue_tonight'] = df_pitcher_tonight['player_id'].map({k: v['venue'] for k, v in pitcher_to_game.items()})
    df_pitcher_tonight['home_away_tonight'] = df_pitcher_tonight['player_id'].map({k: v['home_away'] for k, v in pitcher_to_game.items()})
    df_pitcher_tonight['opp_starter'] = df_pitcher_tonight['player_id'].map({k: v['opp_pitcher'] for k, v in pitcher_to_game.items()})

    p_rolling_cols = [c for c in df_pitcher_tonight.columns if any(c.startswith(p) for p in ['L3_', 'L7_', 'L15_', 'Seas_'])]
    p_final_cols = ['player_name', 'team_abbr', 'opp_abbr_tonight', 'venue_tonight', 'home_away_tonight', 'opp_starter'] + p_rolling_cols + ['LAST_UPDATED']
    p_final_cols = [c for c in p_final_cols if c in df_pitcher_tonight.columns]
    df_pitcher_tonight = df_pitcher_tonight[p_final_cols].copy()
    df_pitcher_tonight = df_pitcher_tonight.sort_values('player_name').reset_index(drop=True)

    has_logs = df_pitcher_tonight[p_rolling_cols[0]].notna().sum() if p_rolling_cols else 0
    no_logs = len(df_pitcher_tonight) - has_logs
    print(f"✅ Tonight's pitcher sheet — {len(df_pitcher_tonight)} starting pitchers")
    print(f"   ({has_logs} with game logs, {no_logs} roster-only)")
    print(f"   Columns: {len(df_pitcher_tonight.columns)}")
else:
    df_pitcher_tonight = pd.DataFrame()
    print("⚠️ No games tonight — skipping tonight's pitcher sheet")

# --- 10.75 GEMINI AI DAILY PICKS GENERATOR ---
df_picks = generate_gemini_picks()

# --- 11. WRITE ALL DATA TO GOOGLE SHEETS ---
print("\n" + "=" * 60)
print("WRITING ALL DATA TO GOOGLE SHEETS")
print("=" * 60)

def safe_upload(spreadsheet, sheet_name, df, max_retries=3):
    if df is None or len(df) == 0:
        print(f"   ⏭️  {sheet_name}: No data — skipped")
        return
    df_clean = df.copy()
    df_clean = df_clean.replace([np.inf, -np.inf], None)
    df_clean = df_clean.fillna('')
    for col in df_clean.columns:
        df_clean[col] = df_clean[col].apply(lambda x: x.item() if hasattr(x, 'item') else x)
    for attempt in range(max_retries):
        try:
            try:
                ws = spreadsheet.worksheet(sheet_name)
                ws.clear()
            except gspread.exceptions.WorksheetNotFound:
                ws = spreadsheet.add_worksheet(title=sheet_name, rows=max(len(df_clean)+1, 100), cols=max(len(df_clean.columns), 26))
            needed_rows = len(df_clean) + 1
            needed_cols = len(df_clean.columns)
            if ws.row_count < needed_rows or ws.col_count < needed_cols:
                ws.resize(rows=max(needed_rows, ws.row_count), cols=max(needed_cols, ws.col_count))
            set_with_dataframe(ws, df_clean)
            print(f"   ✅ {sheet_name}: {len(df_clean)} rows × {len(df_clean.columns)} cols")
            return
        except gspread.exceptions.APIError as e:
            if 'RATE_LIMIT' in str(e) or '429' in str(e):
                wait = 30 * (attempt + 1)
                print(f"   ⏳ Rate limited on {sheet_name} — waiting {wait}s (attempt {attempt+1}/{max_retries})")
                time.sleep(wait)
            else:
                print(f"   ❌ {sheet_name}: {e}")
                return
        except Exception as e:
            print(f"   ❌ {sheet_name}: {e}")
            return
    print(f"   ❌ {sheet_name}: Failed after {max_retries} retries")

SHEETS_TO_WRITE = {
    'Batter_Game_Logs': df_logs,
    'Tonights_Batters': df_tonight,
    'LHP_RHP_Splits': df_splits,
    'Home_Away_Splits': ha_pivot,
    'Tonights_Schedule': df_schedule,
    'Tonights_Pitchers': df_pitchers,
    'Venue_Weather': df_weather,
    'Odds': df_odds if len(df_odds) > 0 else pd.DataFrame(),
    'DK_Player_Props': df_props if len(df_props) > 0 else pd.DataFrame(),
    'Teams': df_teams,
    'Pitcher_Game_Logs': df_pitcher_logs if len(df_pitcher_logs) > 0 else pd.DataFrame(),
    'Tonights_Starters': df_pitcher_tonight if len(df_pitcher_tonight) > 0 else pd.DataFrame(),
    'Pitcher_Home_Away': p_ha_pivot if len(p_ha_pivot) > 0 else pd.DataFrame(),
    'Batter_vs_SP': df_vs_sp if len(df_vs_sp) > 0 else pd.DataFrame(),
}

print(f"\nWriting {len(SHEETS_TO_WRITE)} sheets to '{SHEET_NAME}'...\n")
for sheet_name, df in SHEETS_TO_WRITE.items():
    safe_upload(sh, sheet_name, df)
    time.sleep(2)
# --- DAILY PICKS: APPEND-ONLY (preserves history) ---
if df_picks is not None and len(df_picks) > 0:
    print(f"\n📌 Appending today's picks to Daily_Picks...")
    try:
        try:
            ws_picks = sh.worksheet('Daily_Picks')
            existing = ws_picks.get_all_records()
            df_existing = pd.DataFrame(existing)
            print(f"   📋 Existing picks: {len(df_existing)} rows")
        except gspread.exceptions.WorksheetNotFound:
            df_existing = pd.DataFrame()
            ws_picks = sh.add_worksheet(title='Daily_Picks', rows=500, cols=26)
            print(f"   🆕 Created Daily_Picks sheet")

        df_existing = refresh_clv_frame(df_existing, schedule_date, df_props, timestamp_est)

        # Stack old + new
        df_all_picks = pd.concat([df_existing, df_picks], ignore_index=True)
        df_all_picks = df_all_picks.fillna('')
        for col in df_all_picks.columns:
            df_all_picks[col] = df_all_picks[col].apply(lambda x: x.item() if hasattr(x, 'item') else x)

        ws_picks.clear()
        from gspread_dataframe import set_with_dataframe
        needed_rows = len(df_all_picks) + 1
        needed_cols = len(df_all_picks.columns)
        if ws_picks.row_count < needed_rows or ws_picks.col_count < needed_cols:
            ws_picks.resize(rows=max(needed_rows, ws_picks.row_count), cols=max(needed_cols, ws_picks.col_count))
        set_with_dataframe(ws_picks, df_all_picks)
        print(f"   ✅ Daily_Picks: {len(df_all_picks)} total rows ({len(df_existing)} old + {len(df_picks)} new)")
    except Exception as e:
        print(f"   ❌ Daily_Picks append failed: {e}")
    time.sleep(2)
else:
    print(f"\n⏭️  Daily_Picks: No new picks to append")

# --- SUMMARY ---
print(f"\n{'='*60}")
print(f"✅ MLB DASHBOARD ENGINE v1.3.0 — COMPLETE")
print(f"{'='*60}")
print(f"📅 Schedule date:    {schedule_date}")
print(f"📆 Season:           {SEASON}")
print(f"🗂️  Snapshot:         {SNAPSHOT_DATE}")
print(f"⚾ Games tonight:    {len(games_tonight)}")
print(f"🏏 Active batters:   {len(df_tonight)}")
print(f"⚾ Pitchers tracked: {len(df_pitcher_logs) if len(df_pitcher_logs) > 0 else 0}")
print(f"🎯 Tonight's SPs:    {len(df_pitcher_tonight) if len(df_pitcher_tonight) > 0 else 0}")
print(f"📊 Batter game logs: {len(df_logs)}")
print(f"📊 Pitcher game logs:{len(df_pitcher_logs) if len(df_pitcher_logs) > 0 else 0}")
print(f"🌤️  Venues w/ weather: {len(df_weather)}")
print(f"🎰 Odds:             {'✅ ' + str(len(df_odds)) + ' games' if len(df_odds) > 0 else 'Skipped'}")
if len(df_props) > 0:
    print(f"🎲 DK Props:         {len(df_props)} props across {df_props['METRIC'].nunique()} markets")
else:
    print(f"🎲 DK Props:         Skipped")
if len(df_picks) > 0:
    print(f"🤖 AI Picks:         {len(df_picks)} picks (Top: {df_picks.iloc[0]['player']} {df_picks.iloc[0]['confidence']})")
    if 'two_way_tonight' in dir() and two_way_tonight:
        print(f"⚠️  Two-way players: {', '.join(sorted(two_way_tonight))}")
else:
    print(f"🤖 AI Picks:         Skipped")
print(f"📝 Google Sheet:     {SHEET_NAME}")
print(f"🕐 Last updated:     {timestamp_est}")
print(f"🆕 v1.3.0: Underdog FP (UD_FP), QS, HBP, 1B/3B rolling averages added")
print(f"{'='*60}")

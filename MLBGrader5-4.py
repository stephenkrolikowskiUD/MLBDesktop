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
PICK_PERF_MIN_SAMPLE = 25
PICK_PERF_STANDARD_ODDS = -115
PICK_PERF_WILSON_Z = 1.96
PICK_PERF_DRIFT_ALERT_PP = 10
PICK_PERF_TIME_WINDOWS = {
    'last_7d': 7,
    'last_30d': 30,
    'last_90d': 90,
    'all_time': None,
}
PICK_PERF_SNAPSHOT_WINDOWS = ('all_time', 'last_30d')
PICK_PERF_DIMENSIONS = (
    'confidence_norm',
    'selection_method_norm',
    'recommendation_status_norm',
    'prop_type_norm',
    'lean_norm',
    'consensus_bucket',
    'clv_bucket',
    'has_lineup_risk',
    'day_of_week',
    'RUN_NUMBER',
)

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

PICK_PERFORMANCE_COLUMNS = [
    'DIMENSION_TYPE', 'DIMENSION_VALUE', 'TIME_WINDOW',
    'N_PICKS', 'N_PICKS_DECISIVE', 'N_HITS', 'N_MISSES', 'N_PUSHES', 'N_DNP',
    'HIT_RATE', 'HIT_RATE_RAW', 'PUSH_RATE', 'DNP_RATE',
    'ROI_FLAT', 'ROI_PER_PICK',
    'AVG_CLV_EDGE', 'CLV_POSITIVE_RATE', 'CLV_POS_HIT_RATE', 'CLV_NEG_HIT_RATE',
    'WILSON_LOWER_95', 'MIN_SAMPLE_FLAG',
    'LAST_UPDATED',
]
PICK_PERFORMANCE_SNAPSHOT_COLUMNS = ['SNAPSHOT_DATE', 'METRIC_KEY', 'METRIC_VALUE', 'N_PICKS', 'TIME_WINDOW']

def normalize_prop_metric(metric):
    text = str(metric or '').strip().upper()
    text = re.sub(r"\s+", "", text)
    if text == 'BATTER_SO':
        return 'SO'
    return text

def normalize_confidence(val):
    conf = str(val or '').strip().upper()
    return conf if conf in {'SMASH', 'STRONG', 'LEAN'} else 'LEAN'

def pick_perf_clean_cell(val):
    if hasattr(val, 'item'):
        val = val.item()
    if val is None:
        return ''
    if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
        return ''
    return val

def pick_perf_safe_upload(spreadsheet, sheet_name, df):
    if df is None or df.empty:
        print(f"   ⏭️  {sheet_name}: No data — skipped")
        return False
    df_clean = df.copy().replace([np.inf, -np.inf], np.nan).fillna('')
    values = [df_clean.columns.tolist()] + [
        [pick_perf_clean_cell(v) for v in row]
        for row in df_clean.values.tolist()
    ]
    try:
        try:
            ws_out = spreadsheet.worksheet(sheet_name)
            ws_out.clear()
        except gspread.exceptions.WorksheetNotFound:
            ws_out = spreadsheet.add_worksheet(title=sheet_name, rows=max(len(values), 100), cols=max(len(df_clean.columns), 26))
        if ws_out.row_count < len(values) or ws_out.col_count < len(df_clean.columns):
            ws_out.resize(rows=max(len(values), ws_out.row_count), cols=max(len(df_clean.columns), ws_out.col_count))
        ws_out.update(values, value_input_option='RAW')
        print(f"   ✅ {sheet_name}: {len(df_clean)} rows × {len(df_clean.columns)} cols")
        return True
    except Exception as e:
        print(f"   ❌ {sheet_name}: {e}")
        return False

def pick_perf_append_upload(spreadsheet, sheet_name, df):
    if df is None or df.empty:
        print(f"   ⏭️  {sheet_name}: No snapshot rows — skipped")
        return False
    df_clean = df.copy().replace([np.inf, -np.inf], np.nan).fillna('')
    rows = [[pick_perf_clean_cell(v) for v in row] for row in df_clean.values.tolist()]
    try:
        try:
            ws_out = spreadsheet.worksheet(sheet_name)
            existing = ws_out.get_all_values()
        except gspread.exceptions.WorksheetNotFound:
            ws_out = spreadsheet.add_worksheet(title=sheet_name, rows=max(len(rows) + 1, 100), cols=max(len(df_clean.columns), 26))
            existing = []
        if not existing:
            ws_out.update([df_clean.columns.tolist()], value_input_option='RAW')
        if ws_out.col_count < len(df_clean.columns):
            ws_out.resize(rows=ws_out.row_count, cols=len(df_clean.columns))
        ws_out.append_rows(rows, value_input_option='RAW')
        print(f"   ✅ {sheet_name}: appended {len(rows)} rows")
        return True
    except Exception as e:
        print(f"   ❌ {sheet_name}: {e}")
        return False

def wilson_lower_bound(p, n, z=PICK_PERF_WILSON_Z):
    if n <= 0:
        return 0.0
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n)
    return max(0.0, (centre - margin) / denom)

def pick_perf_rate(hits, misses):
    denom = hits + misses
    return hits / denom if denom > 0 else np.nan

def pick_perf_prepare_df(df_all):
    if df_all is None or df_all.empty or 'HIT' not in df_all.columns:
        return pd.DataFrame()
    df = df_all[df_all['HIT'].isin(['YES', 'NO', 'PUSH', 'DNP'])].copy()
    if df.empty:
        return df
    idx = df.index
    df['player_norm'] = df.get('player', pd.Series('', index=idx)).map(normalize_person_name)
    df['prop_type_norm'] = df.get('prop_type', pd.Series('', index=idx)).map(normalize_prop_metric)
    df['lean_norm'] = df.get('lean', pd.Series('', index=idx)).fillna('').astype(str).str.upper().replace({'FADE': 'UNDER'})
    explicit_method = df.get('SELECTION_METHOD', pd.Series('', index=idx)).fillna('').astype(str).str.strip().str.upper()
    consensus_tag = df.get('CONSENSUS_TAG', pd.Series('', index=idx)).fillna('').astype(str).str.strip().str.upper()
    df['selection_method_norm'] = np.where(
        explicit_method.ne(''),
        explicit_method,
        np.where(consensus_tag.eq('VALIDATED FALLBACK'), 'VALIDATED_MODEL', 'GEMINI'),
    )
    raw_status = df.get('RECOMMENDATION_STATUS', pd.Series('', index=idx)).fillna('').astype(str).str.strip().str.upper()
    df['recommendation_status_norm'] = np.where(raw_status.ne(''), raw_status, 'LEGACY_RESEARCH')
    df['confidence_norm'] = df.get('confidence', pd.Series('', index=idx)).map(normalize_confidence)
    df.loc[df['selection_method_norm'].eq('VALIDATED_MODEL'), 'confidence_norm'] = 'VALIDATED'
    df['clv_open_f'] = pd.to_numeric(df.get('CLV_OPEN_LINE', pd.Series(np.nan, index=idx)), errors='coerce')
    df['clv_latest_f'] = pd.to_numeric(df.get('CLV_LATEST_LINE', pd.Series(np.nan, index=idx)), errors='coerce')
    df['clv_edge'] = np.where(df['lean_norm'] == 'UNDER', df['clv_open_f'] - df['clv_latest_f'], df['clv_latest_f'] - df['clv_open_f'])
    df['clv_edge'] = pd.to_numeric(df['clv_edge'], errors='coerce')
    df['clv_bucket'] = np.where(df['clv_edge'].isna(), 'unknown', np.where(df['clv_edge'] > 0, 'positive', np.where(df['clv_edge'] < 0, 'negative', 'flat')))
    df['consensus_bucket'] = pd.to_numeric(df.get('CONSENSUS_COUNT', pd.Series(1, index=idx)), errors='coerce').fillna(1).astype(int)
    df['has_lineup_risk'] = df.get('injury_context', pd.Series('', index=idx)).fillna('').astype(str).str.strip().str.startswith('LINEUP RISK')
    df['date_parsed'] = pd.to_datetime(df.get('DATE', pd.Series('', index=idx)), errors='coerce')
    bad_dates = int(df['date_parsed'].isna().sum())
    if bad_dates:
        print(f"   ⚠️ Pick_Performance: {bad_dates} graded rows have unparseable DATE and count only all_time")
    df['day_of_week'] = df['date_parsed'].dt.strftime('%a').fillna('unknown')
    if 'RUN_NUMBER' not in df.columns:
        df['RUN_NUMBER'] = 'unknown'
    else:
        df['RUN_NUMBER'] = df['RUN_NUMBER'].replace('', np.nan).fillna('unknown').astype(str)
    return df

def pick_perf_metrics_row(df_slice, dim_type, dim_value, window_name):
    n_picks = len(df_slice)
    n_hits = int((df_slice['HIT'] == 'YES').sum())
    n_misses = int((df_slice['HIT'] == 'NO').sum())
    n_pushes = int((df_slice['HIT'] == 'PUSH').sum())
    n_dnp = int((df_slice['HIT'] == 'DNP').sum())
    n_decisive = n_picks - n_dnp
    hit_rate = pick_perf_rate(n_hits, n_misses)
    hit_rate_raw = n_hits / n_decisive if n_decisive > 0 else np.nan
    roi_flat = (n_hits * (100 / abs(PICK_PERF_STANDARD_ODDS)) - n_misses) * 100
    roi_per_pick = roi_flat / n_decisive if n_decisive > 0 else np.nan
    clv_numeric = df_slice.dropna(subset=['clv_edge'])
    clv_pos = df_slice[df_slice['clv_edge'] > 0]
    clv_neg = df_slice[df_slice['clv_edge'].notna() & (df_slice['clv_edge'] <= 0)]
    pos_hits = int((clv_pos['HIT'] == 'YES').sum())
    pos_misses = int((clv_pos['HIT'] == 'NO').sum())
    neg_hits = int((clv_neg['HIT'] == 'YES').sum())
    neg_misses = int((clv_neg['HIT'] == 'NO').sum())
    wilson_n = n_hits + n_misses
    wilson_p = n_hits / wilson_n if wilson_n > 0 else 0
    return {
        'DIMENSION_TYPE': dim_type,
        'DIMENSION_VALUE': '' if dim_value is None else str(dim_value),
        'TIME_WINDOW': window_name,
        'N_PICKS': n_picks,
        'N_PICKS_DECISIVE': n_decisive,
        'N_HITS': n_hits,
        'N_MISSES': n_misses,
        'N_PUSHES': n_pushes,
        'N_DNP': n_dnp,
        'HIT_RATE': round(hit_rate, 3) if pd.notna(hit_rate) else np.nan,
        'HIT_RATE_RAW': round(hit_rate_raw, 3) if pd.notna(hit_rate_raw) else np.nan,
        'PUSH_RATE': round(n_pushes / n_picks, 3) if n_picks else 0,
        'DNP_RATE': round(n_dnp / n_picks, 3) if n_picks else 0,
        'ROI_FLAT': round(roi_flat, 3),
        'ROI_PER_PICK': round(roi_per_pick, 3) if pd.notna(roi_per_pick) else np.nan,
        'AVG_CLV_EDGE': round(clv_numeric['clv_edge'].mean(), 3) if not clv_numeric.empty else np.nan,
        'CLV_POSITIVE_RATE': round((clv_numeric['clv_edge'] > 0).mean(), 3) if not clv_numeric.empty else np.nan,
        'CLV_POS_HIT_RATE': round(pick_perf_rate(pos_hits, pos_misses), 3) if pd.notna(pick_perf_rate(pos_hits, pos_misses)) else np.nan,
        'CLV_NEG_HIT_RATE': round(pick_perf_rate(neg_hits, neg_misses), 3) if pd.notna(pick_perf_rate(neg_hits, neg_misses)) else np.nan,
        'WILSON_LOWER_95': round(wilson_lower_bound(wilson_p, wilson_n), 3),
        'MIN_SAMPLE_FLAG': bool(n_decisive >= PICK_PERF_MIN_SAMPLE),
        'LAST_UPDATED': timestamp_est,
    }

def pick_perf_window_df(df, window_name, days, today):
    if days is None:
        return df.copy()
    cutoff = pd.Timestamp(today - timedelta(days=days))
    return df[df['date_parsed'].notna() & (df['date_parsed'] >= cutoff)].copy()

def build_pick_performance_metrics(df_all):
    df = pick_perf_prepare_df(df_all)
    if df.empty:
        return pd.DataFrame(columns=PICK_PERFORMANCE_COLUMNS), df
    today = datetime.now(pytz.timezone('US/Eastern')).date()
    rows = []
    for window_name, days in PICK_PERF_TIME_WINDOWS.items():
        win_df = pick_perf_window_df(df, window_name, days, today)
        if win_df.empty:
            continue
        rows.append(pick_perf_metrics_row(win_df, 'overall', '', window_name))
        for dim in PICK_PERF_DIMENSIONS:
            if dim not in win_df.columns:
                continue
            for dim_value, grp in win_df.groupby(dim, dropna=False):
                rows.append(pick_perf_metrics_row(grp, dim, dim_value, window_name))
    metrics_df = pd.DataFrame(rows, columns=PICK_PERFORMANCE_COLUMNS)
    if metrics_df.empty:
        return metrics_df, df
    window_order = {name: i for i, name in enumerate(PICK_PERF_TIME_WINDOWS.keys())}
    metrics_df['_window_order'] = metrics_df['TIME_WINDOW'].map(window_order).fillna(99)
    metrics_df = metrics_df.sort_values(['_window_order', 'DIMENSION_TYPE', 'WILSON_LOWER_95'], ascending=[True, True, False])
    metrics_df = metrics_df.drop(columns=['_window_order']).reset_index(drop=True)
    return metrics_df, df

def build_snapshot_rows(metrics_df, snapshot_date):
    if metrics_df is None or metrics_df.empty:
        return []
    rows = []
    snap = metrics_df[metrics_df['TIME_WINDOW'].isin(PICK_PERF_SNAPSHOT_WINDOWS)].copy()
    for _, row in snap.iterrows():
        dim_type = row['DIMENSION_TYPE']
        dim_val = str(row['DIMENSION_VALUE'])
        key_suffix = 'overall' if dim_type == 'overall' else f"{dim_type.replace('_norm', '')}.{dim_val}"
        rows.append({'SNAPSHOT_DATE': snapshot_date, 'METRIC_KEY': f"hit_rate.{key_suffix}", 'METRIC_VALUE': row['HIT_RATE'], 'N_PICKS': row['N_PICKS_DECISIVE'], 'TIME_WINDOW': row['TIME_WINDOW']})
        if dim_type in {'overall', 'confidence_norm'}:
            rows.append({'SNAPSHOT_DATE': snapshot_date, 'METRIC_KEY': f"roi_per_pick.{key_suffix}", 'METRIC_VALUE': row['ROI_PER_PICK'], 'N_PICKS': row['N_PICKS_DECISIVE'], 'TIME_WINDOW': row['TIME_WINDOW']})
    return rows

def snapshot_already_exists(spreadsheet, snapshot_date):
    try:
        ws_snap = spreadsheet.worksheet('Pick_Performance_Snapshots')
        rows = ws_snap.get_all_records()
    except gspread.exceptions.WorksheetNotFound:
        return False
    except Exception as e:
        print(f"   ⚠️ Snapshot check failed: {e}")
        return False
    if not rows:
        return False
    df_snap = pd.DataFrame(rows)
    return 'SNAPSHOT_DATE' in df_snap.columns and str(snapshot_date) in set(df_snap['SNAPSHOT_DATE'].astype(str))

def print_pick_performance_summary(metrics_df, sport):
    print("\n" + "=" * 60)
    print(f"📊 PICK PERFORMANCE — {sport}")
    print("=" * 60)
    if metrics_df is None or metrics_df.empty:
        print("   No graded picks to analyze.")
        print("=" * 60)
        return
    overall_all = metrics_df[(metrics_df['DIMENSION_TYPE'] == 'overall') & (metrics_df['TIME_WINDOW'] == 'all_time')]
    overall_30 = metrics_df[(metrics_df['DIMENSION_TYPE'] == 'overall') & (metrics_df['TIME_WINDOW'] == 'last_30d')]
    def fmt_row(df_row):
        if df_row.empty:
            return "n/a"
        r = df_row.iloc[0]
        return f"{r['HIT_RATE'] * 100:.1f}% (n={int(r['N_PICKS_DECISIVE'])})" if pd.notna(r['HIT_RATE']) else f"n/a (n={int(r['N_PICKS_DECISIVE'])})"
    print(f"   Overall:       {fmt_row(overall_all)}  |  last 30d: {fmt_row(overall_30)}")
    conf = metrics_df[(metrics_df['DIMENSION_TYPE'] == 'confidence_norm') & (metrics_df['TIME_WINDOW'] == 'all_time')]
    for tier in ['VALIDATED', 'SMASH', 'STRONG', 'LEAN']:
        row = conf[conf['DIMENSION_VALUE'] == tier]
        if not row.empty:
            print(f"   {tier:<14} {fmt_row(row)}")
    prop = metrics_df[(metrics_df['DIMENSION_TYPE'] == 'prop_type_norm') & (metrics_df['TIME_WINDOW'] == 'all_time') & (metrics_df['MIN_SAMPLE_FLAG'] == True)].copy()
    if not prop.empty:
        top = prop.sort_values('WILSON_LOWER_95', ascending=False).head(5)
        worst = prop.sort_values('WILSON_LOWER_95', ascending=True).head(5)
        print("\n   ✅ Top prop types (all-time, Wilson LB):")
        for _, r in top.iterrows():
            print(f"      {r['DIMENSION_VALUE']:<8} {r['HIT_RATE'] * 100:.1f}% (n={int(r['N_PICKS_DECISIVE'])})   LB={r['WILSON_LOWER_95']:.3f}")
        print("\n   🚨 Worst prop types (all-time, Wilson LB):")
        for _, r in worst.iterrows():
            print(f"      {r['DIMENSION_VALUE']:<8} {r['HIT_RATE'] * 100:.1f}% (n={int(r['N_PICKS_DECISIVE'])})   LB={r['WILSON_LOWER_95']:.3f}")
    alerts = []
    all_time = metrics_df[metrics_df['TIME_WINDOW'] == 'all_time']
    last_30 = metrics_df[metrics_df['TIME_WINDOW'] == 'last_30d']
    for _, r30 in last_30[last_30['MIN_SAMPLE_FLAG'] == True].iterrows():
        rall = all_time[
            (all_time['DIMENSION_TYPE'] == r30['DIMENSION_TYPE']) &
            (all_time['DIMENSION_VALUE'] == r30['DIMENSION_VALUE']) &
            (all_time['MIN_SAMPLE_FLAG'] == True)
        ]
        if rall.empty or pd.isna(r30['HIT_RATE']) or pd.isna(rall.iloc[0]['HIT_RATE']):
            continue
        delta = (r30['HIT_RATE'] - rall.iloc[0]['HIT_RATE']) * 100
        if abs(delta) >= PICK_PERF_DRIFT_ALERT_PP:
            label = r30['DIMENSION_TYPE'].replace('_norm', '')
            alerts.append(f"{label}.{r30['DIMENSION_VALUE']}: 30d={r30['HIT_RATE']*100:.1f}% vs all-time={rall.iloc[0]['HIT_RATE']*100:.1f}% (Δ={delta:+.1f}pp)")
    print("\n   ⚠️ Drift alerts:")
    if alerts:
        for alert in alerts[:8]:
            print(f"      {alert}")
    else:
        print("      none")
    print("=" * 60)

def run_pick_performance_section(df_all, sport):
    metrics_df, prepared_df = build_pick_performance_metrics(df_all)
    if prepared_df.empty:
        print("\n📊 Pick_Performance: no graded picks to analyze.")
        return
    wrote_perf = pick_perf_safe_upload(sh, 'Pick_Performance', metrics_df)
    snapshot_date = datetime.now(pytz.timezone('US/Eastern')).strftime('%Y-%m-%d')
    if snapshot_already_exists(sh, snapshot_date):
        print(f"   ⏭️  Pick_Performance_Snapshots: snapshot already exists for {snapshot_date}")
    else:
        snapshot_df = pd.DataFrame(build_snapshot_rows(metrics_df, snapshot_date), columns=PICK_PERFORMANCE_SNAPSHOT_COLUMNS)
        pick_perf_append_upload(sh, 'Pick_Performance_Snapshots', snapshot_df)
    print_pick_performance_summary(metrics_df, sport)
    if wrote_perf:
        print("   📈 Pick_Performance written.")

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
    raw_conf = df_all.get('confidence', pd.Series('', index=df_all.index)).fillna('').astype(str).str.upper()
    validated_mask = df_all.get('CONSENSUS_TAG', pd.Series('', index=df_all.index)).fillna('').astype(str).str.upper().eq('VALIDATED FALLBACK')
    for tier in ['VALIDATED', 'SMASH', 'STRONG', 'LEAN']:
        tier_mask = validated_mask if tier == 'VALIDATED' else (raw_conf.eq(tier) & ~validated_mask)
        tier_df = df_all[tier_mask]
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
    run_pick_performance_section(df_all, 'MLB')

print("\n⚾ Done! Run this every morning after games.")
print("💡 Tip: The cumulative hit rate feeds back into the +EV engine — the more data, the sharper the edge calculations.")

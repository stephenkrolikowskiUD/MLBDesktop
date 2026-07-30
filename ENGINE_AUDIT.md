# MLBDesktop Engine & Grader Audit — 2026-07-30

Read-only audit of `MLBEnginev5-4.py` (4,105 lines), `MLBGrader5-4.py` (1,185),
`run_logger.py` (85), plus the GitHub Actions workflows. Conducted the day before
the 7/31 feature freeze, ahead of the NFL build starting 8/1.

Findings marked ✅ were verified directly against the code. Findings marked 📋 were
reported by an auditor and read as credible but were not independently re-verified.

---

## 1. Operational context (important, easy to forget)

The engine is **not** a manual script. It runs on GitHub Actions:

| Workflow | Cron (UTC) | Eastern | Frequency |
|---|---|---|---|
| `mlb-engine.yml` | `45 5`, `0 13`, `45 15`, `0 21` | 1:45a, 9:00a, 11:45a, 5:00p | 4×/day |
| `mlb-grader.yml` | `30 6`, `0 10` | 2:30a, 6:00a | 2×/day |

**Freezing the repo does not stop the engine.** It keeps writing to the Google Sheet
4× a day indefinitely, and keeps spending metered Odds API credits.

---

## 2. Fixed today

### ✅ `.gitignore` added; tracked build artifact removed
There was **no `.gitignore` in the repo at all**, and
`__pycache__/MLBGrader5-4.cpython-314.pyc` was tracked in git. Added a `.gitignore`
covering `__pycache__/`, `*.pyc`, `.env*`, `*credential*.json`,
`*service-account*.json`, `*.pem`, `*.key`, and the engine's local cache dir;
untracked the `.pyc` (left on disk).

Secrets themselves are **clean** — see §5.

### ✅ Season history can no longer be destroyed by a transient Sheets error
`load_existing_log_sheet` returned an **identical empty DataFrame** for "read
failed" and "sheet is empty":

```python
except Exception:
    return pd.DataFrame(columns=keep_cols)   # read FAILED
if not rows:
    return pd.DataFrame(columns=keep_cols)   # sheet EMPTY — indistinguishable
```

Failure chain: a 429/500 on the read → engine believes there is no history → full
re-fetch of ~500 players → any player whose MLB API call times out contributes
zero rows → write phase does `ws.clear()` + rewrite → **that player's entire
season is permanently gone.** There is no other copy. With 15 concurrent workers,
partial timeouts are the expected failure mode, not an exotic one.

Now: a genuinely missing tab returns empty (correct for a first run); any other
error raises `RuntimeError`. All three call sites are top-level and outside any
`try`, so the run dies **before** the write phase — nothing is written, nothing is
destroyed, and the next scheduled run picks up. Same guard applied to
`load_existing_daily_picks`, where a read failure instead reset `today_run_number`
to 1 and emptied the dedupe set, duplicating picks already appended.

### ✅ Failed fetches are no longer cached
`cached_odds_fetch` wrote whatever `fetch_fn()` returned to disk — and `_fetch`
returns `[]` on any non-200 or exception. That `[]` was cached for the full
900-second TTL, so **re-running to recover from a transient odds outage replayed
the empty payload from cache without touching the API.** Now only non-empty
payloads are cached, the cache write is atomic (temp + `os.replace`), and a
truncated/corrupt cache file is treated as a miss instead of raising
`JSONDecodeError` mid-run.

### ✅ `Tonights_Schedule` now carries an explicit Eastern `game_date`
The tab had **no date column at all** — the only date-bearing field was
`game_time`, a raw UTC instant (`2026-07-30T23:05:00Z`). For any game after 8pm ET
the UTC *date* is already tomorrow, so every consumer had to convert before
comparing. Added `game_date` (the Eastern slate date the run actually requested),
so a stale tab is self-identifying.

Frontend interaction checked: `entryRowDate` looks for `game_date` (4th in its key
list) before `game_time` (12th), while `entryStartValue` still prefers `game_time`
— so start times keep working and date-matching becomes more direct.

### ✅ Grader: wrong-player / wrong-game grading closed off
`box_lookup` was keyed on `(player_name, game_date)`, which is **not unique**:

1. MLB has multiple concurrently active players sharing a name (Luis Garcia, Will
   Smith, Josh Bell). Both produce rows for the same key; a plain dict assignment
   keeps whichever was read last → the pick is graded against **a different
   human's box score**.
2. Doubleheaders put two games on one date → graded against the wrong game.
3. The pitcher pass ran `.update()` with an unprefixed `'SO'` (comment: *"Also
   store without prefix"*), **clobbering the batter's strikeout count**. Since
   `normalize_prop_metric` maps `BATTER_SO → SO` while pitcher props use `P_SO`, a
   batter "SO UNDER 1.5" could be graded against strikeouts *thrown* (e.g. 8),
   flipping a win to a loss. Same for `DK_FP` / `UD_FP`.
   The guard above it (`if key not in box_lookup: # Don't overwrite batter data`)
   only prevented replacing the *whole dict* — the `.update()` still clobbered
   those four fields.

Fixes: the unprefixed `'SO'` write is **removed entirely** (no pitcher prop reads
it); `DK_FP`/`UD_FP` use `setdefault` so a batter's value is never overwritten;
and identity is now tracked per key (`player_id` + `game_pk` sets). If a key maps
to more than one player or game, the pick is **left ungraded with a warning**
rather than graded against a guess.

> **Behavior change to expect:** doubleheader picks and duplicate-name picks will
> now appear as *ungraded* (blank `HIT`) with a `⚠️` line in the grader log and an
> "Ambiguous" count in the summary, where previously they were silently graded —
> possibly wrongly. Ungraded is recoverable; a wrong record is not.

### ✅ Grader: `lean` is stripped
`lean = (...).upper()` with no `.strip()`, and `grade_pick` treats anything that is
not `UNDER`/`FADE` as `OVER`. A cell containing `"Under "` graded as OVER —
**inverting the result.** Added `.strip()`.
*Severity note:* the engine writes `lean` via `.strip().upper()` at all six write
sites, so clean values reach the sheet. This was latent, only reachable by a manual
sheet edit — **records were not being inverted in practice.**

---

## 3. QA checklist for 7/31

Frontend QA won't surface any of the above — every one of these fails silently and
prints success. Things worth actually checking:

- [ ] **Trigger one manual engine run** (`workflow_dispatch`) and read the log for:
      `⏭️ … No data — skipped` (a skipped tab means that tab still holds old data),
      and any `⚠️ Cache unreadable` / `empty result — NOT caching` lines.
- [ ] **Confirm `Tonights_Schedule` has the new `game_date` column** and that it
      equals today's Eastern date.
- [ ] **Check game start times render** on Shortlist / Picks / dashboard hero
      (regression check on yesterday's `gameTimeLookup` fix).
- [ ] **Trigger one manual grader run** and check the summary for a non-zero
      `⚠️ Ambiguous` count — if it's high, duplicate names are more common than
      expected and deserve a proper `player_id` join.
- [ ] **Spot-check 3–5 recently graded picks** in `Daily_Picks` against real box
      scores, ideally including a pitcher and a batter with a common name.
- [ ] **Verify `git status` is clean** and no credentials-shaped file is untracked.

---

## 4. Open — deliberately NOT fixed before the freeze

### Non-atomic writes leave tabs blank ✅
`safe_upload` does `ws.clear()` and *then* `set_with_dataframe` as a separate API
call. If the write fails and retries exhaust, the function returns normally with
the worksheet **emptied** — worse than stale, since there is no fallback content.
Proper fix is write-to-staging-tab-then-swap, or a read-back verification. Too
invasive for the day before a freeze.

### No transactional guard across the write phase ✅
20 tabs written independently in a loop with `time.sleep(2)` between. Any failure
leaves a **mixed-date sheet** (some tabs today, some last week) with no marker.
Worse, `validate_sheet_schema` is called *outside* the retry `try` and **raises**,
so a schema violation on tab 5 of 20 kills the process and tabs 6-20 never get
written. The final summary prints "✅ COMPLETE" regardless.

### An odds outage republishes yesterday's betting lines as live ✅
`df_props` is only assigned after the full event loop; any mid-loop exception
discards everything collected and reverts to empty → `safe_upload` skips →
`DK_Player_Props` keeps yesterday's lines and the run reports success. For a
betting dashboard this is the most damaging silent outcome. (Partly mitigated by
the cache fix, which at least lets a re-run reach the API.)

### Silent defaults that fabricate data ✅
- `get_pitcher_hand` returns `'R'` on any exception → a lefty starter looks
  right-handed → ~9-13 hitters get the wrong handedness split written to the sheet
  *and fed to Gemini as fact*, indistinguishable from a real RHP.
- Missing `Team_Rankings` → every `*_OPP_ADJ` becomes a legitimate-looking `0.0`
  via `.fillna(0)`. Nothing distinguishes "opponent is league-average" from "we had
  no opponent data."
- Statcast failure re-stamps stale cached rows with a fresh `LAST_UPDATED` — the
  freshness column is actively falsified.

### Grader statistics quirks 📋
- `HIT_RATE_RAW` is byte-identical to `HIT_RATE` ✅ (`pick_perf_rate(h,m) =
  h/(h+m)` and `n_decisive = h+m`). Presumably intended a push-inclusive
  denominator; any dashboard comparison of the two shows zero push impact, which
  is false whenever pushes exist.
- `ROI_PER_PICK` is percent-scaled while `ACTUAL_ROI_PER_PICK` is a unit fraction ✅.
  Not currently visible on the Overall card (the frontend happens to use
  `statSigned` for one and `statRoiPct` for the other), but both land in the same
  `METRIC_VALUE` column of `Pick_Performance_Snapshots`, so any time-series over
  that column mixes scales.
- `Pick_Performance_Snapshots` is append-only and its dedupe guard `return False`s
  on *any* exception → a transient Sheets error during the check appends the day's
  snapshot twice, double-counting the series 📋.
- Grade row targeting uses the positional index of a read taken ~400 lines earlier;
  any manual row insert/sort in `Daily_Picks` between read and write puts grades on
  the **wrong picks** 📋.
- No "game is final" check anywhere — a suspended/rain-shortened game can be graded
  off a partial line, and because `HIT` is then non-blank it is never revisited 📋.

### `run_logger.py` reports OK for crashed grader runs 📋
`fail()` is never called from the grader (no try/except around its main flow) and
the `FAIL` fallback is gated on `kind == "engine"`, so an exception still fires
`atexit` and appends `status = OK`. The heartbeat cannot distinguish a clean run
from a half-completed one — exactly the case where the dashboard is stale.

---

## 5. Secrets: clean ✅

- **No hardcoded keys, tokens, or service-account JSON** in the working tree or in
  the **full git history** (scanned for `AIza…`, `sk-…`, `BEGIN … PRIVATE KEY`,
  `"private_key"`): zero hits.
- `load_secret` is a good pattern: env → Colab `userdata` → interactive prompt, and
  it **explicitly refuses to prompt** under `GITHUB_ACTIONS` or a non-TTY.
- Google credentials are parsed in memory from env via
  `Credentials.from_service_account_info` — no file path, no embedded blob.
- Workflows inject all four secrets via `${{ secrets.* }}`.

**One design-level exposure, by choice:** the Sheet ID is public in `app.js`
because the frontend reads the unauthenticated `gviz` endpoint. That requires the
**entire workbook** to be world-readable — not just the tabs the dashboard renders.
`Run_Log`, `Pick_Performance_Snapshots`, and all raw logs are publicly readable.
Not a leak, but *which tabs live in the public workbook* is the real access-control
boundary. For NFL, consider splitting a public read-only presentation workbook from
a private working workbook.

---

## 6. Decision needed: the off-season ✅

`derive_mlb_season_context` + a hardcoded `gameType=R` in the schedule URL produce:

| Window | Behavior |
|---|---|
| **October (postseason)** | `gameType=R` → **0 games** |
| **Nov → Feb** | `in_season` is trivially True (Jan 2027 > Mar 25 2026) → real date → **0 games** |
| **Mar 1–24** | `in_season` flips False → `schedule_date = "{season}-07-02"` → a **future date in the current year**; would publish a July slate as "tonight" |

Combined with skip-if-empty writes, from early October onward the "tonight" tabs
**freeze at the last day with games, permanently**, while the log tabs keep getting
rewritten with a fresh `LAST_UPDATED` — a stale dashboard that looks current. Plus
4×/day of metered Odds API credits spent on a frozen app all winter.

**Options:** disable both workflows after the World Series; cut to 1×/day; or add an
explicit off-season guard that fails loudly instead of skipping writes.

---

## 7. Carry-forward for the NFL build

Architectural findings — all verified, none worth fixing in a frozen MLB codebase,
all worth designing around from day one:

1. **It is a script, not a program.** No `main()`, no `__main__` guard, **409
   top-level statements**; the whole file executes on import. Nothing can be
   imported, unit-tested, or re-run in isolation — if the odds fetch fails at line
   1878, the only recovery is re-running all 4,105 lines and re-spending every API
   call. Start NFL with `main()` + staged functions taking explicit arguments.
2. **`generate_gemini_picks()` is 1,063 lines** with **zero declared parameters**
   and **18 implicit module-level globals** (including 5 DataFrames), defined at
   line 2332 and called at 3870 — 1,538 lines apart. Never let a function grow an
   invisible input surface.
3. **Section numbering has lost sync with execution order** (`1…10, 10.6, 12, 13,
   14, 15, 10.75, 11`) — the fingerprint of append-only editing, because inserting
   anywhere felt unsafe.
4. **Extract a `sports_common/` module**: auth + `load_secret`, one `safe_upload`
   (with retry *and* schema validation), name/metric/confidence/lean
   normalization, odds math, and a per-sport `scoring.py`. The copy-paste has
   already cost correctness — the engine's pitcher `DK_FP` includes `HBP * -0.6`
   and the grader's re-implementation omits it 📋. The grader also hard-depends on
   tab names the engine owns, with no shared declaration.
5. **Hoist sport identity into config**: `SPORT_LABEL`, API base + `sportId`, odds
   sport key, market/prop maps, team abbreviations, venue tables, season rollover,
   tab names.
6. **Immediate NFL blocker** ✅ — these two dicts subscript directly at module
   level with no `NFL` key, so `SPORT_LABEL = "NFL"` **raises `KeyError` at import**
   before anything runs:
   - `DEFAULT_QUOTA_FLOOR_THIS_SPORT` (`MLBEnginev5-4.py:74`)
   - `CACHE_TTL_SECONDS` (`MLBEnginev5-4.py:84`)
7. **Don't build a fetch stage until something consumes it.** 6 of 20 tabs the
   engine writes are never read by the dashboard ✅ — `LHP_RHP_Splits`,
   `Statcast_Daily`, `Batter_Statcast`, `Pitcher_Statcast`, `Odds`, `Teams` — and
   two of those (§5 LHP/RHP, §6.5 Statcast) are expensive parallel fetch sections.
   That's ~30% of the write phase spending quota and runtime on unread data.
8. **Dead code to not carry over** ✅: `refresh_clv_frame` (40 lines, never called),
   `implied_to_american`, `QUOTA_FLOOR_GLOBAL`, `BEST_BOOK_TIE_BREAK`, grader's
   `SNAPSHOT_DATE`/`SHEET_NAME`. Also `promote_consensus_confidence` takes
   `consensus_count` and ignores it — a no-op left in place.

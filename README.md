# MLB DFS Dashboard

A personal MLB player-prop research dashboard for turning daily baseball data, live prop markets, AI picks, and grader feedback into one mobile-first view.

This repo is the GitHub Pages frontend for the MLB system. The engine writes data to Google Sheets, the grader closes the loop after games, and the dashboard reads the workbook through public Sheets CSV endpoints.

## Status: Frozen (2026-07-31)

MLB is in feature freeze ahead of the NFL build. The engine and grader keep running on their existing GitHub Actions schedule — this is a freeze on *changes*, not on operation. Before freezing, the codebase went through a first structural audit; see [`ENGINE_AUDIT.md`](ENGINE_AUDIT.md) for the full findings, what was fixed, and the open items carried forward (deliberately, not by omission) to the NFL build.

Scheduled runs will be retired the same way NBA's and World Cup's were once the season ends in November — delete the `schedule:` block in each workflow, keep `workflow_dispatch:` for manual runs.

## What It Does

- Shows batter and pitcher context for tonight's slate: game logs, splits, Statcast signals, weather, probable pitchers, venue notes, and props.
- Surfaces AI picks across nine boards: Tonight's Shortlist, Model Picks, Best Bets, Smart Slips, Streaks, Dingers, Ks, Draft Board, and Prop Explorer.
- Displays Pick Performance analytics so confidence tiers, prop types, leans, CLV buckets, and ROI can be judged from graded history.
- Surfaces multi-book best-price routing from DraftKings, FanDuel, BetMGM, and ESPN BET when the engine has current prop data.
- Supports a Game Entry flow for building a quick single-game entry from available props and historical prop-type performance.

## How It Works

1. `MLBEnginev5-4.py` pulls MLB data, Statcast, weather, odds, live props, and Gemini picks, then writes dashboard tabs to the MLB Google Sheet.
2. The dashboard (`index.html` + `styles.css` + `app.js`) loads those tabs through Google Sheets CSV endpoints. No build step — it's plain HTML/CSS/JS, deployed as-is via GitHub Pages.
3. `MLBGrader5-4.py` grades completed picks and writes `HIT`, `ACTUAL_STAT`, and `RESULT` back to `Daily_Picks`.
4. Pick Performance turns that graded history into the Stats tab.

`app.js` is organized around named per-view renderers (one function per board/page) rather than one monolithic render routine — see `ENGINE_AUDIT.md` for why that mattered and how it was done.

## Key Tabs

- **Dash**: selected-player context, matchup, props, splits, weather, and logs.
- **Picks**: nine sub-views — Shortlist, Model Picks, Best Bets, Smart Slips, Streaks, Dingers, Ks, Draft Board, Prop Explorer.
- **Leaders**: daily statistical leaderboards.
- **Game Entry**: single-game auto-entry builder.
- **Lookup**: MLB player lookup and deeper player context.
- **Stats**: Pick Performance hit rate, ROI, CLV, confidence tiers, prop types, and drift checks.
- **Info**: method notes and glossary.

## Run Mode

MLB is automated through GitHub Actions: the engine runs four times a day during active slates, and the grader runs twice each morning to close the feedback loop on the prior night's picks.

## Data Sources

- Google Sheets workbook: `1AAwSwFCGIqS6JGdYTdkSau91BtnM_sMdWl2By5A9nFQ`
- MLB Stats API
- Baseball Savant / Statcast
- The Odds API
- OpenWeather
- Gemini output from the engine

## Shipped Features

Started as experiments, now stable parts of the product:

- Multi-book line shopping and best-book routing.
- Pick Performance driven prompt tuning.
- MLB-first Soft DK line detection, used to flag possible stale DraftKings prices against the current market.
- Game Entry, a fast single-game parlay builder.

## Important Notes

- Keep the dashboard entry point named `index.html`; GitHub Pages depends on it.
- No private API keys live in this repo, in the HTML/JS, or anywhere in git history (audited 2026-07-31).
- Public Sheet IDs are identifiers, not secrets — but because the dashboard reads the workbook through an unauthenticated public endpoint, the *entire* workbook is world-readable, not just the tabs the dashboard renders. Treat "which tabs live in this workbook" as the real access-control boundary.
- This is a personal research tool, not betting advice.

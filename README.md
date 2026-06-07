# MLB DFS Dashboard

A personal MLB player-prop research dashboard for turning daily baseball data, live prop markets, AI picks, and grader feedback into one mobile-first view.

This repo is the GitHub Pages frontend for the MLB system. The engine writes data to Google Sheets, the grader closes the loop after games, and `index.html` reads the workbook through public Sheets CSV endpoints.

## What It Does

- Shows batter and pitcher context for tonight's slate: game logs, splits, Statcast signals, weather, probable pitchers, venue notes, and props.
- Surfaces AI picks, smart slips, streaks, due spots, dinger-style views, parlay helpers, and prop browsing.
- Displays Pick Performance analytics so confidence tiers, prop types, leans, CLV buckets, and ROI can be judged from graded history.
- Surfaces multi-book best-price routing from DraftKings, FanDuel, BetMGM, and ESPN BET when the engine has current prop data.
- Supports a Game Entry flow for building a quick single-game entry from available props and historical prop-type performance.

## How It Works

1. `MLBEnginev5-4.py` pulls MLB data, Statcast, weather, odds, live props, and Gemini picks.
2. The engine writes dashboard tabs to the MLB Google Sheet.
3. `index.html` loads those tabs through Google Sheets CSV endpoints.
4. `MLBGrader5-4.py` grades completed picks and writes `HIT`, `ACTUAL_STAT`, and `RESULT` back to `Daily_Picks`.
5. Pick Performance turns that graded history into the Stats tab.

## Key Tabs

- **Dash**: selected-player context, matchup, props, splits, weather, and logs.
- **Log**: game-log focused view.
- **Picks**: AI picks, best bets, slips, streaks, due spots, props, and related boards.
- **Stats**: Pick Performance hit rate, ROI, CLV, confidence tiers, prop types, and drift checks.
- **Game Entry**: single-game auto-entry builder.
- **Lookup**: MLB player lookup and deeper player context.
- **Info**: method notes and glossary.

## Run Mode

MLB is automated through GitHub Actions. The engine runs during the day, and the grader runs after games to update the feedback loop.

## Data Sources

- Google Sheets workbook: `1AAwSwFCGIqS6JGdYTdkSau91BtnM_sMdWl2By5A9nFQ`
- MLB Stats API
- Baseball Savant / Statcast
- The Odds API
- OpenWeather
- Gemini output from the engine

## Current Experiments

- Multi-book line shopping and best-book routing.
- Pick Performance driven prompt tuning.
- MLB-first Soft DK line detection, used to flag possible stale DraftKings prices against the current market.
- Game Entry, a fast single-game parlay builder.

## Important Notes

- Keep the dashboard file named `index.html`; GitHub Pages depends on it.
- No private API keys live in this repo or in the HTML.
- Public Sheet IDs are identifiers, not secrets.
- This is a personal research tool, not betting advice.

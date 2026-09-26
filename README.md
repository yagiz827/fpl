# FPL Draft Optimization Engine

A GPU-accelerated engine for **Fantasy Premier League draft leagues**. It pulls live data from the official FPL API, projects each player's expected points with XGBoost, ranks players by **Value Over Replacement Player (VORP)**, and runs thousands of Monte Carlo drafts to tell you who to pick and when.

## Features

- **Async data ingestion.** An `aiohttp` client fetches bootstrap, per-player history and fixture data from the FPL API with bounded concurrency, request throttling and retries.
- **Feature engineering.** Rolling-window form features (3 and 5 gameweeks), non-penalty xG estimates and team expected-goals-conceded tables.
- **Projection models.** Two XGBoost models, trained on CUDA:
  - `MinutesModel` predicts expected minutes.
  - `PointsModel` predicts expected FPL points per gameweek, blended with a season-average prior and capped with per-position ceilings.
- **VORP rankings.** Each projection is compared against a replacement-level baseline per position, sized for a 16-team league (GKP 16th, DEF 48th, MID 64th, FWD 32nd).
- **Monte Carlo draft simulator.** Runs 10,000 GPU-parallel snake drafts (CuPy) with ADP-weighted opponent picks, and gives the probability that each watch-list player is still available at your next pick.
- **Waiver engine.** Flags droppable players whose expected minutes are falling, and recommends free-agent pickups and streaming defenders based on fixture difficulty over the next few gameweeks.

## Project structure

| File | Purpose |
|---|---|
| `main.py` | CLI entry point (`pipeline`, `draft`, `waiver`) |
| `config.py` | All tunable parameters: league size, scoring rules, model hyperparameters, simulation settings |
| `ingest.py` | Async FPL API client, prior-season loader and feature engineering |
| `models.py` | XGBoost minutes/points models, projection engine, VORP |
| `draft_simulator.py` | GPU Monte Carlo snake-draft simulator |
| `optimization.py` | Weekly waiver and streaming recommendations |

## Requirements

- Python 3.10+
- An NVIDIA GPU with CUDA 12.x. XGBoost trains with `device="cuda"`, and the draft simulator runs on CuPy.

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Full pipeline: ingest -> train -> project -> top-30 VORP rankings
python main.py pipeline

# Monte Carlo draft simulation from seat #3 in the draft order
python main.py draft --pick-position 3

# Waiver analysis for gameweek 12, given your roster's FPL player IDs
python main.py waiver --current-gw 12 --roster-ids 10,44,87,102
```

## Optional: prior-season data

Adding last season's gameweek data gives the models more history to train on. Put it in `./data/2024-25/`, laid out as:

```
data/2024-25/cleaned_players.csv
data/2024-25/players/<player>/gw.csv
```

To keep the data somewhere else, set the `FPL_DATA_DIR` environment variable. If there's no prior-season data, the engine trains on the current season only.

## Configuration

League settings (number of managers, squad size, position limits), scoring rules, XGBoost hyperparameters, the number of simulations and the waiver thresholds all live in `config.py`.
